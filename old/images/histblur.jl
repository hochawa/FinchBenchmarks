function histblur_opencv_kernel(img, channels, mask, bins, regions)
    img = OpenCV.blur(reshape(img, 1, size(img)...), blur_opencv_kernelSize; borderType=OpenCV.BORDER_CONSTANT)
    imgs = Vector{OpenCV.InputArray}([img])
    OpenCV.calcHist(imgs, channels, mask, bins, regions)
end
function histblur_opencv((img, mask),)
    # step 1: gray scale the image
    # vector of images - don't ask
    mask = reshape(mask, 1, size(mask)...)
    result = Ref{Any}()
    channels = Vector{Int32}([0])
    bins = Vector{Int32}([16])
    regions = Vector{Float32}([0, 256])
    time = @belapsed histblur_opencv_kernel($img, $channels, $mask, $bins, $regions)
    result = histblur_opencv_kernel(img, channels, mask, bins, regions)
    (;time=time, output=map(x->round(Int, x), reshape(Array(result), :)), mem = summarysize(img), nnz = length(img))
end

const blur_opencv_kernelSize = OpenCV.Size(Int32(3), Int32(3))
blur_opencv_kernel(image) = begin
    OpenCV.blur(reshape(image, 1, size(image)...), blur_opencv_kernelSize; borderType=OpenCV.BORDER_CONSTANT)
end

function blur_opencv((image, mask),)
    time = @belapsed blur_opencv_kernel($image)
    blurry = blur_opencv_kernel(image)
    (;time=time, output=reshape(Array(blurry), size(image)) .* mask, mem = summarysize(image), nnz = length(image))
end

function histblur_finch((img, mask),)
    bins = Tensor(Dense(Element(0)), undef, 16)
    img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
    mask = Tensor(Dense(Dense(Element(false))), mask)
    tmp = Tensor(Dense(Element(0)))
    time = @belapsed histblur_finch_kernel($bins, $img, $tmp, $mask)
    result = histblur_finch_kernel(bins, img, tmp, mask)
    (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
end

function histblur_finch_rle((img, mask),)
    bins = Tensor(Dense(Element(0)), undef, 16)
    img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
    mask = Tensor(Dense(SparseRLE(Pattern())), mask .!= 0)
    tmp = Tensor(Dense(Element(0)))
    time = @belapsed histblur_finch_kernel($bins, $img, $tmp, $mask)
    result = histblur_finch_kernel(bins, img, tmp, mask)
    (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
end

function blur_finch((image, mask),)
    output = Tensor(Dense(Dense(Element(UInt8(0)))))
    image = Tensor(Dense(Dense(Element(UInt8(0)))), image)
    tmp = Tensor(Dense(Element(UInt(0))))
    mask = Tensor(Dense(Dense(Element(false))), mask)
    time = @belapsed blur_finch_kernel($output, $image, $tmp, $mask)
    blurry = blur_finch_kernel(output, image, tmp, mask).output
    (;time=time, output=blurry, mem = summarysize(image), nnz = countstored(image))
end

function blur_finch_rle((image, mask),)
    output = Tensor(Dense(Dense(Element(UInt8(0)))))
    image = Tensor(Dense(Dense(Element(UInt8(0)))), image)
    tmp = Tensor(Dense(Element(UInt(0))))
    mask = Tensor(Dense(SparseRLE(Pattern())), mask .!= 0)
    time = @belapsed blur_finch_kernel($output, $image, $tmp, $mask)
    blurry = blur_finch_kernel(output, image, tmp, mask).output
    (;time=time, output=blurry, mem = summarysize(image), nnz = countstored(image))
end

    #=
    for (output, input, tmp, mask) in [
        [
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Element(UInt(0)))),
            Tensor(Dense(SparseRLE(Pattern()))),
        ],
        [
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Element(UInt(0)))),
            Tensor(Dense(Dense(Element(false)))),
        ],
    ]
        push!(kernels, Finch.@finch_kernel function blur_finch_kernel(output, input, tmp, mask)
            output .= false
            for y = _
                tmp .= false
                for x = _
                    if coalesce(mask[~(x - 1), y], false) || mask[x, y] || coalesce(mask[~(x + 1), y], false)
                        tmp[x] = UInt(coalesce(input[x, ~(y-1)], 0)) + UInt(input[x, y]) + UInt(coalesce(input[x, ~(y+1)], 0))
                    end
                end
                for x = _
                    if mask[x, y]
                        output[x, y] = unsafe_trunc(UInt8, round((UInt(coalesce(tmp[~(x-1)], 0)) + tmp[x] + UInt(coalesce(tmp[~(x+1)], 0)))/9))
                    end
                end
            end
            return output
        end)
    end

    for (bins, img, tmp, mask) in [
        [
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(false))))
        ],
        [
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Element(0)))
            Tensor(Dense(SparseRLE(Pattern())))
        ],
    ]
        push!(kernels, @finch_kernel function histblur_finch_kernel(bins, img, tmp, mask)
            bins .= 0 
            for y = _
                tmp .= false
                for x = _
                    tmp[x] = UInt(coalesce(img[x, ~(y-1)], 0)) + UInt(img[x, y]) + UInt(coalesce(img[x, ~(y+1)], 0))
                end
                for x = _
                    if mask[x, y]
                        let t = unsafe_trunc(UInt8, round((UInt(coalesce(tmp[~(x-1)], 0)) + tmp[x] + UInt(coalesce(tmp[~(x+1)], 0)))/9))
                            bins[div(t, 16) + 1] += 1
                        end
                    end
                end
            end
            return bins
        end)
    end
    =#
