using Finch
using BenchmarkTools


function split_rows_grain_mul(grain_size)
        return (y, A, x) -> split_rows_grain_helper(grain_size, y, A, x)
end

function split_rows_grain_helper(grain_size, y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (grain_size, _y, _A, _x) = $(grain_size, _y, _A, _x)
                (num_rows, _) = size(_A)
                cap_size = div(num_rows, grain_size) * grain_size
                @finch mode = :fast begin
                        _y .= 0
                        for group = parallel(1:grain_size:cap_size)
                                for i = group:group+grain_size-1
                                        for j = _
                                                _y[i] += _A[i, j] * _x[j]

                                        end
                                end
                        end

                        for i = parallel(cap_size+1:num_rows)
                                for j = _
                                        _y[i] += _A[i, j] * _x[j]

                                end
                        end

                end
        end samples = 1 evals = 1
        return (; time=time, y=_y)
end
