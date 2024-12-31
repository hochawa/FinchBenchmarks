y = Tensor(Dense(Element(0.0)))
x = Tensor(Dense(Element(0.0)))
y_j = Scalar(0.0)

kernels = []

for (A, diag) in [
    (Tensor(Dense(SparseList(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparseList(Pattern()))), Tensor(Dense(Element(false)))),
    (Tensor(Dense(SparseVBLLevel(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparseBand(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparsePoint(Element(0.0)))), Tensor(Dense(Element(0.0)))),
    (Tensor(Dense(SparsePoint(Pattern()))), Tensor(Dense(Element(false))))
]
    eval(@finch_kernel mode=:fast function spmv_finch_sym_kernel_helper(y, A, x, diag, y_j)
        y .= 0
        for j = _
            let x_j = x[j]
                y_j .= 0
                for i = _
                    let A_ij = A[i, j]
                        y[i] += x_j * A_ij
                        y_j[] += A_ij * x[i]
                    end
                end
                y[j] += y_j[] + diag[j] * x_j
            end
        end
        return y
    end)

    eval(@finch_kernel mode=:fast function spmv_finch_col_maj_kernel_helper(y, A, x
        y .= 0
        for j = _, i = _
            y[i] += A[i, j] * x[j]
        end
        return y
    end))

    eval(@finch_kernel mode=:fast function spmv_finch_row_maj_kernel_helper(y, A, x)
        y .= 0
        for j = _, i = _
            y[j] += A[i, j] * x[i]
        end
        return y
    end)
end


function ssymv_finch_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)

    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end


using Finch
using BenchmarkTools

function ssymv_finch_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            diag_lvl_val = diag_lvl.lvl.val
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                diag_lvl_q = (1 - 1) * diag_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                diag_lvl_2_val = diag_lvl_val[diag_lvl_q]
                y_j_val = 0
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            y_lvl_q = (1 - 1) * A_lvl.shape + A_lvl_2_i
                            x_lvl_q_2 = (1 - 1) * x_lvl.shape + A_lvl_2_i
                            x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                            y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                            y_j_val = A_lvl_3_val * x_lvl_2_val_2 + y_j_val
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                y_lvl_q = (1 - 1) * A_lvl.shape + phase_stop_3
                                x_lvl_q_2 = (1 - 1) * x_lvl.shape + phase_stop_3
                                x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                y_j_val += A_lvl_3_val * x_lvl_2_val_3
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = y_j_val + y_lvl_val[y_lvl_q_2] + x_lvl_2_val * diag_lvl_2_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)

    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function ssymv_finch_band_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseBandLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            diag_lvl_val = diag_lvl.lvl.val
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                diag_lvl_q = (1 - 1) * diag_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                diag_lvl_2_val = diag_lvl_val[diag_lvl_q]
                y_j_val = 0
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1] - 1
                if A_lvl_2_r <= A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r]
                    A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                    A_lvl_2_i_2 = A_lvl_2_i1 - ((A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r]) - 1)
                    A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i1) - 1
                else
                    A_lvl_2_i_2 = 1
                    A_lvl_2_i1 = 0
                end
                phase_start_2 = max(1, A_lvl_2_i_2)
                phase_stop_2 = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop_2 >= phase_start_2
                    for i_8 = phase_start_2:phase_stop_2
                        A_lvl_2_q = A_lvl_2_q_ofs + i_8
                        y_lvl_q = (1 - 1) * A_lvl.shape + i_8
                        x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_8
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                        y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                        y_j_val = A_lvl_3_val * x_lvl_2_val_2 + y_j_val
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = y_j_val + y_lvl_val[y_lvl_q_2] + x_lvl_2_val * diag_lvl_2_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_band_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_band_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_band(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_band_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_band_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseBandLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1] - 1
                if A_lvl_2_r <= A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r]
                    A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                    A_lvl_2_i_2 = A_lvl_2_i1 - ((A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r]) - 1)
                    A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i1) - 1
                else
                    A_lvl_2_i_2 = 1
                    A_lvl_2_i1 = 0
                end
                phase_start_2 = max(1, A_lvl_2_i_2)
                phase_stop_2 = min(A_lvl_2.shape, A_lvl_2_i1)
                if phase_stop_2 >= phase_start_2
                    for i_6 = phase_start_2:phase_stop_2
                        y_lvl_q = (1 - 1) * A_lvl_2.shape + i_6
                        A_lvl_2_q = A_lvl_2_q_ofs + i_6
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_band_kernel(y, A, x)
    spmv_finch_band_kernel_helper(y, A, x)
    y
end

function spmv_finch_band_unsym(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_band_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
endusing Finch
using BenchmarkTools

function spmv_finch_band_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_band::Tensor{DenseLevel{Int64, SparseBandLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_band_lvl = A_band.lvl
            A_band_lvl_2 = A_band_lvl.lvl
            A_band_lvl_ptr = A_band_lvl_2.ptr
            A_band_lvl_idx = A_band_lvl_2.idx
            A_band_lvl_ofs = A_band_lvl_2.ofs
            A_band_lvl_2_val = A_band_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_band_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_band_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_band_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_band_lvl.shape)
            for j_4 = 1:A_band_lvl.shape
                y_lvl_q = (1 - 1) * A_band_lvl.shape + j_4
                A_band_lvl_q = (1 - 1) * A_band_lvl.shape + j_4
                A_band_lvl_2_r = A_band_lvl_ptr[A_band_lvl_q]
                A_band_lvl_2_r_stop = A_band_lvl_ptr[A_band_lvl_q + 1] - 1
                if A_band_lvl_2_r <= A_band_lvl_2_r_stop
                    A_band_lvl_2_i1 = A_band_lvl_idx[A_band_lvl_2_r]
                    A_band_lvl_2_q_stop = A_band_lvl_ofs[A_band_lvl_2_r + 1]
                    A_band_lvl_2_i_2 = A_band_lvl_2_i1 - ((A_band_lvl_2_q_stop - A_band_lvl_ofs[A_band_lvl_2_r]) - 1)
                    A_band_lvl_2_q_ofs = (A_band_lvl_2_q_stop - A_band_lvl_2_i1) - 1
                else
                    A_band_lvl_2_i_2 = 1
                    A_band_lvl_2_i1 = 0
                end
                phase_start_2 = max(1, A_band_lvl_2_i_2)
                phase_stop_2 = min(x_lvl.shape, A_band_lvl_2_i1)
                if phase_stop_2 >= phase_start_2
                    for i_6 = phase_start_2:phase_stop_2
                        x_lvl_q = (1 - 1) * x_lvl.shape + i_6
                        A_band_lvl_2_q = A_band_lvl_2_q_ofs + i_6
                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                        A_band_lvl_3_val = A_band_lvl_2_val[A_band_lvl_2_q]
                        y_lvl_val[y_lvl_q] += A_band_lvl_3_val * x_lvl_2_val
                    end
                end
            end
            resize!(y_lvl_val, A_band_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_band_lvl.shape)),)
        end
end

function spmv_finch_band_kernel_row_maj(y, A, x)
    spmv_finch_band_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_band_unsym_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end

    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_band_kernel_row_maj($_y, $_A, $_x)
    return (;time = time, y = y[])
end

function ssymv_finch_pattern_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, PatternLevel{Int64}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                y_j_val = 0
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            y_lvl_q = (1 - 1) * A_lvl.shape + A_lvl_2_i
                            x_lvl_q_2 = (1 - 1) * x_lvl.shape + A_lvl_2_i
                            x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                            y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                            y_j_val = x_lvl_2_val_2 + y_j_val
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                y_lvl_q = (1 - 1) * A_lvl.shape + phase_stop_3
                                x_lvl_q_2 = (1 - 1) * x_lvl.shape + phase_stop_3
                                x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                                y_j_val += x_lvl_2_val_3
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = x_lvl_2_val + y_lvl_val[y_lvl_q_2] + y_j_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end


function ssymv_finch_pattern_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_pattern_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_pattern(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))))
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end

    A_pattern = pattern!(_A)
    d_pattern = pattern!(_d)
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(A_pattern)
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_pattern_kernel($_y, $A_pattern, $_x, $d_pattern)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_pattern_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            y_lvl_q = (1 - 1) * A_lvl_2.shape + A_lvl_2_i
                            y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                y_lvl_q = (1 - 1) * A_lvl_2.shape + phase_stop_3
                                y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_pattern_kernel(y, A, x)
    spmv_finch_pattern_kernel_helper(y, A, x)
    y
end

function spmv_finch_pattern_unsym(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    A = pattern!(_A)
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(A)
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_pattern_kernel($_y, $A, $_x)
    return (;time = time, y = y[])
endusing Finch
using BenchmarkTools

function spmv_finch_pattern_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_pattern::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_pattern_lvl = A_pattern.lvl
            A_pattern_lvl_2 = A_pattern_lvl.lvl
            A_pattern_lvl_ptr = A_pattern_lvl_2.ptr
            A_pattern_lvl_idx = A_pattern_lvl_2.idx
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_pattern_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_pattern_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_pattern_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_pattern_lvl.shape)
            for j_4 = 1:A_pattern_lvl.shape
                y_lvl_q = (1 - 1) * A_pattern_lvl.shape + j_4
                A_pattern_lvl_q = (1 - 1) * A_pattern_lvl.shape + j_4
                A_pattern_lvl_2_q = A_pattern_lvl_ptr[A_pattern_lvl_q]
                A_pattern_lvl_2_q_stop = A_pattern_lvl_ptr[A_pattern_lvl_q + 1]
                if A_pattern_lvl_2_q < A_pattern_lvl_2_q_stop
                    A_pattern_lvl_2_i1 = A_pattern_lvl_idx[A_pattern_lvl_2_q_stop - 1]
                else
                    A_pattern_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_pattern_lvl_2_i1)
                if phase_stop >= 1
                    if A_pattern_lvl_idx[A_pattern_lvl_2_q] < 1
                        A_pattern_lvl_2_q = Finch.scansearch(A_pattern_lvl_idx, 1, A_pattern_lvl_2_q, A_pattern_lvl_2_q_stop - 1)
                    end
                    while true
                        A_pattern_lvl_2_i = A_pattern_lvl_idx[A_pattern_lvl_2_q]
                        if A_pattern_lvl_2_i < phase_stop
                            x_lvl_q = (1 - 1) * x_lvl.shape + A_pattern_lvl_2_i
                            x_lvl_2_val = x_lvl_val[x_lvl_q]
                            y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                            A_pattern_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_pattern_lvl_2_i, phase_stop)
                            if A_pattern_lvl_2_i == phase_stop_3
                                x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop_3
                                x_lvl_2_val_2 = x_lvl_val[x_lvl_q]
                                y_lvl_val[y_lvl_q] += x_lvl_2_val_2
                                A_pattern_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_pattern_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_pattern_lvl.shape)),)
        end
end

function spmv_finch_pattern_kernel_row_maj(y, A, x)
    spmv_finch_pattern_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_pattern_unsym_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    A_pattern = pattern!(_A)
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(A)
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_pattern_kernel_row_maj($_y, $A_pattern, $_x)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_point_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparsePointLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                else
                    A_lvl_2_i = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i)
                if phase_stop >= 1
                    if phase_stop < A_lvl_2_i
                    else
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        y_lvl_q = (1 - 1) * A_lvl_2.shape + phase_stop
                        y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_point_kernel(y, A, x)
    spmv_finch_point_kernel_helper(y, A, x)
    y
end

function spmv_finch_point(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_point_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
endusing Finch
using BenchmarkTools

function spmv_finch_point_pattern_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparsePointLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                else
                    A_lvl_2_i = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i)
                if phase_stop >= 1
                    if phase_stop < A_lvl_2_i
                    else
                        y_lvl_q = (1 - 1) * A_lvl_2.shape + phase_stop
                        y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_point_pattern_kernel(y, A, x)
    spmv_finch_point_pattern_kernel_helper(y, A, x)
    y
end

function spmv_finch_point_pattern(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Element(0.0))), A)
    A_pattern = pattern!(_A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_point_pattern_kernel($_y, $A_pattern, $_x)
    return (;time = time, y = y[])
endusing Finch
using BenchmarkTools

function spmv_finch_point_pattern_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparsePointLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_4 = 1:A_lvl.shape
                y_lvl_q = (1 - 1) * A_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                else
                    A_lvl_2_i = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i)
                if phase_stop >= 1
                    if phase_stop < A_lvl_2_i
                    else
                        x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop
                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                        y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                    end
                end
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function spmv_finch_point_pattern_kernel_row_maj(y, A, x)
    spmv_finch_point_pattern_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_point_pattern_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Element(0.0))), A)
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    A_pattern = pattern!(_A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_point_pattern_kernel_row_maj($_y, $A_pattern, $_x)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_point_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_point::Tensor{DenseLevel{Int64, SparsePointLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_point_lvl = A_point.lvl
            A_point_lvl_2 = A_point_lvl.lvl
            A_point_lvl_ptr = A_point_lvl_2.ptr
            A_point_lvl_idx = A_point_lvl_2.idx
            A_point_lvl_2_val = A_point_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_point_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_point_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_point_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_point_lvl.shape)
            for j_4 = 1:A_point_lvl.shape
                y_lvl_q = (1 - 1) * A_point_lvl.shape + j_4
                A_point_lvl_q = (1 - 1) * A_point_lvl.shape + j_4
                A_point_lvl_2_q = A_point_lvl_ptr[A_point_lvl_q]
                A_point_lvl_2_q_stop = A_point_lvl_ptr[A_point_lvl_q + 1]
                if A_point_lvl_2_q < A_point_lvl_2_q_stop
                    A_point_lvl_2_i = A_point_lvl_idx[A_point_lvl_2_q]
                else
                    A_point_lvl_2_i = 0
                end
                phase_stop = min(x_lvl.shape, A_point_lvl_2_i)
                if phase_stop >= 1
                    if phase_stop < A_point_lvl_2_i
                    else
                        A_point_lvl_3_val = A_point_lvl_2_val[A_point_lvl_2_q]
                        x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop
                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                        y_lvl_val[y_lvl_q] += A_point_lvl_3_val * x_lvl_2_val
                    end
                end
            end
            resize!(y_lvl_val, A_point_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_point_lvl.shape)),)
        end
end

function spmv_finch_point_kernel_row_maj(y, A, x)
    spmv_finch_point_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_point_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_point_kernel_row_maj($_y, $_A, $_x)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            y_lvl_q = (1 - 1) * A_lvl_2.shape + A_lvl_2_i
                            y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                y_lvl_q = (1 - 1) * A_lvl_2.shape + phase_stop_3
                                y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_kernel(y, A, x)
    spmv_finch_kernel_helper(y, A, x)
    y
end

function spmv_finch_unsym(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
endusing Finch
using BenchmarkTools

function spmv_finch_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_4 = 1:A_lvl.shape
                y_lvl_q = (1 - 1) * A_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            x_lvl_q = (1 - 1) * x_lvl.shape + A_lvl_2_i
                            x_lvl_2_val = x_lvl_val[x_lvl_q]
                            y_lvl_val[y_lvl_q] += A_lvl_3_val * x_lvl_2_val
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop_3
                                x_lvl_2_val_2 = x_lvl_val[x_lvl_q]
                                y_lvl_val[y_lvl_q] += A_lvl_3_val * x_lvl_2_val_2
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function spmv_finch_kernel_row_maj(y, A, x)
    spmv_finch_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_unsym_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_kernel_row_maj($_y, $_A, $_x)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function ssymv_finch_vbl_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            diag_lvl_val = diag_lvl.lvl.val
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                diag_lvl_q = (1 - 1) * diag_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                diag_lvl_2_val = diag_lvl_val[diag_lvl_q]
                y_j_val = 0
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_r < A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_lvl_idx[A_lvl_2_r] < 1
                        A_lvl_2_r = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_r, A_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_r]
                        A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                        A_lvl_2_i_2 = A_lvl_2_i - (A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r])
                        A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i) - 1
                        if A_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_lvl_2_i_2)
                            if A_lvl_2_i >= phase_start_3
                                for i_10 = phase_start_3:A_lvl_2_i
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_10
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_10
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_10
                                    A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                    x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val = A_lvl_3_val * x_lvl_2_val_2 + y_j_val
                                end
                            end
                            A_lvl_2_r += A_lvl_2_i == A_lvl_2_i
                            i = A_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_14 = phase_start_6:phase_stop_5
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_14
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_14
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_14
                                    A_lvl_3_val_2 = A_lvl_2_val[A_lvl_2_q]
                                    x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = A_lvl_3_val_2 * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val += A_lvl_3_val_2 * x_lvl_2_val_3
                                end
                            end
                            A_lvl_2_r += phase_stop_5 == A_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = y_j_val + y_lvl_val[y_lvl_q_2] + x_lvl_2_val * diag_lvl_2_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_vbl_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_vbl_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_vbl(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_vbl_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function ssymv_finch_vbl_int8_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int8, Int64, Vector{Int8}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, ElementLevel{0, Int8, Int64, Vector{Int8}}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            diag_lvl_val = diag_lvl.lvl.val
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                diag_lvl_q = (1 - 1) * diag_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                diag_lvl_2_val = diag_lvl_val[diag_lvl_q]
                y_j_val = 0
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_r < A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_lvl_idx[A_lvl_2_r] < 1
                        A_lvl_2_r = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_r, A_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_r]
                        A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                        A_lvl_2_i_2 = A_lvl_2_i - (A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r])
                        A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i) - 1
                        if A_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_lvl_2_i_2)
                            if A_lvl_2_i >= phase_start_3
                                for i_10 = phase_start_3:A_lvl_2_i
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_10
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_10
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_10
                                    A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                    x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val = A_lvl_3_val * x_lvl_2_val_2 + y_j_val
                                end
                            end
                            A_lvl_2_r += A_lvl_2_i == A_lvl_2_i
                            i = A_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_14 = phase_start_6:phase_stop_5
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_14
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_14
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_14
                                    A_lvl_3_val_2 = A_lvl_2_val[A_lvl_2_q]
                                    x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = A_lvl_3_val_2 * x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val += A_lvl_3_val_2 * x_lvl_2_val_3
                                end
                            end
                            A_lvl_2_r += phase_stop_5 == A_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = y_j_val + y_lvl_val[y_lvl_q_2] + x_lvl_2_val * diag_lvl_2_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_vbl_int8_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_vbl_int8_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_vbl_int8(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(Int8(0)))), A)
    _d = Tensor(Dense(Element(Int8(0))))
    @finch mode=:fast begin
        _A .= Int8(0)
        _d .= Int8(0)
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_vbl_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function ssymv_finch_vbl_pattern_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, PatternLevel{Int64}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                y_j_val = 0
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_r < A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_lvl_idx[A_lvl_2_r] < 1
                        A_lvl_2_r = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_r, A_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_r]
                        A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                        A_lvl_2_i_2 = A_lvl_2_i - (A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r])
                        if A_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_lvl_2_i_2)
                            if A_lvl_2_i >= phase_start_3
                                for i_10 = phase_start_3:A_lvl_2_i
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_10
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_10
                                    x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val = x_lvl_2_val_2 + y_j_val
                                end
                            end
                            A_lvl_2_r += A_lvl_2_i == A_lvl_2_i
                            i = A_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_14 = phase_start_6:phase_stop_5
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_14
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_14
                                    x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val += x_lvl_2_val_3
                                end
                            end
                            A_lvl_2_r += phase_stop_5 == A_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = x_lvl_2_val + y_lvl_val[y_lvl_q_2] + y_j_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_vbl_pattern_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_vbl_pattern_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_vbl_pattern(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)
    
    A_pattern = pattern!(_A)
    d_pattern = pattern!(_d)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_vbl_pattern_kernel($_y, $A_pattern, $_x, $d_pattern)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_vbl_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_r < A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_lvl_idx[A_lvl_2_r] < 1
                        A_lvl_2_r = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_r, A_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_r]
                        A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                        A_lvl_2_i_2 = A_lvl_2_i - (A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r])
                        A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i) - 1
                        if A_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_lvl_2_i_2)
                            if A_lvl_2_i >= phase_start_3
                                for i_8 = phase_start_3:A_lvl_2_i
                                    y_lvl_q = (1 - 1) * A_lvl_2.shape + i_8
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_8
                                    A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                                end
                            end
                            A_lvl_2_r += A_lvl_2_i == A_lvl_2_i
                            i = A_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_11 = phase_start_6:phase_stop_5
                                    y_lvl_q = (1 - 1) * A_lvl_2.shape + i_11
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_11
                                    A_lvl_3_val_2 = A_lvl_2_val[A_lvl_2_q]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val_2 + y_lvl_val[y_lvl_q]
                                end
                            end
                            A_lvl_2_r += phase_stop_5 == A_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_vbl_kernel(y, A, x)
    spmv_finch_vbl_kernel_helper(y, A, x)
    y
end

function spmv_finch_vbl_unsym(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_vbl_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end
using Finch
using BenchmarkTools

function spmv_finch_vbl_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_vbl::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_vbl_lvl = A_vbl.lvl
            A_vbl_lvl_2 = A_vbl_lvl.lvl
            A_vbl_lvl_ptr = A_vbl_lvl_2.ptr
            A_vbl_lvl_idx = A_vbl_lvl_2.idx
            A_vbl_lvl_ofs = A_vbl_lvl_2.ofs
            A_vbl_lvl_2_val = A_vbl_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_vbl_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_vbl_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_vbl_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_vbl_lvl.shape)
            for j_4 = 1:A_vbl_lvl.shape
                y_lvl_q = (1 - 1) * A_vbl_lvl.shape + j_4
                A_vbl_lvl_q = (1 - 1) * A_vbl_lvl.shape + j_4
                A_vbl_lvl_2_r = A_vbl_lvl_ptr[A_vbl_lvl_q]
                A_vbl_lvl_2_r_stop = A_vbl_lvl_ptr[A_vbl_lvl_q + 1]
                if A_vbl_lvl_2_r < A_vbl_lvl_2_r_stop
                    A_vbl_lvl_2_i1 = A_vbl_lvl_idx[A_vbl_lvl_2_r_stop - 1]
                else
                    A_vbl_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_vbl_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_vbl_lvl_idx[A_vbl_lvl_2_r] < 1
                        A_vbl_lvl_2_r = Finch.scansearch(A_vbl_lvl_idx, 1, A_vbl_lvl_2_r, A_vbl_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_vbl_lvl_2_i = A_vbl_lvl_idx[A_vbl_lvl_2_r]
                        A_vbl_lvl_2_q_stop = A_vbl_lvl_ofs[A_vbl_lvl_2_r + 1]
                        A_vbl_lvl_2_i_2 = A_vbl_lvl_2_i - (A_vbl_lvl_2_q_stop - A_vbl_lvl_ofs[A_vbl_lvl_2_r])
                        A_vbl_lvl_2_q_ofs = (A_vbl_lvl_2_q_stop - A_vbl_lvl_2_i) - 1
                        if A_vbl_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_vbl_lvl_2_i_2)
                            if A_vbl_lvl_2_i >= phase_start_3
                                for i_8 = phase_start_3:A_vbl_lvl_2_i
                                    x_lvl_q = (1 - 1) * x_lvl.shape + i_8
                                    A_vbl_lvl_2_q = A_vbl_lvl_2_q_ofs + i_8
                                    x_lvl_2_val = x_lvl_val[x_lvl_q]
                                    A_vbl_lvl_3_val = A_vbl_lvl_2_val[A_vbl_lvl_2_q]
                                    y_lvl_val[y_lvl_q] += A_vbl_lvl_3_val * x_lvl_2_val
                                end
                            end
                            A_vbl_lvl_2_r += A_vbl_lvl_2_i == A_vbl_lvl_2_i
                            i = A_vbl_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_vbl_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_vbl_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_11 = phase_start_6:phase_stop_5
                                    x_lvl_q = (1 - 1) * x_lvl.shape + i_11
                                    A_vbl_lvl_2_q = A_vbl_lvl_2_q_ofs + i_11
                                    x_lvl_2_val_2 = x_lvl_val[x_lvl_q]
                                    A_vbl_lvl_3_val_2 = A_vbl_lvl_2_val[A_vbl_lvl_2_q]
                                    y_lvl_val[y_lvl_q] += A_vbl_lvl_3_val_2 * x_lvl_2_val_2
                                end
                            end
                            A_vbl_lvl_2_r += phase_stop_5 == A_vbl_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_vbl_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_vbl_lvl.shape)),)
        end
end

function spmv_finch_vbl_kernel_row_maj(y, A, x)
    spmv_finch_vbl_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_vbl_unsym_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_vbl_kernel_row_maj($_y, $_A, $_x)
    return (;time = time, y = y[])
end