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
end

using Finch
using BenchmarkTools

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
end
using Finch
using BenchmarkTools

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
end
using Finch
using BenchmarkTools

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
end
using Finch
using BenchmarkTools

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
end
using Finch
using BenchmarkTools

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