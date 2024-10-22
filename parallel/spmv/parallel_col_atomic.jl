using Finch
using BenchmarkTools
using Base.Threads


function parallel_col_atomic_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                parallel_col_atomic(_y, _A, _x)
        end
        return (; time=time, y=_y)
end

function parallel_col_atomic(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
        @inbounds @fastmath(begin
                y_lvl = y.lvl # DenseLevel
                # y_lvl_2 = y_lvl.lvl # ElementLevel
                y_lvl_val = y_lvl.lvl.val # Vector{Float64}

                A_lvl = A.lvl # DenseLevel
                A_lvl_2 = A_lvl.lvl # SparseListLevel
                A_lvl_ptr = A_lvl_2.ptr # Vector{Int64}
                A_lvl_idx = A_lvl_2.idx # Vector{Int64}
                # A_lvl_3 = A_lvl_2.lvl # ElementLevel
                A_lvl_2_val = A_lvl_2.lvl.val # Vector{Float64}

                x_lvl = x.lvl # DenseLevel
                # x_lvl_2 = x_lvl.lvl # ElementLevel
                x_lvl_val = x_lvl.lvl.val # Vector{Float64}

                x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
                Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)

                Threads.@threads for j = 1:A_lvl.shape
			Finch.@barrier begin
				for q in A_lvl_ptr[j]:A_lvl_ptr[j+1]-1
					i = A_lvl_idx[q]
					#y_lvl_val[i] += A_lvl_2_val[q] * x_lvl_val[j]
					#Core.Intrinsics.atomic_pointermodify(pointer(y_lvl_val, i), +, A_lvl_2_val[q] * x_lvl_val[j], :sequentially_consistent)
					Base.unsafe_modify!(pointer(y_lvl_val, i), +, A_lvl_2_val[q] * x_lvl_val[j])
				end
			end
                end
        end)
end
