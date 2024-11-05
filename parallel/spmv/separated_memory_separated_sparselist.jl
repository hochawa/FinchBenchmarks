using Finch
using BenchmarkTools
using Base.Threads


function separated_memory_separated_sparselist_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = Tensor(Dense(Separate(SparseList(Element(0.0)))), A)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                separated_memory_separated_sparselist(_y, _A, _x)
        end
        return (; time=time, y=_y)
end

function separated_memory_separated_sparselist(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SeparateLevel{SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}},Vector{SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
        @inbounds @fastmath(begin
                y_lvl = y.lvl
                y_lvl_val = y_lvl.lvl.val

                A_lvl = A.lvl
                A_lvl_val = A_lvl.lvl.val
                A_lvl_3 = A_lvl.lvl.lvl

                x_lvl = x.lvl
                x_lvl_val = x_lvl.lvl.val

                x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
                Finch.resize_if_smaller!(y_lvl_val, A_lvl_3.shape)
                Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_3.shape)

                num_threads = Threads.nthreads()
                y_temps = Vector{typeof(y_lvl_val)}(undef, num_threads)

                Threads.@threads for k = 1:num_threads
                        y_temps[k] = copy(y_lvl_val)
                        for j = 1+div((k - 1) * A_lvl.shape, num_threads):div(k * A_lvl.shape, num_threads)
                                A_lvl_2_ptr = A_lvl_val[j]
                                for q in A_lvl_2_ptr.ptr[1]:A_lvl_2_ptr.ptr[2]-1
                                        i = A_lvl_2_ptr.idx[q]
                                        temp_val = A_lvl_2_ptr.lvl.val[q] * x_lvl_val[j]
                                        y_temps[k][i] += temp_val
                                end
                        end
                end

                Threads.@threads for k = 1:num_threads
                        for j = 1:num_threads
                                for i = 1+div((k - 1) * y_lvl.shape, num_threads):div(k * y_lvl.shape, num_threads)
                                        y_lvl_val[i] += y_temps[j][i]
                                end
                        end
                end
        end)
end
