using Finch
using BenchmarkTools
using Base.Threads


function separate_sparselist_separated_memory_add_static_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = Tensor(Dense(Separate(SparseList(Element(0.0)))), A)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                separate_sparselist_separated_memory_add_static(_y, _A, _x)
        end
        return (; time=time, y=_y)
end

function separate_sparselist_separated_memory_add_static(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SeparateLevel{SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}},Vector{SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
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
                                A_lvl_2_ptr_lvl_val = A_lvl_2_ptr.lvl.val
                                A_lvl_2_ptr_idx = A_lvl_2_ptr.idx
                                A_lvl_2_ptr_ptr = A_lvl_2_ptr.ptr
                                for q in A_lvl_2_ptr_ptr[1]:A_lvl_2_ptr_ptr[2]-1
                                        i = A_lvl_2_ptr_idx[q]
                                        temp_val = A_lvl_2_ptr_lvl_val[q] * x_lvl_val[j]
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
