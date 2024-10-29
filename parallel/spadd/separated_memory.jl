using Finch
using BenchmarkTools
using Base.Threads

include("concat.jl")

function separated_memory_add(A, B)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _B = Tensor(Dense(SparseList(Element(0.0))), B)
    time = @belapsed begin
        (_A, _B) = $(_A, _B)
        num_threads = Threads.nthreads()
        partial_sum = Vector{Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}(undef, num_threads)
        num_col = size(_A)[2]
        Threads.@threads for k = 1:num_threads
            partial_sum[k] = partial_add(_A, _B, 1 + div((k - 1) * num_col, num_threads), div(k * num_col, num_threads))
        end

        global _C = concat_vec(partial_sum)
    end
    return (; time=time, C=_C)
end

# Add A and B from column start_col to stop_col (inclusive)
function partial_add(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, start_col::Int64, stop_col::Int64)
    @inbounds @fastmath(begin
        A_lvl = A.lvl # DenseLevel
        A_lvl_2 = A_lvl.lvl # SparseListLevel
        A_lvl_ptr = A_lvl_2.ptr # Vector{Int64}
        A_lvl_idx = A_lvl_2.idx # Vector{Int64}
        # A_lvl_3 = A_lvl_2.lvl # ElementLevel
        A_lvl_2_val = A_lvl_2.lvl.val # Vector{Float64}

        B_lvl = B.lvl # DenseLevel
        B_lvl_2 = B_lvl.lvl # SparseListLevel
        B_lvl_ptr = B_lvl_2.ptr # Vector{Int64}
        B_lvl_idx = B_lvl_2.idx # Vector{Int64}
        # B_lvl_3 = B_lvl_2.lvl # ElementLevel
        B_lvl_2_val = B_lvl_2.lvl.val # Vector{Float64}

        # Assertion
        # A_lvl.shape == B_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(B_lvl.shape))"))
        # A_lvl_2.shape == B_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl_2.shape) != $(B_lvl_2.shape))"))
        # A_lvl.shape >= stop_col || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) < $(stop_col))"))

        C_lvl_2_val = Vector{Float64}()
        C_lvl_idx = Vector{Int64}()
        C_lvl_ptr = Vector{Int64}([1])
        current_ptr = 1
        for j = start_col:stop_col
            A_idx = A_lvl_ptr[j]
            B_idx = B_lvl_ptr[j]
            while A_idx < A_lvl_ptr[j+1] && B_idx < B_lvl_ptr[j+1]
                current_ptr += 1
                A_row_idx = A_lvl_idx[A_idx]
                B_row_idx = B_lvl_idx[B_idx]

                if A_row_idx < B_row_idx
                    push!(C_lvl_2_val, A_lvl_2_val[A_idx])
                    push!(C_lvl_idx, A_row_idx)
                    A_idx += 1
                elseif A_row_idx > B_row_idx
                    push!(C_lvl_2_val, B_lvl_2_val[B_idx])
                    push!(C_lvl_idx, B_row_idx)
                    B_idx += 1
                else
                    push!(C_lvl_2_val, A_lvl_2_val[A_idx] + B_lvl_2_val[B_idx])
                    push!(C_lvl_idx, A_row_idx)
                    A_idx += 1
                    B_idx += 1
                end
            end

            append!(C_lvl_2_val, A_lvl_2_val[A_idx:A_lvl_ptr[j+1]-1])
            append!(C_lvl_idx, A_lvl_idx[A_idx:A_lvl_ptr[j+1]-1])
            current_ptr += A_lvl_ptr[j+1] - A_idx

            append!(C_lvl_2_val, B_lvl_2_val[B_idx:B_lvl_ptr[j+1]-1])
            append!(C_lvl_idx, B_lvl_idx[B_idx:B_lvl_ptr[j+1]-1])
            current_ptr += B_lvl_ptr[j+1] - B_idx

            append!(C_lvl_ptr, current_ptr)
        end

        C_lvl_3 = Element{0.0,Float64,Int64}(C_lvl_2_val)
        C_lvl_shape = A_lvl_2.shape

        C_lvl_2 = SparseList{Int64}(C_lvl_3, C_lvl_shape, C_lvl_ptr, C_lvl_idx)
        C_lvl = Dense{Int64}(C_lvl_2, stop_col - start_col + 1)

        C = Tensor(C_lvl)
        return C
    end)
end
