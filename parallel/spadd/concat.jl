using Finch

function concat(A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}})
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

        # val
        C_lvl_2_val = vcat(A_lvl_2_val, B_lvl_2_val)
        C_lvl_3 = Element{0.0,Float64,Int64}(C_lvl_2_val)
        # shape
        A_lvl_2.shape == B_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl_2.shape) != $(B_lvl_2.shape))"))
        C_lvl_shape = A_lvl_2.shape
        # pointer
        B_lvl_ptr_shift = B_lvl_ptr[2:end] .+ (last(A_lvl_ptr) - 1)
        C_lvl_ptr = vcat(A_lvl_ptr, B_lvl_ptr_shift)
        # index
        C_lvl_idx = vcat(A_lvl_idx, B_lvl_idx)

        C_lvl_2 = SparseList{Int64}(C_lvl_3, C_lvl_shape, C_lvl_ptr, C_lvl_idx)
        C_lvl = Dense{Int64}(C_lvl_2, A_lvl.shape + B_lvl.shape)

        C = Tensor(C_lvl)
        return C
    end)
end

function concat_vec(V::Vector{Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}})
    @inbounds @fastmath(begin
        # val
        B_lvl_2_val = reduce(vcat, (A.lvl.lvl.lvl.val for A in V))
        B_lvl_3 = Element{0.0,Float64,Int64}(B_lvl_2_val)
        # shape
        B_lvl_shape = V[1].lvl.lvl.shape
        # pointer
        B_lvl_ptr = V[1].lvl.lvl.ptr
        for index = 2:length(V)
            append!(B_lvl_ptr, V[index].lvl.lvl.ptr[2:end] .+ (last(B_lvl_ptr) - 1))
        end
        # index
        B_lvl_idx = reduce(vcat, (A.lvl.lvl.idx for A in V))

        B_lvl_2 = SparseList{Int64}(B_lvl_3, B_lvl_shape, B_lvl_ptr, B_lvl_idx)
        B_lvl = Dense{Int64}(B_lvl_2, mapreduce(A -> A.lvl.shape, +, V))

        B = Tensor(B_lvl)
        return B
    end)
end

