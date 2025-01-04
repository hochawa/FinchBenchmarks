@testset "Galley Tests" begin

using Test
using Finch
using Finch: AsArray
using SparseArrays
using LinearAlgebra

using Finch.Galley
using Finch.Galley: t_sparse_list, t_dense
using Finch.Galley: canonicalize, insert_statistics!, get_reduce_query, AnnotatedQuery, reduce_idx!, cost_of_reduce, greedy_query_to_plan
using Finch.Galley: estimate_nnz, reduce_tensor_stats, condense_stats!, merge_tensor_stats
 

@testset verbose = true "Plan Equality" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @test Input(A, :i, :j, "a1") == Input(A, :i, :j, "a1")
    @test Input(A, :i, :j, "a1") != Input(A, :i, :j, "a2")
    @test Input(A, :i, :j, "a1") != Input(A, :i, :k, "a1")
    @test MapJoin(exp, Input(A, :i, :j, "a1")) == MapJoin(exp, Input(A, :i, :j, "a1"))
    @test MapJoin(exp, Input(A, :i, :j, "a1")) != MapJoin(exp, Input(A, :i, :k, "a1"))
    @test MapJoin(exp, Input(A, :i, :j, "a1")) != MapJoin(+, Input(A, :i, :j, "a1"))
    @test MapJoin(exp, Input(A, :i, :j, "a1")) != MapJoin(exp, Input(A, :i, :j, "a2"))

    @test MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2")) == MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2"))
    @test MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2")) != MapJoin(exp, Input(A, :i, :j, "a2"), Input(A, :i, :j, "a1"))



end

@testset verbose = true "Plan Hash" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @test hash(Input(A, :i, :j, "a1")) == hash(Input(A, :i, :j, "a1"))
    @test hash(Input(A, :i, :j, "a1")) != hash(Input(A, :i, :j, "a2"))
    @test hash(Input(A, :i, :j, "a1")) != hash(Input(A, :i, :k, "a1"))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) == hash(MapJoin(exp, Input(A, :i, :j, "a1")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) != hash(MapJoin(exp, Input(A, :i, :k, "a1")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) != hash(MapJoin(+, Input(A, :i, :j, "a1")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"))) != hash(MapJoin(exp, Input(A, :i, :j, "a2")))

    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2"))) == hash(MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2")))
    @test hash(MapJoin(exp, Input(A, :i, :j, "a1"), Input(A, :i, :j, "a2"))) != hash(MapJoin(exp, Input(A, :i, :j, "a2"), Input(A, :i, :j, "a1")))


end


@testset verbose = true "Annotated Queries" begin
    A = Tensor(Dense(Sparse(Element(0.0))), fsprand(5, 5, .2))

    @testset "get_reduce_query" begin
        chain_expr = Query(:out, Materialize(Aggregate(+, 0, :i, :j, :k, MapJoin(*, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:i, aq)
        expected_expr = Aggregate(+, 0, :i, Input(A, :i, :j, "a1"))
        @test query.expr == expected_expr

        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:j, aq)
        expected_expr = Aggregate(+, 0, :i, :j, :k, MapJoin(*, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))
        @test query.expr == expected_expr

        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:k, aq)
        expected_expr = Aggregate(+, 0, :k, Input(A, :j, :k, "a2"))
        @test query.expr == expected_expr

        # Check that we don't push aggregates past operations which don't distribute over them.
        chain_expr = Query(:out, Materialize(Aggregate(+, 0, :i, :j, :k, MapJoin(max, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:i, aq)
        expected_expr = Aggregate(+, 0, :i, :j, :k, MapJoin(max, Input(A, :i, :j, "a1"), Input(A, :j, :k, "a2")))
        @test query.expr == expected_expr

        # Check that we respect aggregates' position in the exression
        chain_expr = Query(:out, Materialize(Aggregate(+, 0, :j, :k, MapJoin(max, Aggregate(+, 0, :i, Input(A, :i, :j, "a1")), Input(A, :j, :k, "a2")))))
        aq = AnnotatedQuery(chain_expr, NaiveStats)
        query = reduce_idx!(:i, aq)
        expected_expr = Aggregate(+, 0, :i, Input(A, :i, :j, "a1"))
        @test query.expr == expected_expr
    end
end

@testset verbose = true "matrix operations" begin
    verbose = 0
    @testset "2x2 matrices, element-wise mult" begin
        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(*, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix .* b_matrix
        @test result.value[1] == correct_matrix
    end

    @testset "2x2 matrices, element-wise add" begin
        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(+, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix .+ b_matrix
        @test result.value[1] == correct_matrix
    end

    @testset "2x2 matrices, element-wise custom" begin
        f(x,y) = min(x,y)
        a_matrix = [1 0; 0 1]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [0 1; 1 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(f, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = [0 0; 0 0]
        @test result.value[1] == correct_matrix
    end

    @testset "2x2 matrices, element-wise mult, reverse input" begin
        a_matrix = [1 1; 0 0]
        a_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = [1 1; 0 0]
        b_data = Tensor(SparseList(SparseList(Element(0.0), 2), 2))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :i)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :j, MapJoin(*, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix .* (b_matrix')
        @test result.value[1] == correct_matrix
    end

    @testset "100x100 matrices, element-wise mult, reverse output" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .01)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :j, :i, MapJoin(*, a, b)))
        result = galley(q, verbose=verbose)
        correct_matrix = (a_matrix.*b_matrix)'
        @test result.value[1] == correct_matrix
    end

    @testset "100x100 matrices, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .01)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :k, Aggregate(+, 0, :j, MapJoin(*, a, b))))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test result.value[1] == correct_matrix
    end


    @testset "100x100 matrices, matrix mult, custom add" begin
        f(args...) = +(0, args...)
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        q = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :k, Aggregate(f, 0, :j, MapJoin(*, a, b))))
        result = galley(q, verbose=verbose)
        correct_matrix = a_matrix * b_matrix
        @test result.value[1] == correct_matrix
    end


    @testset "100x100 matrices, full sum" begin
        a_matrix = sprand(Bool, 100, 100, .01)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        q = Query(:out, Materialize(Aggregate(+, 0, :i, :j, a)))
        result = galley(q, verbose=verbose)
        correct_matrix = sum(a_matrix)
        @test result.value[1][] == correct_matrix
    end

    @testset "100x100 matrices, multi-line, matrix mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        c_matrix = sprand(Bool, 100, 100, .1)
        c_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(c_data, c_matrix)
        c = Input(c_data, :k, :l)
        d = Aggregate(+, 0, :j, MapJoin(*, a, b))
        e = Query(:out, Materialize(t_sparse_list, t_sparse_list, :i, :l, Aggregate(+, 0, :k, MapJoin(*, d, c))))
        result = galley(e, verbose=verbose)
        d_matrix = a_matrix * b_matrix
        correct_matrix = d_matrix * c_matrix
        @test result.value[1] == correct_matrix
    end

    @testset "100x100 matrices, multi-line, matrix mult, reuse" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        d = Materialize(t_sparse_list, t_sparse_list, :i, :k, Aggregate(+, 0, :j, MapJoin(*, a, b)))
        e = Query(:out, Materialize(t_dense, t_dense, :i, :l, Aggregate(+, 0, :k, MapJoin(*, Input(d, :i, :k), Input(d, :k, :l)))))
        result = galley(e, verbose=verbose)
        d_matrix = a_matrix * b_matrix
        correct_matrix = d_matrix * d_matrix
        @test result.value[1] == correct_matrix
    end

    @testset "100x100 matrices, diagonal mult" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :i)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :i)
        d = Query(:out, Materialize(t_dense, :i, MapJoin(*, a, b)))
        result = galley(d, verbose=verbose)
        correct_matrix = spzeros(100)
        for i in 1:100
            correct_matrix[i] = a_matrix[i,i] * b_matrix[i,i]
        end
        @test result.value[1] == correct_matrix
    end

    @testset "100x100 matrices, diagonal mult, then sum" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :i)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :i)
        d = Query(:out, Materialize(Aggregate(+, 0, :i, MapJoin(*, a, b))))
        result = galley(d, verbose=verbose)
        correct_result = 0
        for i in 1:100
            correct_result += a_matrix[i,i] * b_matrix[i,i]
        end
        @test result.value[1][] == correct_result
    end


    @testset "100x100 matrices, elementwise +, then sum" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :i, :j)
        d = Query(:out, Materialize(Aggregate(+, 0, :i, :j, MapJoin(+, a, b))))
        result = galley(d, verbose=verbose)
        correct_result = sum(a_matrix) + sum(b_matrix)
        @test result.value[1][] == correct_result
    end

    @testset "100x100 matrices, + on j, then sum all" begin
        a_matrix = sprand(Bool, 100, 100, .1)
        a_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(a_data, a_matrix)
        a = Input(a_data, :i, :j)
        b_matrix = sprand(Bool, 100, 100, .1)
        b_data = Tensor(SparseList(SparseList(Element(0), 100), 100))
        copyto!(b_data, b_matrix)
        b = Input(b_data, :j, :k)
        d = Query(:out, Materialize(Aggregate(+, 0, :i, :j, :k, MapJoin(+, a, b))))
        result = galley(d, verbose=verbose)
        correct_result = sum(a_matrix)*100 + sum(b_matrix)*100
        @test result.value[1][] == correct_result
    end

end


@testset "NaiveStats" begin


end


@testset verbose = true "DCStats" begin


    @testset "Single Tensor Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        dims = Dict(i=>1000, j=>1000)
        def = TensorDef(Set([i,j]), dims, 0.0, nothing, nothing, nothing)
        i, j = 1, 2
        dcs = Set([DC(Set([i]), Set([j]), 5),
                 DC(Set([j]), Set([i]), 25),
                 DC(Set{Int}(), Set([i, j]), 50),
                ])
        idx_2_int = Dict(:i=>1, :j=>2)
        int_2_idx = Dict(1=>:i, 2=>:j)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        @test estimate_nnz(stat) == 50
    end

    @testset "1 Join DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        i, j, k= 1, 2, 3
        dcs = Set([
                 DC(Set([j]), Set([k]), 5),
                 DC(Set{Int}(), Set([i, j]), 50),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        @test estimate_nnz(stat) == 50*5
    end

    @testset "2 Join DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        l = IndexExpr("l")
        dims = Dict(i=>1000, j=>1000, k=>1000, l=>1000)
        def = TensorDef(Set([i,j,k,l]), dims, 0.0, nothing, nothing, nothing)
        i, j, k, l = 1, 2, 3, 4
        dcs = Set([
                DC(Set{Int}(), Set([i, j]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([l]), 5),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3, :l=>4)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k, 4=>:l)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        @test estimate_nnz(stat) == 50*5*5
    end

    @testset "Triangle DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        i, j, k = 1, 2, 3
        dcs = Set([
                DC(Set{Int}(), Set([i, j]), 50),
                DC(Set([i]), Set([j]), 5),
                DC(Set([j]), Set([i]), 5),
                DC(Set{Int}(), Set([j, k]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([j]), 5),
                DC(Set{Int}(), Set([i, k]), 50),
                DC(Set([i]), Set([k]), 5),
                DC(Set([k]), Set([i]), 5),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        @test estimate_nnz(stat) == 50*5
    end

    @testset "Triangle-Small DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        # In this version, |R(i,j)| = 1
        i, j, k = 1, 2, 3
        dcs = Set([
                DC(Set{Int}(), Set([i, j]), 1),
                DC(Set([i]), Set([j]), 1),
                DC(Set([j]), Set([i]), 1),
                DC(Set{Int}(), Set([j, k]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([j]), 5),
                DC(Set{Int}(), Set([i, k]), 50),
                DC(Set([i]), Set([k]), 5),
                DC(Set([k]), Set([i]), 5),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        @test estimate_nnz(stat) == 1*5
    end

    @testset "Full Reduce DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        i, j, k = 1, 2, 3
        dcs = Set([
                DC(Set{Int}(), Set([i, j]), 50),
                DC(Set([i]), Set([j]), 5),
                DC(Set([j]), Set([i]), 5),
                DC(Set{Int}(), Set([j, k]), 50),
                DC(Set([j]), Set([k]), 5),
                DC(Set([k]), Set([j]), 5),
                DC(Set{Int}(), Set([i, k]), 50),
                DC(Set([i]), Set([k]), 5),
                DC(Set([k]), Set([i]), 5),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        reduce_stats = reduce_tensor_stats(+, 0, Set([:i,:j,:k]), stat)
        @test estimate_nnz(reduce_stats) == 1
    end

    @testset "1-Attr Reduce DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        i, j, k = 1, 2, 3
        dcs = Set([
                    DC(Set{Int}(), Set([i, j]), 1),
                    DC(Set([i]), Set([j]), 1),
                    DC(Set([j]), Set([i]), 1),
                    DC(Set{Int}(), Set([j, k]), 50),
                    DC(Set([j]), Set([k]), 5),
                    DC(Set([k]), Set([j]), 5),
                    DC(Set{Int}(), Set([i, k]), 50),
                    DC(Set([i]), Set([k]), 5),
                    DC(Set([k]), Set([i]), 5),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        condense_stats!(stat)
        reduce_stats = reduce_tensor_stats(+, 0, Set([:i, :j]), stat)
        @test estimate_nnz(reduce_stats) == 5
    end

    @testset "2-Attr Reduce DC Card" begin
        i = IndexExpr("i")
        j = IndexExpr("j")
        k = IndexExpr("k")
        dims = Dict(i=>1000, j=>1000, k=>1000)
        def = TensorDef(Set([i,j,k]), dims, 0.0, nothing, nothing, nothing)
        i, j, k = 1, 2, 3
        dcs = Set([
                    DC(Set{Int}(), Set([i, j]), 1),
                    DC(Set([i]), Set([j]), 1),
                    DC(Set([j]), Set([i]), 1),
                    DC(Set{Int}(), Set([j, k]), 50),
                    DC(Set([j]), Set([k]), 5),
                    DC(Set([k]), Set([j]), 5),
                    DC(Set{Int}(), Set([i, k]), 50),
                    DC(Set([i]), Set([k]), 5),
                    DC(Set([k]), Set([i]), 5),
                ])
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        stat = DCStats(def, idx_2_int, int_2_idx, dcs)
        reduce_stats = reduce_tensor_stats(+, 0, Set([:i]), stat)
        @test estimate_nnz(reduce_stats) == 5
    end

    @testset "1D Disjunction DC Card" begin
        dims = Dict(:i=>1000)
        def = TensorDef(Set([:i]), dims, 0.0, nothing, nothing, nothing)
        i = 1
        idx_2_int = Dict(:i=>1)
        int_2_idx = Dict(1=>:i)
        dcs1 = Set([DC(Set{Int}(), Set([i]), 1),])
        stat1 = DCStats(def, idx_2_int, int_2_idx, dcs1)
        dcs2 = Set([DC(Set{Int}(), Set([i]), 1),])
        stat2 = DCStats(def, idx_2_int, int_2_idx, dcs2)
        reduce_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(reduce_stats) == 2
    end

    @testset "2D Disjunction DC Card" begin
        dims = Dict(:i=>1000, :j => 100)
        def = TensorDef(Set([:i, :j]), dims, 0.0, nothing, nothing, nothing)
        idx_2_int = Dict(:i=>1, :j=>2)
        int_2_idx = Dict(1=>:i, 2=>:j)
        dcs1 = Set([DC(Set{Int}(), Set([1, 2]), 1),])
        stat1 = DCStats(def, idx_2_int, int_2_idx, dcs1)
        dcs2 = Set([DC(Set{Int}(), Set([1, 2]), 1),])
        stat2 = DCStats(def, idx_2_int, int_2_idx, dcs2)
        merge_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(merge_stats) == 2
    end

    @testset "2D Disjoint Disjunction DC Card" begin
        dims1 = Dict(:i=>1000)
        def1 = TensorDef(Set([:i]), dims1, 0.0, nothing, nothing, nothing)
        idx_2_int = Dict(:i=>1)
        int_2_idx = Dict(1=>:i)
        dcs1 = Set([DC(Set{Int}(), Set([1]), 5),])
        stat1 = DCStats(def1, idx_2_int, int_2_idx, dcs1)
        idx_2_int = Dict(:j=>2)
        int_2_idx = Dict(2=>:j)
        dims2 = Dict(:j => 100)
        def2 = TensorDef(Set([:j]), dims2, 0.0, nothing, nothing, nothing)
        dcs2 = Set([DC(Set{Int}(), Set([2]), 10),])
        stat2 = DCStats(def2, idx_2_int, int_2_idx, dcs2)
        merge_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(merge_stats) == (10*1000 + 5*100)
    end

    @testset "3D Disjoint Disjunction DC Card" begin
        dims1 = Dict(:i=>1000, :j=>100)
        def1 = TensorDef(Set([:i, :j]), dims1, 0.0, nothing, nothing, nothing)
        idx_2_int = Dict(:i=>1, :j=>2)
        int_2_idx = Dict(1=>:i, 2=>:j)
        dcs1 = Set([DC(Set{Int}(), Set([1, 2]), 5),])
        stat1 = DCStats(def1, idx_2_int, int_2_idx, dcs1)
        dims2 = Dict(:j => 100, :k=>1000)
        idx_2_int = Dict(:j=>2, :k=>3)
        int_2_idx = Dict(2=>:j, 3=>:k)
        def2 = TensorDef(Set([:j, :k]), dims2, 0.0, nothing, nothing, nothing)
        dcs2 = Set([DC(Set{Int}(), Set([2, 3]), 10),])
        stat2 = DCStats(def2, idx_2_int, int_2_idx, dcs2)
        merge_stats = merge_tensor_stats(+, stat1, stat2)
        @test estimate_nnz(merge_stats) == (10*1000 + 5*1000)
    end

    @testset "Mixture Disjunction Conjunction DC Card" begin
        dims1 = Dict(:i=>1000, :j=>100)
        def1 = TensorDef(Set([:i, :j]), dims1, 1, nothing, nothing, nothing)
        idx_2_int = Dict(:i=>1, :j=>2)
        int_2_idx = Dict(1=>:i, 2=>:j)
        dcs1 = Set([DC(Set{Int}(), Set([1, 2]), 5),])
        stat1 = DCStats(def1, idx_2_int, int_2_idx, dcs1)
        dims2 = Dict(:j => 100, :k=>1000)
        def2 = TensorDef(Set([:j, :k]), dims2, 1, nothing, nothing, nothing)
        idx_2_int = Dict(:j=>2, :k=>3)
        int_2_idx = Dict(2=>:j, 3=>:k)
        dcs2 = Set([DC(Set{Int}(), Set([2, 3]), 10),])
        stat2 = DCStats(def2, idx_2_int, int_2_idx, dcs2)
        dims3 = Dict(:i=>1000, :j => 100, :k=>1000)
        idx_2_int = Dict(:i=>1, :j=>2, :k=>3)
        int_2_idx = Dict(1=>:i, 2=>:j, 3=>:k)
        def3 = TensorDef(Set([:i, :j, :k]), dims3, 0.0, nothing, nothing, nothing)
        dcs3 = Set([DC(Set{Int}(), Set([1, 2, 3]), 10),])
        stat3 = DCStats(def3, idx_2_int, int_2_idx, dcs3)
        merge_stats = merge_tensor_stats(*, stat1, stat2, stat3)
        @test estimate_nnz(merge_stats) == 10
    end

end

end

nothing
