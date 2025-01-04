@testset "typical" begin
    @info "Testing Typical Usage"

    let
        io = IOBuffer()

        @repl io A = Tensor(SparseList(Element(0.0)), [2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0])
        @repl io B = Tensor(Dense(Element(0.0)), fill(1.1, 10))
        @repl io @finch_code for i=_; B[i] += A[i] end
        @repl io @finch for i=_; B[i] += A[i] end

        @test check_output("typical/typical_inplace_sparse_add.txt", String(take!(io)))
    end

    let
        io = IOBuffer()

        @repl io A = sparse([0 0 3.3; 1.1 0 0; 2.2 0 4.4; 0 0 5.5])
        @repl io y = [1.0, 2.0, 3.0, 4.0]
        @repl io x = [1, 2, 3]
        @repl io @finch_code begin
            y .= 0
            for j = _
                for i = _
                    y[i] += A[i, j] * x[j]
                end
            end
        end
        @repl io @finch begin
            y .= 0
            for j = _
                for i = _
                    y[i] += A[i, j] * x[j]
                end
            end
        end

        @test check_output("typical/typical_spmv_sparsematrixcsc.txt", String(take!(io)))
    end

    let
        io = IOBuffer()

        @repl io A = Tensor(Dense(SparseList(Element(0.0))), [0 0 3.3; 1.1 0 0; 2.2 0 4.4; 0 0 5.5])
        @repl io y = Tensor([1.0, 2.0, 3.0, 4.0])
        @repl io x = Tensor([1, 2, 3])
        @repl io @finch_code begin
            y .= 0
            for j = _
                for i = _
                    y[i] += A[i, j] * x[j]
                end
            end
        end
        @repl io @finch begin
            y .= 0
            for j = _
                for i = _
                    y[i] += A[i, j] * x[j]
                end
            end
        end

        @test check_output("typical/typical_spmv_csc.txt", String(take!(io)))
    end

    let
        io = IOBuffer()

        @repl io A = Tensor(Dense(SparseList(Element(0.0))), [0 0 3.3; 1.1 0 0; 2.2 0 4.4; 0 0 5.5])
        @repl io B = Tensor(SparseDict(SparseDict(Element(0.0))))
        @repl io @finch_code mode=:fast begin
            B .= 0
            for j = _
                for i = _
                    B[j, i] = A[i, j]
                end
            end
        end
        @repl io @finch mode=:fast begin
            B .= 0
            for j = _
                for i = _
                    B[j, i] = A[i, j]
                end
            end
        end

        @test check_output("typical/typical_transpose_csc_to_coo.txt", String(take!(io)))
    end

    let
        x = Tensor(
            SparseList(
                Element{0.0, Float64}([2.0, 3.0, 4.0, 5.0, 6.0]),
                10, [1, 6], [1, 3, 5, 7, 9]))
        y = Tensor(
            SparseList(
                Element{0.0, Float64}([1.0, 1.0, 1.0]),
                10, [1, 4], [2, 5, 8]))
        z = Tensor(SparseList(Element{0.0}(), 10))

        io = IOBuffer()

        @repl io @finch_code (z .= 0; for i=_; z[i] = x[gallop(i)] + y[gallop(i)] end)
        @repl io @finch (z .= 0; for i=_; z[i] = x[gallop(i)] + y[gallop(i)] end)

        @test check_output("typical/typical_merge_gallop.txt", String(take!(io)))

        io = IOBuffer()

        @repl io @finch_code (z .= 0; for i=_; z[i] = x[gallop(i)] + y[i] end)
        @repl io @finch (z .= 0; for i=_; z[i] = x[gallop(i)] + y[i] end)

        @test check_output("typical/typical_merge_leadfollow.txt", String(take!(io)))

        io = IOBuffer()

        @repl io @finch_code (z .= 0; for i=_; z[i] = x[i] + y[i] end)
        @repl io @finch (z .= 0; for i=_; z[i] = x[i] + y[i] end)

        @test check_output("typical/typical_merge_twofinger.txt", String(take!(io)))

        io = IOBuffer()

        @repl io X = Tensor(SparseList(Element(0.0)), [1.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0])
        @repl io x_min = Scalar(Inf)
        @repl io x_max = Scalar(-Inf)
        @repl io x_sum = Scalar(0.0)
        @repl io x_var = Scalar(0.0)
        @repl io @finch_code begin
            for i = _
                let x = X[i]
                    x_min[] <<min>>= x
                    x_max[] <<max>>= x
                    x_sum[] += x
                    x_var[] += x * x
                end
            end
        end

        @test check_output("typical/typical_stats_example.txt", String(take!(io)))
    end
end