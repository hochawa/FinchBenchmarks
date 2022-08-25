using Finch
using SparseArrays
using BenchmarkTools
using Scratch
using Profile

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket

function triangle_taco(A, key)
    c_file = joinpath(mktempdir(prefix="triangle_taco_$(key)"), "c.ttx")
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "triangle_taco_$(key)")
    mkpath(persist_dir)
    A_file = joinpath(persist_dir, "A.ttx")
    A2_file = joinpath(persist_dir, "A2.ttx")
    AT_file = joinpath(persist_dir, "AT.ttx")

    ttwrite(c_file, (), [0], ())
    if !(isfile(A_file) && isfile(A2_file) && isfile(AT_file))
        (I, J, V) = findnz(A)
        ttwrite(A_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite(A2_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite(AT_file, (J, I), ones(Int32, length(V)), size(A))
    end

    io = IOBuffer()

    run(pipeline(`./triangle_taco $c_file $A_file $A2_file $AT_file`, stdout=io))

    c = ttread(c_file)[2][1]

    c_ref = Scalar{0}()
    A_ref = pattern!(fiber(A))
    AT_ref = pattern!(fiber(permutedims(A)))
    @finch @loop i j k c_ref[] += A_ref[i, j] && A_ref[j, k] && AT_ref[i, k]

    @assert c == c_ref()

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function triangle_finch_kernel(A, AT)
    c = Scalar{0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k] && AT[i, k]
    return c()
end
function triangle_finch(_A, key)
    A = pattern!(fiber(_A))
    AT = pattern!(fiber(permutedims(_A)))
    return @belapsed triangle_finch_kernel($A, $AT)
end

function triangle_finch_gallop_kernel(A, AT)
    c = Scalar{0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k::gallop] && AT[i, k::gallop]
    return c()
end
function triangle_finch_gallop(_A, key)
    A = pattern!(fiber(_A))
    AT = pattern!(fiber(permutedims(_A)))
    c_ref = Scalar{0}()
    @finch @loop i j k c_ref[] += A[i, j] && A[j, k] && AT[i, k]
    c = triangle_finch_gallop_kernel(A, AT)
    @assert c_ref() == c
    return @belapsed triangle_finch_gallop_kernel($A, $AT)
end

function main()
    for (mtx, key) in [
        ("SNAP/web-NotreDame", "web-NotreDame"),
        ("SNAP/roadNet-PA", "roadNet-PA"),
        ("DIMACS10/sd2010", "sd2010"),
        ("SNAP/soc-Epinions1", "soc-Epinions1"),
        ("SNAP/email-EuAll", "email-EuAll"),
        ("SNAP/wiki-Talk", "wiki-Talk"),
        ("SNAP/web-BerkStan", "web-BerkStan"),
        ("Gleich/flickr", "flickr"),
        ("Gleich/usroads", "usroads"),
        ("Pajek/USpowerGrid", "USpowerGrid"),
    ]
        A = SparseMatrixCSC(matrixdepot(mtx))
        println((key, size(A), nnz(A)))
        #println(maximum(A.colptr[2:end] - A.colptr[1:end-1]))
        #println(maximum(permutedims(A).colptr[2:end] - permutedims(A).colptr[1:end-1]))

        println("taco_time: ", triangle_taco(A, key))
        println("finch_time: ", triangle_finch(A, key))
        println("finch_gallop_time: ", triangle_finch_gallop(A, key))

    end
end

foo(A, AT) = @inbounds begin
    A_lvl = A.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_pos_alloc = length(A_lvl_2.pos)
    A_lvl_2_idx_alloc = length(A_lvl_2.idx)
    A_lvl_3 = A.lvl
    A_lvl_4 = A_lvl_3.lvl
    A_lvl_4_pos_alloc = length(A_lvl_4.pos)
    A_lvl_4_idx_alloc = length(A_lvl_4.idx)
    A_lvl_5 = AT.lvl
    A_lvl_6 = A_lvl_5.lvl
    A_lvl_6_pos_alloc = length(A_lvl_6.pos)
    A_lvl_6_idx_alloc = length(A_lvl_6.idx)
    j_stop = A_lvl_2.I
    k_stop = A_lvl_4.I
    i_stop = A_lvl.I
    c_val = 0
    for i = 1:i_stop
        A_lvl_q = (1 - 1) * A_lvl.I + i
        A_lvl_5_q = (1 - 1) * A_lvl_5.I + i
        A_lvl_2_q_start = A_lvl_2.pos[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_2.pos[A_lvl_q + 1]
        if A_lvl_2_q_start < A_lvl_2_q_stop
            A_lvl_2_i_start = A_lvl_2.idx[A_lvl_2_q_start]
            A_lvl_2_i_stop = A_lvl_2.idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i_start = 1
            A_lvl_2_i_stop = 0
        end
        A_lvl_6_q_start = A_lvl_6.pos[A_lvl_5_q]
        A_lvl_6_q_stop = A_lvl_6.pos[A_lvl_5_q + 1]
        if A_lvl_6_q_start < A_lvl_6_q_stop
            A_lvl_6_i_start = A_lvl_6.idx[A_lvl_6_q_start]
            A_lvl_6_i_stop = A_lvl_6.idx[A_lvl_6_q_stop - 1]
        else
            A_lvl_6_i_start = 1
            A_lvl_6_i_stop = 0
        end
        A_lvl_2_q = A_lvl_2_q_start
        A_lvl_2_i = A_lvl_2_i_start
        j = 1
        j_start = j
        phase_start = max(j_start)
        phase_stop = min(A_lvl_2_i_stop, j_stop)
        if phase_stop >= phase_start
            j = j
            j = phase_start
            while A_lvl_2_q < A_lvl_2_q_stop && A_lvl_2.idx[A_lvl_2_q] < phase_start
                A_lvl_2_q += 1
            end
            while j <= phase_stop
                j_start_2 = j
                A_lvl_2_i = A_lvl_2.idx[A_lvl_2_q]
                phase_stop_2 = A_lvl_2_i
                j_2 = j
                if A_lvl_2_i == phase_stop_2
                    j_3 = phase_stop_2
                    A_lvl_3_q = (1 - 1) * A_lvl_3.I + j_3
                    A_lvl_4_q_start = A_lvl_4.pos[A_lvl_3_q]
                    A_lvl_4_q_stop = A_lvl_4.pos[A_lvl_3_q + 1]
                    if A_lvl_4_q_start < A_lvl_4_q_stop
                        A_lvl_4_i_start = A_lvl_4.idx[A_lvl_4_q_start]
                        A_lvl_4_i_stop = A_lvl_4.idx[A_lvl_4_q_stop - 1]
                    else
                        A_lvl_4_i_start = 1
                        A_lvl_4_i_stop = 0
                    end
                    A_lvl_4_q = A_lvl_4_q_start
                    A_lvl_4_i = A_lvl_4_i_start
                    A_lvl_6_q = A_lvl_6_q_start
                    A_lvl_6_i = A_lvl_6_i_start
                    k = 1
                    k_start = k
                    phase_start_3 = max(k_start)
                    phase_stop_3 = min(A_lvl_4_i_stop, A_lvl_6_i_stop, k_stop)
                    if phase_stop_3 >= phase_start_3
                        k = k
                        k = phase_start_3
                        #while A_lvl_4_q < A_lvl_4_q_stop && A_lvl_4.idx[A_lvl_4_q] < phase_start_3
                        #    A_lvl_4_q += 1
                        #end
                        #while A_lvl_6_q < A_lvl_6_q_stop && A_lvl_6.idx[A_lvl_6_q] < phase_start_3
                        #    A_lvl_6_q += 1
                        #end
                        #while k <= phase_stop_3
                        while A_lvl_4_q < A_lvl_4_q_stop && A_lvl_6_q < A_lvl_6_q_stop
                            #k_start_2 = k
                            A_lvl_4_i = A_lvl_4.idx[A_lvl_4_q]
                            A_lvl_6_i = A_lvl_6.idx[A_lvl_6_q]
                            #phase_start_4 = max(k_start_2)
                            phase_stop_4 = min(A_lvl_4_i, A_lvl_6_i)#, phase_stop_3)
                            #if phase_stop_4 >= phase_start_4
                                #k_2 = k
                                if A_lvl_4_i == phase_stop_4 && A_lvl_6_i == phase_stop_4
                                    c_val = c_val + true
                                end
                                A_lvl_4_q += A_lvl_4_i == phase_stop_4
                                A_lvl_6_q += A_lvl_6_i == phase_stop_4
                                #k = phase_stop_4 + 1
                            #end
                        end
                        k = phase_stop_3 + 1
                    end

                    A_lvl_2_q += 1
                else
                end
                j = phase_stop_2 + 1
            end
            j = phase_stop + 1
        end
        j_start = j
        phase_start_8 = max(j_start)
        phase_stop_8 = min(j_stop)
        if phase_stop_8 >= phase_start_8
            j_4 = j
            j = phase_stop_8 + 1
        end
    end
    (c = (Scalar){0, Int64}(c_val),)
end

main()