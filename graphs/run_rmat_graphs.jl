#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include("../deps/diagnostics.jl")
print_diagnostics()

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using SparseArrays
using Printf
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using MatrixDepot
using Finch
using Graphs
using SimpleWeightedGraphs
#using LightGraphs

s = ArgParseSettings("Run graph experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "graphs_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "yang"
        #default = "willow"
end

parsed_args = parse_args(ARGS, s)

include("datasets.jl")
include("shortest_paths.jl")
include("bfs.jl")

function bfs_finch_push_pull(mtx)
    A = pattern!(Tensor(SparseMatrixCSC(mtx)))
    AT = pattern!(Tensor(permutedims(SparseMatrixCSC(mtx))))
    time = @belapsed bfs_finch_kernel($A, $AT, 1)
    output = bfs_finch_kernel(A, AT, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function bfs_finch_push_only(mtx)
    A = pattern!(Tensor(SparseMatrixCSC(mtx)))
    AT = pattern!(Tensor(permutedims(SparseMatrixCSC(mtx))))
    time = @belapsed bfs_finch_kernel($A, $AT, 1, 2)
    output = bfs_finch_kernel(A, AT, 1, 2)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function bfs_graphs(mtx)
    A = SimpleDiGraph(transpose(mtx))
    time = @belapsed Graphs.bfs_parents($A, 1)
    output = Graphs.bfs_parents(A, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function bellmanford_finch(mtx)
    A = redefault!(Tensor(SparseMatrixCSC{Float64}(mtx)), Inf)
    time = @belapsed bellmanford_finch_kernel($A, 1)
    output = bellmanford_finch_kernel(A, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function bellmanford_graphs(mtx)
    A = SimpleWeightedDiGraph(transpose(SparseMatrixCSC{Float64}(mtx)))
    time = @belapsed Graphs.bellman_ford_shortest_paths($A, 1)
    output = Graphs.bellman_ford_shortest_paths(A, 1)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

function check_bfs(A, src, res_parent, ref_parent)
    g = SimpleDiGraph(transpose(A))
    ref_levels = gdistances(g, src)
    for i in 1:nv(g)
        if ref_parent[i] == 0
            @assert res_parent[i] == 0
        elseif ref_parent[i] == i
            @assert res_parent[i] == i
        else
            @assert ref_levels[res_parent[i]] == ref_levels[i] - 1
        end
    end
    return true
end

function check_bellman(A, src, res, ref)
    n = length(ref.dists)
    for i in 1:n
        if ref.dists[i] != res.dists[i]
            @info "dists" i res.dists[i] ref.dists[i]
        end
        if ref.parents[i] != 0
            @assert A[i, res.parents[i]] + ref.dists[res.parents[i]] == ref.dists[i]
        end
    end
    return true
end

function generate_rmat_graph(scale, edge_factor, a=0.57, b=0.19, c=0.19, d=0.05)
	num_vertices = 2^scale
	num_edges = num_vertices * edge_factor

	edges = Set{Tuple{Int64, Int64}}()  # To store the edges without duplicates
 	edge_weights = Dict{Tuple{Int64, Int64}, Float64}()

	while length(edges) < num_edges
		src, dst = 1, 1
		stride = num_vertices /2

		for _ in 1:scale
			rand_val = rand()
			if rand_val < a
			elseif rand_val < a + b
				dst += stride
			elseif rand_val < a + b + c
				src += stride
			else
				src += stride
				dst += stride
			end
			stride = stride / 2
		end

		if src != dst
			push!(edges, (src, dst))
			edge_weights[(src, dst)] = rand()  # Assign a random weight to the edge (between 0 and 1)
		end

	end

	g = SimpleWeightedGraph(num_vertices)
	for (src, dst) in edges
		add_edge!(g, src, dst, edge_weights[(src, dst)])

	end

	return g, edge_weights
end

function graph_to_sparsematrix_with_weights(g::SimpleWeightedGraph, edge_weights::Dict{Tuple{Int64, Int64}, Float64})
    num_vertices = nv(g)
    rows = Vector{Int64}()
    cols = Vector{Int64}()
    vals = Vector{Float64}()

    for edge in edges(g)
	    src_vertex = src(edge)
	    dst_vertex = dst(edge)
	weight = get(edge_weights, (src_vertex, dst_vertex), 0.0) 
        push!(rows, src_vertex)
        push!(cols, dst_vertex)
	push!(vals, weight)
        #push!(vals, edge_weights[(src, dst)])  # Use the edge weights
    end

    return sparse(rows, cols, vals, num_vertices, num_vertices)  # Create SparseMatrixCSC
end

results = []


for mtx in datasets[parsed_args["dataset"]]
    scale = 0
    edge_factor = 0
    if mtx == "rmat_s22_e64"
	    scale = 22
	    edge_factor = 64
    elseif mtx == "rmat_s23_e32"
	    scale = 23
	    edge_factor = 32
    elseif mtx == "rmat_s24_e16"
	    scale = 24
	    edge_factor = 16
    end

    if scale > 0
	    g, edge_weights = generate_rmat_graph(scale, edge_factor)
	    #src = edges[:, 1]
	    #dst = edges[:, 2]
	    #n = 2^scale
	    #vals = rand(Float64, length(src))
	    #A = SparseMatrixCSC(n, n, src, dst, vals)
	    A= graph_to_sparsematrix_with_weights(g, edge_weights)

    else
	    A = SparseMatrixCSC(matrixdepot(mtx))
    end
    (n, n) = size(A)
    for (op_name, check, methods) in [
        ("bfs",
            check_bfs,
            [
                "Graphs.jl" => bfs_graphs,
                "finch_push_pull" => bfs_finch_push_pull,
                "finch_push_only" => bfs_finch_push_only,
            ]
        ),
        ("bellmanford",
            check_bellman,
            [
                "Graphs.jl" => bellmanford_graphs,
                "Finch" => bellmanford_finch,
            ]
        ),
    ]
        @info "testing" op_name mtx
        reference = nothing
        for (key, method) in methods
            result = method(A)

            time = result.time
            reference = something(reference, result.output)

            check(A, 1, result.output, reference) || @warn("incorrect result")

            # res.y == y_ref || @warn("incorrect result")
            @info "results" key result.time result.mem
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "operation" => op_name,
                "matrix" => mtx,
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end
