using Base: nothing_sentinel
#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using Random

Random.seed!(1234)

# Parsing Arguments
s = ArgParseSettings("Run Parallel SpAdd Experiments.")
@add_arg_table! s begin
    "--output", "-o"
    arg_type = String
    help = "output file path"
    "--dataset", "-d"
    arg_type = String
    help = "dataset keyword"
    "--method", "-m"
    arg_type = String
    help = "method keyword"
    "--accuracy-check", "-a"
    action = :store_true
    help = "check method accuracy"
end
parsed_args = parse_args(ARGS, s)

# Mapping from dataset types to datasets
datasets = Dict(
    "uniform" => [
        OrderedDict("size" => 1_000, "sparsity" => 0.1),
        OrderedDict("size" => 10_000, "sparsity" => 0.1),
        OrderedDict("size" => 1_000_000, "sparsity" => 3_000_000),
    ],
    "FEMLAB" => [
        "FEMLAB/poisson3Da",
        "FEMLAB/poisson3Db",
    ],
)

# Mapping from method keywords to methods
include("serial_default_implementation.jl")
include("parallel_default_implementation.jl")
include("separated_memory_concatenate_results.jl")

methods = OrderedDict(
    "serial_default_implementation" => serial_default_implementation_add,
    "parallel_default_implementation" => parallel_default_implementation_add,
    "separated_memory_concatenate_results" => separated_memory_concatenate_results_add,
)

if !isnothing(parsed_args["method"])
    method_name = parsed_args["method"]
    @assert haskey(methods, method_name) "Unrecognize method for $method_name"
    methods = OrderedDict(
        method_name => methods[method_name]
    )
end

function calculate_results(dataset, mtxs, results)
    for mtx in mtxs
        # Get relevant matrix
        if dataset == "uniform"
            A = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
            B = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
        elseif dataset == "FEMLAB"
            A = matrixdepot(mtx)
            row_permutation = randperm(size(A, 1))
            col_permutation = randperm(size(A, 2))
            B = A[row_permutation, col_permutation]
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        for (key, method) in methods
            result = method(A, B)

            if parsed_args["accuracy-check"]
                # Check the result of the multiplication
                serial_default_implementation_result = serial_default_implementation_add(A, B)
                @assert result.C == serial_default_implementation_result.C "Incorrect result for $key"
            end

            # Write result
            time = result.time
            @info "result for $key on $mtx" time
            push!(results, OrderedDict(
                "time" => time,
                "n_threads" => Threads.nthreads(),
                "method" => key,
                "dataset" => dataset,
                "matrix" => mtx,
            ))

            if isnothing(parsed_args["output"])
                write("results/spadd_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
            else
                write(parsed_args["output"], JSON.json(results, 4))
            end
        end
    end
end

results = []
if isnothing(parsed_args["dataset"])
    for (dataset, mtxs) in datasets
        calculate_results(dataset, mtxs, results)
    end
else
    dataset = parsed_args["dataset"]
    mtxs = datasets[dataset]
    calculate_results(dataset, mtxs, results)
end


