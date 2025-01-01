#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include("../deps/diagnostics.jl")
print_diagnostics()

using Finch
using TensorMarket

mkpath("data")

for N = 7:13
    m = n = 2^N
    nnz = 4*m
    fwrite("data/rand_$(m).ttx", fsprand(m, n, nnz))
end


