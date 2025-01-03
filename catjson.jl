#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include(joinpath(@__DIR__, "deps/diagnostics.jl"))
print_diagnostics()

using JSON

res = []
for arg in ARGS
    append!(res, JSON.parsefile(arg))
end
println(json(res, 4))
