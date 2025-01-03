using Finch
using TensorMarket
using JSON
function bellmanford_graphblas(A)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    gb_path = joinpath(@__DIR__, "../deps/LAGraph/build/experimental/test/test_BF")
    withenv("MATRIX"=>A_path, "HAS_NEGATIVE_CYCLES"=>"false", "HAS_INTEGER_WEIGHTS"=>"false") do
        run(`$gb_path` > output.txt)
    end
    file_contents = read("output.txt", String)
    pattern = r"BF_full2\s+time:\s+([0-9]+\.[0-9]+e[+-]?[0-9]+)"
    match = match(pattern, file_contents)
    time = parse(Float64, match.captures[1]) 
    return (;time=time*10^-9, y=nothing)
end
