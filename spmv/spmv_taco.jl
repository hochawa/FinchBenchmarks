using Finch
using TensorMarket
using JSON
function spmv_taco_helper(args, A, x)
    tmpdir = mktempdir(@__DIR__, prefix="experiment_")
    A_path = joinpath(tmpdir, "A.ttx")
    x_path = joinpath(tmpdir, "x.ttx")
    y_path = joinpath(tmpdir, "y.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
    fwrite(x_path, Tensor(Dense(Element(0.0)), x))
    taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        spmv_path = joinpath(@__DIR__, "spmv_taco")
        run(`$spmv_path -i $tmpdir -o $tmpdir -- $args`)
    end
    y = fread(y_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, y=y)
end

spmv_taco_row_maj(y, A, x) = spmv_taco_helper(`--schedule row-major`, A, x)
spmv_taco_col_maj(y, A, x) = spmv_taco_helper(`--schedule column-major`, permutedims(A), x)

has_taco() = isfile(joinpath(@__DIR__, "spmv_taco"))