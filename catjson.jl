using JSON

res = []
for arg in ARGS
    push!(res, JSON.parsefile(arg))
end
println(json(res, 4))
