"""
    ScopeContext

A context for managing variable bindings and tensor modes.
"""
@kwdef struct ScopeContext
    bindings::Dict{FinchNode, FinchNode} = Dict{FinchNode, FinchNode}()
    modes::Dict{Any, Any} = Dict()
    defs = Set()
end

"""
    get_binding(ctx, var)

Get the binding of a variable in the context.
"""
function get_binding(ctx::ScopeContext, var)
    @assert haskey(ctx.bindings, var) "$var unbound"
    ctx.bindings[var]
end
"""
    has_binding(ctx, var)

Check if a variable is bound in the context.
"""
has_binding(ctx::ScopeContext, var) = haskey(ctx.bindings, var)
"""
    set_binding!(ctx, var, val)

Set the binding of a variable in the context.
"""
set_binding!(ctx::ScopeContext, var, val) = ctx.bindings[var] = val
"""
    get_binding(ctx, var, val)

Get the binding of a variable in the context, or return a default value.
"""
get_binding(ctx::AbstractCompiler, var, val) = has_binding(ctx, var) ? get_binding(ctx, var) : val
"""
    get_binding!(ctx, var, val)

Get the binding of a variable in the context, or set it to a default value.
"""
get_binding!(ctx::AbstractCompiler, var, val) = has_binding(ctx, var) ? get_binding(ctx, var) : set_binding!(ctx, var, val)

"""
    set_declared!(ctx, var, val)

Mark a tensor variable as declared in the context.
"""
function set_declared!(ctx::ScopeContext, var, val)
    @assert var.kind === variable
    @assert get(ctx.modes, var, reader) === reader
    push!(ctx.defs, var)
    set_binding!(ctx, var, val)
    ctx.modes[var] = updater
end

"""
    set_frozen!(ctx, var, val)

Mark a tensor variable as frozen in the context.
"""
function set_frozen!(ctx::ScopeContext, var, val)
    @assert var.kind === variable
    @assert ctx.modes[var] === updater
    set_binding!(ctx, var, val)
    ctx.modes[var] = reader
end

"""
    set_thawed!(ctx, var, val)

Mark a tensor variable as thawed in the context.
"""
function set_thawed!(ctx::ScopeContext, var, val)
    @assert var.kind === variable
    @assert get(ctx.modes, var, reader) === reader
    set_binding!(ctx, var, val)
    ctx.modes[var] = updater
end
"""
    get_tensor_mode(ctx, var)

Get the mode of a tensor variable in the context.
"""
get_tensor_mode(ctx::ScopeContext, var) = get(ctx.modes, var, reader)

"""
    open_scope(f, ctx)

Call the function `f(ctx_2)` in a new scope `ctx_2`.
"""
function open_scope(f::F, ctx::ScopeContext) where {F}
    ctx_2 = ScopeContext(bindings=copy(ctx.bindings), modes=ctx.modes)
    res = f(ctx_2)
    for tns in ctx_2.defs
        pop!(ctx.modes, tns, nothing)
    end
    res
end