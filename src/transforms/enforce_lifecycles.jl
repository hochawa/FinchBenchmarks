@kwdef struct EnforceLifecyclesVisitor
    uses = OrderedDict()
    scoped_uses = Dict()
    global_uses = uses
    modes = Dict()
end

struct EnforceLifecyclesError
    msg
end

function open_scope(ctx::EnforceLifecyclesVisitor, prgm)
    ctx_2 = EnforceLifecyclesVisitor(;kwfields(ctx)..., uses=Dict())
    close_scope(prgm, ctx_2)
end

function getmodified(node::FinchNode)
    if node.kind === block
        return unique(mapreduce(getmodified, vcat, node.bodies, init=[]))
    elseif node.kind === declare || node.kind === thaw
        return [node.tns]
    else
        return []
    end
end

function close_scope(prgm, ctx::EnforceLifecyclesVisitor)
    prgm = ctx(prgm)
    for tns in getmodified(prgm)
        if ctx.modes[tns] !== reader
            prgm = block(prgm, freeze(tns))
        end
    end
    prgm
end

"""
    enforce_lifecycles(prgm)

A transformation which adds `freeze` and `thaw` statements automatically to
tensor roots, depending on whether they appear on the left or right hand side.
"""
function enforce_lifecycles(prgm)
    close_scope(prgm, EnforceLifecyclesVisitor())
end

#assumes arguments to prgm have been visited already and their uses collected
function open_stmt(prgm, ctx::EnforceLifecyclesVisitor)
    for (tns, mode) in ctx.uses
        cur_mode = get(ctx.modes, tns, reader)
        if mode === reader && cur_mode === updater
            prgm = block(freeze(tns), prgm)
        elseif mode === updater && cur_mode === reader
            prgm = block(thaw(tns), prgm)
        end
        ctx.modes[tns] = mode
    end
    empty!(ctx.uses)
    prgm
end

function (ctx::EnforceLifecyclesVisitor)(node::FinchNode)
    if node.kind === loop
        open_stmt(loop(node.idx, ctx(node.ext), open_scope(ctx, node.body)), ctx)
    elseif node.kind === sieve
        open_stmt(sieve(ctx(node.cond), open_scope(ctx, node.body)), ctx)
    elseif node.kind === define
        open_stmt(define(node.lhs, ctx(node.rhs), open_scope(ctx, node.body)), ctx)
    elseif node.kind === declare
        ctx.scoped_uses[node.tns] = ctx.uses
        if get(ctx.modes, node.tns, reader) === updater
            node = block(freeze(node.tns), node)
        end
        ctx.modes[node.tns] = updater
        node
    elseif node.kind === freeze
        haskey(ctx.modes, node.tns) || throw(EnforceLifecyclesError("cannot freeze undefined $(node.tns)"))
        ctx.modes[node.tns] === reader && return block()
        ctx.modes[node.tns] = reader
        node
    elseif node.kind === thaw
        get(ctx.modes, node.tns, reader) === updater && return block()
        ctx.modes[node.tns] = updater
        node
    elseif node.kind === assign
        return open_stmt(assign(ctx(node.lhs), ctx(node.op), ctx(node.rhs)), ctx)
    elseif node.kind === access
        idxs = map(ctx, node.idxs)
        uses = get(ctx.scoped_uses, getroot(node.tns), ctx.global_uses)
        get(uses, getroot(node.tns), node.mode.val) !== node.mode.val &&
            throw(EnforceLifecyclesError("cannot mix reads and writes to $(node.tns) outside of defining scope (hint: perhaps add a declaration like `var .= 0` or use an updating operator like `var += 1`)"))
        uses[getroot(node.tns)] = node.mode.val
        access(node.tns, node.mode, idxs...)
    elseif node.kind === yieldbind
        args_2 = map(node.args) do arg
            uses = get(ctx.scoped_uses, getroot(arg), ctx.global_uses)
            get(uses, getroot(arg), reader) !== reader &&
                throw(EnforceLifecyclesError("cannot return $(arg) outside of defining scope"))
            uses[getroot(arg)] = reader
            ctx(arg)
        end
        open_stmt(yieldbind(args_2...), ctx)
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end
