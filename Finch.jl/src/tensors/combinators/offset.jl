struct OffsetArray{Delta<:Tuple, Body} <: AbstractCombinator
    body::Body
    delta::Delta
end

Base.show(io::IO, ex::OffsetArray) = print(io, "OffsetArray($(ex.body), $(ex.delta)")

labelled_show(io::IO, ::OffsetArray) =
    print(io, "OffsetArray [$(join(map(d -> ":+$d", ex.delta), ", "))]")

labelled_children(ex::OffsetArray) = [LabelledTree(ex.body)]

struct VirtualOffsetArray <: AbstractVirtualCombinator
    body
    delta
end

is_injective(ctx, lvl::VirtualOffsetArray) = is_injective(ctx, lvl.body)
is_atomic(ctx, lvl::VirtualOffsetArray) = is_atomic(ctx, lvl.body)
is_concurrent(ctx, lvl::VirtualOffsetArray) = is_concurrent(ctx, lvl.body)

Base.show(io::IO, ex::VirtualOffsetArray) = Base.show(io, MIME"text/plain"(), ex)

Base.summary(io::IO, ex::VirtualOffsetArray) = print(io, "VOffset($(summary(ex.body)), $(ex.delta))")

FinchNotation.finch_leaf(x::VirtualOffsetArray) = virtual(x)

function virtualize(ctx, ex, ::Type{OffsetArray{Delta, Body}}) where {Delta, Body}
    delta = map(enumerate(Delta.parameters)) do (n, param)
        virtualize(ctx, :($ex.delta[$n]), param)
    end
    VirtualOffsetArray(virtualize(ctx, :($ex.body), Body), delta)
end

"""
    offset(tns, delta...)

Create an `OffsetArray` such that `offset(tns, delta...)[i...] == tns[i .+ delta...]`.
The dimensions declared by an OffsetArray are shifted, so that `size(offset(tns, delta...)) == size(tns) .+ delta`.
"""
offset(body, delta...) = OffsetArray(body, delta)
virtual_call(ctx, ::typeof(offset), body, delta...) = VirtualOffsetArray(body, delta)

unwrap(ctx, arr::VirtualOffsetArray, var) = call(offset, unwrap(ctx, arr.body, var), arr.delta...)

lower(ctx::AbstractCompiler, tns::VirtualOffsetArray, ::DefaultStyle) = :(OffsetArray($(ctx(tns.body)), $(ctx(tns.delta))))

virtual_size(ctx::AbstractCompiler, arr::VirtualOffsetArray) =
    map(zip(virtual_size(ctx, arr.body), arr.delta)) do (dim, delta)
        shiftdim(dim, call(-, delta))
    end

function virtual_resize!(ctx::AbstractCompiler, arr::VirtualOffsetArray, dims...)
    dims_2 = map(zip(dims, arr.delta)) do (dim, delta)
        shiftdim(dim, delta)
    end
    virtual_resize!(ctx, arr.body, dims_2...)
end

virtual_fill_value(ctx::AbstractCompiler, arr::VirtualOffsetArray) = virtual_fill_value(ctx, arr.body)

instantiate(ctx, arr::VirtualOffsetArray, mode) =
    VirtualOffsetArray(instantiate(ctx, arr.body, mode), arr.delta)

get_style(ctx, node::VirtualOffsetArray, root) = get_style(ctx, node.body, root)

function popdim(node::VirtualOffsetArray)
    if length(node.delta) == 1
        return node.body
    else
        return VirtualOffsetArray(node.body, node.delta[1:end-1])
    end
end

truncate(ctx, node::VirtualOffsetArray, ext, ext_2) =
    VirtualOffsetArray(truncate(ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])), node.delta)

get_point_body(ctx, node::VirtualOffsetArray, ext, idx) =
    pass_nothing(get_point_body(ctx, node.body, shiftdim(ext, node.delta[end]), call(+, idx, node.delta[end]))) do body_2
        popdim(VirtualOffsetArray(body_2, node.delta))
    end

unwrap_thunk(ctx, node::VirtualOffsetArray) = VirtualOffsetArray(unwrap_thunk(ctx, node.body), node.delta)

get_run_body(ctx, node::VirtualOffsetArray, ext) =
    pass_nothing(get_run_body(ctx, node.body, shiftdim(ext, node.delta[end]))) do body_2
        popdim(VirtualOffsetArray(body_2, node.delta))
    end

get_acceptrun_body(ctx, node::VirtualOffsetArray, ext) =
    pass_nothing(get_acceptrun_body(ctx, node.body, shiftdim(ext, node.delta[end]))) do body_2
        popdim(VirtualOffsetArray(body_2, node.delta))
    end

get_sequence_phases(ctx, node::VirtualOffsetArray, ext) =
    map(get_sequence_phases(ctx, node.body, shiftdim(ext, node.delta[end]))) do (keys, body)
        return keys => VirtualOffsetArray(body, node.delta)
    end

phase_body(ctx, node::VirtualOffsetArray, ext, ext_2) =
    VirtualOffsetArray(phase_body(ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])), node.delta)
phase_range(ctx, node::VirtualOffsetArray, ext) =
    shiftdim(phase_range(ctx, node.body, shiftdim(ext, node.delta[end])), call(-, node.delta[end]))

get_spike_body(ctx, node::VirtualOffsetArray, ext, ext_2) =
    VirtualOffsetArray(get_spike_body(ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])), node.delta)
get_spike_tail(ctx, node::VirtualOffsetArray, ext, ext_2) =
    VirtualOffsetArray(get_spike_tail(ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])), node.delta)

visit_fill_leaf_leaf(node, tns::VirtualOffsetArray) =
    visit_fill_leaf_leaf(node, tns.body)
visit_simplify(node::VirtualOffsetArray) =
    VirtualOffsetArray(visit_simplify(node.body), node.delta)

get_switch_cases(ctx, node::VirtualOffsetArray) = map(get_switch_cases(ctx, node.body)) do (guard, body)
    guard => VirtualOffsetArray(body, node.delta)
end

stepper_range(ctx, node::VirtualOffsetArray, ext) = shiftdim(stepper_range(ctx, node.body, shiftdim(ext, node.delta[end])), call(-, node.delta[end]))
stepper_body(ctx, node::VirtualOffsetArray, ext, ext_2) = VirtualOffsetArray(stepper_body(ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])), node.delta)
stepper_seek(ctx, node::VirtualOffsetArray, ext) = stepper_seek(ctx, node.body, shiftdim(ext, node.delta[end]))

jumper_range(ctx, node::VirtualOffsetArray, ext) = shiftdim(jumper_range(ctx, node.body, shiftdim(ext, node.delta[end])), call(-, node.delta[end]))
jumper_body(ctx, node::VirtualOffsetArray, ext, ext_2) = VirtualOffsetArray(jumper_body(ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])), node.delta)
jumper_seek(ctx, node::VirtualOffsetArray, ext) = jumper_seek(ctx, node.body, shiftdim(ext, node.delta[end]))

short_circuit_cases(ctx, node::VirtualOffsetArray, op) =
    map(short_circuit_cases(ctx, node.body, op)) do (guard, body)
        guard => VirtualOffsetArray(body, node.delta)
    end

getroot(tns::VirtualOffsetArray) = getroot(tns.body)

unfurl(ctx, tns::VirtualOffsetArray, ext, mode, proto) =
    VirtualOffsetArray(unfurl(ctx, tns.body, shiftdim(ext, tns.delta[end]), mode, proto), tns.delta)

function lower_access(ctx::AbstractCompiler, tns::VirtualOffsetArray, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::VirtualOffsetArray, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
