"""
    DenseLevel{[Ti=Int]}(lvl, [dim])

A subfiber of a dense level is an array which stores every slice `A[:, ..., :,
i]` as a distinct subfiber in `lvl`. Optionally, `dim` is the size of the last
dimension. `Ti` is the type of the indices used to index the level.

```jldoctest
julia> ndims(Tensor(Dense(Element(0.0))))
1

julia> ndims(Tensor(Dense(Dense(Element(0.0)))))
2

julia> tensor_tree(Tensor(Dense(Dense(Element(0.0))), [1 2; 3 4]))
2×2-Tensor
└─ Dense [:,1:2]
   ├─ [:, 1]: Dense [1:2]
   │  ├─ [1]: 1.0
   │  └─ [2]: 3.0
   └─ [:, 2]: Dense [1:2]
      ├─ [1]: 2.0
      └─ [2]: 4.0
```
"""
struct DenseLevel{Ti, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
end
DenseLevel(lvl) = DenseLevel{Int}(lvl)
#DenseLevel(lvl, shape::Ti) where {Ti} = DenseLevel{Ti}(lvl, shape)
DenseLevel{Ti}(lvl) where {Ti} = DenseLevel{Ti}(lvl, zero(Ti))
DenseLevel{Ti}(lvl::Lvl, shape) where {Ti, Lvl} = DenseLevel{Ti, Lvl}(lvl, shape)

const Dense = DenseLevel

Base.summary(lvl::Dense) = "Dense($(summary(lvl.lvl)))"

similar_level(lvl::DenseLevel, fill_value, eltype::Type, dims...) =
    Dense(similar_level(lvl.lvl, fill_value, eltype, dims[1:end-1]...), dims[end])

function postype(::Type{DenseLevel{Ti, Lvl}}) where {Ti, Lvl}
    return postype(Lvl)
end

function moveto(lvl::DenseLevel{Ti}, device) where {Ti}
    return DenseLevel{Ti}(moveto(lvl.lvl, device), lvl.shape)
end

pattern!(lvl::DenseLevel{Ti, Lvl}) where {Ti, Lvl} =
    DenseLevel{Ti}(pattern!(lvl.lvl), lvl.shape)

set_fill_value!(lvl::DenseLevel{Ti}, init) where {Ti} =
    DenseLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape)

Base.resize!(lvl::DenseLevel{Ti}, dims...) where {Ti} =
    DenseLevel{Ti}(resize!(lvl.lvl, dims[1:end-1]...), dims[end])

@inline level_ndims(::Type{<:DenseLevel{Ti, Lvl}}) where {Ti, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::DenseLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::DenseLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:DenseLevel{Ti, Lvl}}) where {Ti, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:DenseLevel{Ti, Lvl}}) where {Ti, Lvl} = level_fill_value(Lvl)
data_rep_level(::Type{<:DenseLevel{Ti, Lvl}}) where {Ti, Lvl} = DenseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:DenseLevel})() = fbr
function (fbr::SubFiber{<:DenseLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    q = (p - 1) * lvl.shape + idxs[end]
    fbr_2 = SubFiber(lvl.lvl, q)
    fbr_2(idxs[1:end-1]...)
end

function countstored_level(lvl::DenseLevel, pos)
    countstored_level(lvl.lvl, pos * lvl.shape)
end

function Base.show(io::IO, lvl::DenseLevel{Ti}) where {Ti}
    if get(io, :compact, false)
        print(io, "Dense(")
    else
        print(io, "Dense{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(io, lvl.shape)
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:DenseLevel}) =
    print(io, "Dense [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:DenseLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    map(1:lvl.shape) do idx
        LabelledTree(cartesian_label([range_label() for _ = 1:ndims(fbr) - 1]..., idx), SubFiber(lvl.lvl, (pos - 1) * lvl.shape + idx))
    end
end

mutable struct VirtualDenseLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    shape
end

is_level_injective(ctx, lvl::VirtualDenseLevel) = [is_level_injective(ctx, lvl.lvl)..., true]
function is_level_atomic(ctx, lvl::VirtualDenseLevel)
    (data, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([data; atomic], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualDenseLevel)
    (data, concurrent) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; concurrent], concurrent)
end

function virtualize(ctx, ex, ::Type{DenseLevel{Ti, Lvl}}, tag=:lvl) where {Ti, Lvl}
    sym = freshen(ctx, tag)
    shape = value(:($sym.shape), Ti)
    push_preamble!(ctx, quote
        $sym = $ex
    end)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    VirtualDenseLevel(lvl_2, sym, Ti, shape)
end
function lower(ctx::AbstractCompiler, lvl::VirtualDenseLevel, ::DefaultStyle)
    quote
        $DenseLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
        )
    end
end

Base.summary(lvl::VirtualDenseLevel) = "Dense($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualDenseLevel)
    ext = Extent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualDenseLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualDenseLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualDenseLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualDenseLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualDenseLevel, pos, init)
    lvl.lvl = declare_level!(ctx, lvl.lvl, call(*, pos, lvl.shape), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualDenseLevel, pos_start, pos_stop)
    qos_start = call(+, call(*, call(-, pos_start, lvl.Ti(1)), lvl.shape), 1)
    qos_stop = call(*, pos_stop, lvl.shape)
    assemble_level!(ctx, lvl.lvl, qos_start, qos_stop)
end

supports_reassembly(::VirtualDenseLevel) = true
function reassemble_level!(ctx, lvl::VirtualDenseLevel, pos_start, pos_stop)
    qos_start = call(+, call(*, call(-, pos_start, lvl.Ti(1)), lvl.shape), 1)
    qos_stop = call(*, pos_stop, lvl.shape)
    reassemble_level!(ctx, lvl.lvl, qos_start, qos_stop)
    lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualDenseLevel, pos)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, call(*, pos, lvl.shape))
    return lvl
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualDenseLevel, pos)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, call(*, pos, lvl.shape))
    return lvl
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualDenseLevel, arch)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

struct DenseTraversal
    fbr
    subfiber_ctr
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualDenseLevel}, ext, mode, proto) =
    unfurl(ctx, DenseTraversal(fbr, VirtualSubFiber), ext, mode, proto)
unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualDenseLevel}, ext, mode, proto) =
    unfurl(ctx, DenseTraversal(fbr, (lvl, pos) -> VirtualHollowSubFiber(lvl, pos, fbr.dirty)), ext, mode, proto)

function unfurl(ctx, trv::DenseTraversal, ext, mode, ::Union{typeof(defaultread), typeof(follow), typeof(defaultupdate), typeof(laminate), typeof(extrude)})
    (lvl, pos) = (trv.fbr.lvl, trv.fbr.pos)
    tag = lvl.ex
    Ti = lvl.Ti

    q = freshen(ctx, tag, :_q)

    Lookup(
        body = (ctx, i) -> Thunk(
            preamble = quote
                $q = ($(ctx(pos)) - $(Ti(1))) * $(ctx(lvl.shape)) + $(ctx(i))
            end,
            body = (ctx) -> instantiate(ctx, trv.subfiber_ctr(lvl.lvl, value(q, lvl.Ti)), mode)
        )
    )
end


