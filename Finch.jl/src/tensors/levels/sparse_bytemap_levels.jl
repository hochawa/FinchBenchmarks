"""
    SparseByteMapLevel{[Ti=Int], [Ptr, Tbl]}(lvl, [dims])

Like the [`SparseListLevel`](@ref), but a dense bitmap is used to encode
which slices are stored. This allows the ByteMap level to support random access.

`Ti` is the type of the last tensor index, and `Tp` is the type used for
positions in the level.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseByteMap(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseByteMap (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseByteMap (0.0) [1:3]
   └─ [:, 3]: SparseByteMap (0.0) [1:3]
      ├─ [1]: 0.0
      └─ [3]: 0.0

julia> tensor_tree(Tensor(SparseByteMap(SparseByteMap(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseByteMap (0.0) [:,1:3]
   ├─ [:, 1]: SparseByteMap (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   └─ [:, 3]: SparseByteMap (0.0) [1:3]
```
"""
struct SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    tbl::Tbl
    srt::Srt
end
const SparseByteMap = SparseByteMapLevel
SparseByteMapLevel(lvl::Lvl) where {Lvl} = SparseByteMapLevel{Int}(lvl)
SparseByteMapLevel(lvl, shape, args...) = SparseByteMapLevel{typeof(shape)}(lvl, shape, args...)
SparseByteMapLevel{Ti}(lvl) where {Ti} = SparseByteMapLevel{Ti}(lvl, zero(Ti))
SparseByteMapLevel{Ti}(lvl, shape) where {Ti} =
    SparseByteMapLevel{Ti}(lvl, shape, postype(lvl)[1], Bool[], Tuple{postype(lvl), Ti}[])
SparseByteMapLevel{Ti}(lvl::Lvl, shape, ptr::Ptr, tbl::Tbl, srt::Srt) where {Ti, Lvl, Ptr, Tbl, Srt} =
    SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}(lvl, shape, ptr, tbl, srt)

Base.summary(lvl::SparseByteMapLevel) = "SparseByteMap($(summary(lvl.lvl)))"
similar_level(lvl::SparseByteMapLevel, fill_value, eltype::Type, dims...) =
    SparseByteMap(similar_level(lvl.lvl, fill_value, eltype, dims[1:end-1]...), dims[end])

function postype(::Type{SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}}) where {Ti, Ptr, Tbl, Srt, Lvl}
    return postype(Lvl)
end

function moveto(lvl::SparseByteMapLevel{Ti}, device) where {Ti}
    lvl_2 = moveto(lvl.lvl, device)
    ptr_2 = moveto(lvl.ptr, device)
    tbl_2 = moveto(lvl.tbl, device)
    srt_2 = moveto(lvl.srt, device)
    return  SparseByteMapLevel{Ti}(lvl_2, lvl.shape, ptr_2, tbl_2, srt_2)
end


pattern!(lvl::SparseByteMapLevel{Ti}) where {Ti} =
    SparseByteMapLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.tbl, lvl.srt)

set_fill_value!(lvl::SparseByteMapLevel{Ti}, init) where {Ti} =
    SparseByteMapLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.tbl, lvl.srt)

Base.resize!(lvl::SparseByteMapLevel{Ti}, dims...) where {Ti} =
    SparseByteMapLevel{Ti}(resize!(lvl.lvl, dims[1:end-1]...), dims[end], lvl.ptr, lvl.tbl, lvl.srt)

function countstored_level(lvl::SparseByteMapLevel, pos)
    countstored_level(lvl.lvl, pos * lvl.shape)
end

function Base.show(io::IO, lvl::SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl},) where {Ti, Ptr, Tbl, Srt, Lvl}
    if get(io, :compact, false)
        print(io, "SparseByteMap(")
    else
        print(io, "SparseByteMap{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo=>Ti), lvl.shape)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.ptr)
        print(io, ", ")
        show(io, lvl.tbl)
        print(io, ", ")
        show(io, lvl.srt)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:SparseByteMapLevel}) =
    print(io, "SparseByteMap (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:SparseByteMapLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:lvl.ptr[pos + 1] - 1) do qos
        LabelledTree(cartesian_label([range_label() for _ = 1:ndims(fbr) - 1]..., lvl.srt[qos][2]), SubFiber(lvl.lvl, qos))
    end
end

@inline level_ndims(::Type{<:SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}}) where {Ti, Ptr, Tbl, Srt, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseByteMapLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseByteMapLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}}) where {Ti, Ptr, Tbl, Srt, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}}) where {Ti, Ptr, Tbl, Srt, Lvl}= level_fill_value(Lvl)
data_rep_level(::Type{<:SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}}) where {Ti, Ptr, Tbl, Srt, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseByteMapLevel})() = fbr
function (fbr::SubFiber{<:SparseByteMapLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    q = (p - 1) * lvl.shape + idxs[end]
    if lvl.tbl[q]
        fbr_2 = SubFiber(lvl.lvl, q)
        fbr_2(idxs[1:end-1]...)
    else
        fill_value(fbr)
    end
end

mutable struct VirtualSparseByteMapLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    ptr
    tbl
    srt
    shape
    qos_fill
    qos_stop
end

is_level_injective(ctx, lvl::VirtualSparseByteMapLevel) = [is_level_injective(ctx, lvl.lvl)..., false]
function is_level_atomic(ctx, lvl::VirtualSparseByteMapLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseByteMapLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(ctx, ex, ::Type{SparseByteMapLevel{Ti, Ptr, Tbl, Srt, Lvl}}, tag=:lvl) where {Ti, Ptr, Tbl, Srt, Lvl}
    sym = freshen(ctx, tag)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    ptr = freshen(ctx, tag, :_ptr)
    tbl = freshen(ctx, tag, :_tbl)
    srt = freshen(ctx, tag, :_srt)
    push_preamble!(ctx, quote
        $sym = $ex
        $ptr = $ex.ptr
        $tbl = $ex.tbl
        $srt = $ex.srt
        $qos_stop = $qos_fill = length($sym.srt)
    end)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    VirtualSparseByteMapLevel(lvl_2, sym, Ti, ptr, tbl, srt, shape, qos_fill, qos_stop)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, ::DefaultStyle)
    quote
        $SparseByteMapLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.tbl),
            $(lvl.srt),
        )
    end
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, arch)
    ptr_2 = freshen(ctx, lvl.ptr)
    tbl_2 = freshen(ctx, lvl.tbl)
    srt_2 = freshen(ctx, lvl.srt)
    push_preamble!(ctx, quote
        $ptr_2 = $(lvl.ptr)
        $tbl_2 = $(lvl.tbl)
        $srt_2 = $(lvl.srt)
        $(lvl.ptr) = moveto($(lvl.ptr), $(ctx(arch)))
        $(lvl.tbl) = moveto($(lvl.tbl), $(ctx(arch)))
        $(lvl.srt) = moveto($(lvl.srt), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.ptr) = $ptr_2
        $(lvl.tbl) = $tbl_2
        $(lvl.srt) = $srt_2
    end)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

Base.summary(lvl::VirtualSparseByteMapLevel) = "SparseByteMap($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseByteMapLevel)
    ext = Extent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseByteMapLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseByteMapLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseByteMapLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualSparseByteMapLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, pos, init)
    Ti = lvl.Ti
    Tp = postype(lvl)
    r = freshen(ctx, lvl.ex, :_r)
    p = freshen(ctx, lvl.ex, :_p)
    q = freshen(ctx, lvl.ex, :_q)
    i = freshen(ctx, lvl.ex, :_i)
    push_preamble!(ctx, quote
        for $r = 1:$(lvl.qos_fill)
            $p = first($(lvl.srt)[$r])
            $(lvl.ptr)[$p] = $(Tp(0))
            $(lvl.ptr)[$p + 1] = $(Tp(0))
            $i = last($(lvl.srt)[$r])
            $q = ($p - $(Tp(1))) * $(ctx(lvl.shape)) + $i
            $(lvl.tbl)[$q] = false
            if $(supports_reassembly(lvl.lvl))
                $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(q, Tp), value(q, Tp)), ctx))
            end
        end
        $(lvl.qos_fill) = 0
        if $(!supports_reassembly(lvl.lvl))
            $(lvl.qos_stop) = $(Tp(0))
        end
        $(lvl.ptr)[1] = 1
    end)
    if !supports_reassembly(lvl.lvl)
        lvl.lvl = declare_level!(ctx, lvl.lvl, call(*, pos, lvl.shape), init)
        push_preamble!(ctx, contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, literal(Tp(1)), call(*, pos, lvl.shape)), ctx))
    end
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, pos)
    Ti = lvl.Ti
    Tp = postype(lvl)
    p = freshen(ctx, lvl.ex, :_p)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, call(*, pos, lvl.shape))
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseByteMapLevel, pos_start, pos_stop)
    Ti = lvl.Ti
    Tp = postype(lvl)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    q_start = freshen(ctx, lvl.ex, :q_start)
    q_stop = freshen(ctx, lvl.ex, :q_stop)
    q = freshen(ctx, lvl.ex, :q)
    old = freshen(ctx, lvl.ex, :old)

    quote
        $q_start = ($(ctx(pos_start)) - $(Tp(1))) * $(ctx(lvl.shape)) + $(Tp(1))
        $q_stop = $(ctx(pos_stop)) * $(ctx(lvl.shape))
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
        $old = length($(lvl.tbl)) + 1
        Finch.resize_if_smaller!($(lvl.tbl), $q_stop)
        Finch.fill_range!($(lvl.tbl), false, $old, $q_stop)
        $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(old, Tp), value(q_stop, Tp)), ctx))
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, pos_stop)
    r = freshen(ctx, lvl.ex, :_r)
    p = freshen(ctx, lvl.ex, :_p)
    p_prev = freshen(ctx, lvl.ex, :_p_prev)
    pos_stop = cache!(ctx, :pos_stop, pos_stop)
    Ti = lvl.Ti
    Tp = postype(lvl)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
        resize!($(lvl.tbl), $(ctx(pos_stop)) * $(ctx(lvl.shape)))
        resize!($(lvl.srt), $(lvl.qos_fill))
        sort!($(lvl.srt))
        $p_prev = $(Tp(0))
        for $r = 1:$(lvl.qos_fill)
            $p = first($(lvl.srt)[$r])
            if $p != $p_prev
                $(lvl.ptr)[$p_prev + 1] = $r
                $(lvl.ptr)[$p] = $r
            end
            $p_prev = $p
        end
        $(lvl.ptr)[$p_prev + 1] = $(lvl.qos_fill) + 1
        $(lvl.qos_stop) = $(lvl.qos_fill)
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, call(*, pos_stop, lvl.shape))
    return lvl
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseByteMapLevel}, ext, mode::Reader, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Ti = lvl.Ti
    Tp = postype(lvl)
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_i_stop = freshen(ctx, tag, :_i_stop)

    Unfurled(
        arr = fbr,
        body = Thunk(
            preamble = quote
                $my_r = $(lvl.ptr)[$(ctx(pos))]
                $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + 1]
                if $my_r != 0 && $my_r < $my_r_stop
                    $my_i = last($(lvl.srt)[$my_r])
                    $my_i_stop = last($(lvl.srt)[$my_r_stop - 1])
                else
                    $my_i = $(Ti(1))
                    $my_i_stop = $(Ti(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    stop = (ctx, ext) -> value(my_i_stop),
                    body = (ctx, ext) -> Stepper(
                        seek = (ctx, ext) -> quote
                            while $my_r + $(Tp(1)) < $my_r_stop && last($(lvl.srt)[$my_r]) < $(ctx(getstart(ext)))
                                $my_r += $(Tp(1))
                            end
                        end,
                        preamble = :($my_i = last($(lvl.srt)[$my_r])),
                        stop = (ctx, ext) -> value(my_i),
                        chunk = Spike(
                            body = FillLeaf(virtual_level_fill_value(lvl)),
                            tail = Thunk(
                                preamble = :($my_q = ($(ctx(pos)) - $(Tp(1))) * $(ctx(lvl.shape)) + $my_i),
                                body = (ctx) -> instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, lvl.Ti)), mode),
                            ),
                        ),
                        next = (ctx, ext) -> :($my_r += $(Tp(1))),
                    )
                ),
                Phase(
                    body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
                )
            ])
        )
    )
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseByteMapLevel}, ext, mode::Reader, ::typeof(gallop))
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Ti = lvl.Ti
    Tp = postype(lvl)
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_i_stop = freshen(ctx, tag, :_i_stop)
    my_j = freshen(ctx, tag, :_j)

    Unfurled(
        arr = fbr,
        body = Thunk(
            preamble = quote
                $my_r = $(lvl.ptr)[$(ctx(pos))]
                $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + 1]
                if $my_r != 0 && $my_r < $my_r_stop
                    $my_i = last($(lvl.srt)[$my_r])
                    $my_i_stop = last($(lvl.srt)[$my_r_stop - 1])
                else
                    $my_i = $(Tp(1))
                    $my_i_stop = $(Tp(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    stop = (ctx, ext) -> value(my_i_stop),
                    body = (ctx, ext) -> Jumper(
                        seek = (ctx, ext) -> quote
                            while $my_r + $(Tp(1)) < $my_r_stop && last($(lvl.srt)[$my_r]) < $(ctx(getstart(ext)))
                                $my_r += $(Tp(1))
                            end
                        end,
                        preamble = :($my_i = last($(lvl.srt)[$my_r])),
                        stop = (ctx, ext) -> value(my_i),
                        chunk =  Spike(
                            body = FillLeaf(virtual_level_fill_value(lvl)),
                            tail = Thunk(
                                preamble = :($my_q = ($(ctx(pos)) - $(Tp(1))) * $(ctx(lvl.shape)) + $my_i),
                                body = (ctx) -> instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, lvl.Ti)), mode),
                            ),
                        ),
                        next = (ctx, ext) -> :($my_r += $(Tp(1)))
                    )
                ),
                Phase(
                    body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
                )
            ])
        )
    )
end


function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseByteMapLevel}, ext, mode::Reader, ::typeof(follow))
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    my_q = freshen(ctx, tag, :_q)
    q = pos
    Ti = lvl.Ti

    Unfurled(
        arr = fbr,
        body = Lookup(
            body = (ctx, i) -> Thunk(
                preamble = quote
                    $my_q = ($(ctx(q)) - $(Ti(1))) * $(ctx(lvl.shape)) + $(ctx(i))
                end,
                body = (ctx) -> Switch([
                    value(:($(lvl.tbl)[$my_q])) => instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q)), mode),
                    literal(true) => FillLeaf(virtual_level_fill_value(lvl))
                ])
            )
        )
    )
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseByteMapLevel}, ext, mode::Updater, proto) =
    unfurl(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto)
function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualSparseByteMapLevel}, ext, mode::Updater, ::Union{typeof(defaultupdate), typeof(extrude), typeof(laminate)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)
    dirty = freshen(ctx, :dirty)

    Unfurled(
        arr = fbr,
        body = Lookup(
            body = (ctx, idx) -> Thunk(
                preamble = quote
                    $my_q = ($(ctx(pos)) - $(Tp(1))) * $(ctx(lvl.shape)) + $(ctx(idx))
                    $dirty = false
                end,
                body = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(my_q, lvl.Ti), dirty), mode),
                epilogue = quote
                    if $dirty
                        $(fbr.dirty) = true
                        if !$(lvl.tbl)[$my_q]
                            $(lvl.tbl)[$my_q] = true
                            $(lvl.qos_fill) += 1
                            if $(lvl.qos_fill) > $(lvl.qos_stop)
                                $(lvl.qos_stop) = max($(lvl.qos_stop) << 1, 1)
                                Finch.resize_if_smaller!($(lvl.srt), $(lvl.qos_stop))
                            end
                            $(lvl.srt)[$(lvl.qos_fill)] = ($(ctx(pos)), $(ctx(idx)))
                        end
                    end
                end
            )
        )
    )
end
