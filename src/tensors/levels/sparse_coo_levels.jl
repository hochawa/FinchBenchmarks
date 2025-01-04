"""
    SparseCOOLevel{[N], [TI=Tuple{Int...}], [Ptr, Tbl]}(lvl, [dims])

A subfiber of a sparse level does not need to represent slices which are
entirely [`fill_value`](@ref). Instead, only potentially non-fill slices are
stored as subfibers in `lvl`. The sparse coo level corresponds to `N` indices in
the subfiber, so fibers in the sublevel are the slices `A[:, ..., :, i_1, ...,
i_n]`.  A set of `N` lists (one for each index) are used to record which slices
are stored. The coordinates (sets of `N` indices) are sorted in column major
order.  Optionally, `dims` are the sizes of the last dimensions.

`TI` is the type of the last `N` tensor indices, and `Tp` is the type used for
positions in the level.

The type `Tbl` is an NTuple type where each entry k is a subtype `AbstractVector{TI[k]}`.

The type `Ptr` is the type for the pointer array.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseCOO{1}(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseCOO{1} (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseCOO{1} (0.0) [1:3]
   └─ [:, 3]: SparseCOO{1} (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

julia> tensor_tree(Tensor(SparseCOO{2}(Element(0.0)), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseCOO{2} (0.0) [:,1:3]
   ├─ [1, 1]: 10.0
   ├─ [2, 1]: 30.0
   ├─ [1, 3]: 20.0
   └─ [3, 3]: 40.0
```
"""
struct SparseCOOLevel{N, TI<:Tuple, Ptr, Tbl, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::TI
    ptr::Ptr
    tbl::Tbl
end
const SparseCOO = SparseCOOLevel

SparseCOOLevel(lvl) = throw(ArgumentError("You must specify the number of dimensions in a SparseCOOLevel, e.g. Tensor(SparseCOO{2}(Element(0.0)))"))
SparseCOOLevel(lvl, shape::NTuple{N, Any}, args...) where {N} = SparseCOOLevel{N}(lvl, shape, args...)

SparseCOOLevel{N}(lvl) where {N} = SparseCOOLevel{N, NTuple{N, Int}}(lvl)
SparseCOOLevel{N}(lvl, shape::TI, args...) where {N, TI} = SparseCOOLevel{N, TI}(lvl, shape, args...)
SparseCOOLevel{N, TI}(lvl) where {N, TI} = SparseCOOLevel{N, TI}(lvl, ((zero(Ti) for Ti in TI.parameters)...,))
SparseCOOLevel{N, TI}(lvl, shape) where {N, TI} =
    SparseCOOLevel{N, TI}(lvl, TI(shape), postype(lvl)[1], ((Ti[] for Ti in TI.parameters)...,))

SparseCOOLevel{N, TI}(lvl::Lvl, shape, ptr::Ptr, tbl::Tbl) where {N, TI, Lvl, Ptr, Tbl} =
    SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}(lvl, TI(shape), ptr, tbl)

Base.summary(lvl::SparseCOOLevel{N}) where {N} = "SparseCOO{$N}($(summary(lvl.lvl)))"
similar_level(lvl::SparseCOOLevel{N}, fill_value, eltype::Type, tail...) where {N} =
    SparseCOOLevel{N}(similar_level(lvl.lvl, fill_value, eltype, tail[1:end-N]...), (tail[end-N+1:end]...,))

function postype(::Type{SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}}) where {N, TI, Ptr, Tbl, Lvl}
    return postype(Lvl)
end

function moveto(lvl::SparseCOOLevel{N, TI}, device) where {N, TI}
    lvl_2 = moveto(lvl.lvl, device)
    ptr_2 = moveto(lvl.ptr, device)
    tbl_2 = ntuple(n->moveto(lvl.tbl[n], device), N)
    return SparseCOOLevel{N, TI}(lvl_2, lvl.shape, ptr_2, tbl_2)
end

pattern!(lvl::SparseCOOLevel{N, TI}) where {N, TI} =
    SparseCOOLevel{N, TI}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.tbl)

function countstored_level(lvl::SparseCOOLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

set_fill_value!(lvl::SparseCOOLevel{N, TI}, init) where {N, TI} =
    SparseCOOLevel{N, TI}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.tbl)

Base.resize!(lvl::SparseCOOLevel{N, TI}, dims...) where {N, TI} =
    SparseCOOLevel{N, TI}(resize!(lvl.lvl, dims[1:end-N]...), (dims[end-N + 1:end]...,), lvl.ptr, lvl.tbl)

function Base.show(io::IO, lvl::SparseCOOLevel{N, TI}) where {N, TI}
    if get(io, :compact, false)
        print(io, "SparseCOO{$N}(")
    else
        print(io, "SparseCOO{$N, $TI}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo=>TI), lvl.shape)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.ptr)
        print(io, ", (")
        for (n, Ti) = enumerate(TI.parameters)
            show(io, lvl.tbl[n])
            print(io, ", ")
        end
        print(io, ") ")
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:SparseCOOLevel{N}}) where {N} =
    print(io, "SparseCOO{", N, "} (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:SparseCOOLevel{N}}) where {N}
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:lvl.ptr[pos + 1] - 1) do qos
        LabelledTree(cartesian_label([range_label() for _ = 1:ndims(fbr) - N]..., map(n -> lvl.tbl[n][qos], 1:N)...), SubFiber(lvl.lvl, qos))
    end
end

@inline level_ndims(::Type{<:SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}}) where {N, TI, Ptr, Tbl, Lvl} = N + level_ndims(Lvl)
@inline level_size(lvl::SparseCOOLevel) = (level_size(lvl.lvl)..., lvl.shape...)
@inline level_axes(lvl::SparseCOOLevel) = (level_axes(lvl.lvl)..., map(Base.OneTo, lvl.shape)...)
@inline level_eltype(::Type{<:SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}}) where {N, TI, Ptr, Tbl, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}}) where {N, TI, Ptr, Tbl, Lvl} = level_fill_value(Lvl)
data_rep_level(::Type{<:SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}}) where {N, TI, Ptr, Tbl, Lvl} = (SparseData^N)(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseCOOLevel})() = fbr
(fbr::SubFiber{<:SparseCOOLevel})() = fbr
function (fbr::SubFiber{<:SparseCOOLevel{N, TI}})(idxs...) where {N, TI}
    isempty(idxs) && return fbr
    idx = idxs[end-N + 1:end]
    lvl = fbr.lvl
    target = lvl.ptr[fbr.pos]:lvl.ptr[fbr.pos + 1] - 1
    for n = N:-1:1
        target = searchsorted(view(lvl.tbl[n], target), idx[n]) .+ (first(target) - 1)
    end
    isempty(target) ? fill_value(fbr) : SubFiber(lvl.lvl, first(target))(idxs[1:end-N]...)
end

mutable struct VirtualSparseCOOLevel <: AbstractVirtualLevel
    lvl
    ex
    N
    TI
    ptr
    tbl
    Lvl
    shape
    qos_fill
    qos_stop
    prev_pos
end

is_level_injective(ctx, lvl::VirtualSparseCOOLevel) = [is_level_injective(ctx, lvl.lvl)..., (true for _ in 1:lvl.N)...]
function is_level_atomic(ctx, lvl::VirtualSparseCOOLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic for _ in 1:lvl.N]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseCOOLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false for _ in 1:lvl.N]], false)
end

function virtualize(ctx, ex, ::Type{SparseCOOLevel{N, TI, Ptr, Tbl, Lvl}}, tag=:lvl) where {N, TI, Ptr, Tbl, Lvl}
    sym = freshen(ctx, tag)
    shape = map(n->value(:($sym.shape[$n]), Int), 1:N)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    ptr = freshen(ctx, tag, :_ptr)
    tbl = map(n->freshen(ctx, tag, :_tbl, n), 1:N)
    push_preamble!(ctx, quote
        $sym = $ex
        $ptr = $ex.ptr
    end)
    for n = 1:N
        push_preamble!(ctx, quote
            $(tbl[n]) = $ex.tbl[$n]
        end)
    end
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    prev_coord = map(n->freshen(ctx, sym, :_prev_coord_, n), 1:N)
    VirtualSparseCOOLevel(lvl_2, sym, N, TI, ptr, tbl, Lvl, shape, qos_fill, qos_stop, prev_pos)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseCOOLevel, ::DefaultStyle)
    quote
        $SparseCOOLevel{$(lvl.N), $(lvl.TI)}(
            $(ctx(lvl.lvl)),
            ($(map(ctx, lvl.shape)...),),
            $(lvl.ptr),
            ($(lvl.tbl...),)
        )
    end
end

Base.summary(lvl::VirtualSparseCOOLevel) = "SparseCOO{$(lvl.N)}($(summary(lvl.lvl)))"

function virtual_level_size(ctx::AbstractCompiler, lvl::VirtualSparseCOOLevel)
    ext = map((ti, stop)->Extent(literal(ti(1)), stop), lvl.TI.parameters, lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext...)
end

function virtual_level_resize!(ctx::AbstractCompiler, lvl::VirtualSparseCOOLevel, dims...)
    lvl.shape = map(getstop, dims[end - lvl.N + 1:end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end - lvl.N]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseCOOLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseCOOLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualSparseCOOLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseCOOLevel, pos, init)
    TI = lvl.TI
    Tp = postype(lvl)

    push_preamble!(ctx, quote
        $(lvl.qos_fill) = $(Tp(0))
        $(lvl.qos_stop) = $(Tp(0))
    end)
    if issafe(get_mode_flag(ctx))
        push_preamble!(ctx, quote
            $(lvl.prev_pos) = $(Tp(0))
        end)
    end
    lvl.lvl = declare_level!(ctx, lvl.lvl, literal(Tp(0)), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseCOOLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseCOOLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $pos_stop + 1)
        for $p = 2:($pos_stop + 1)
            $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
        end
        $qos_stop = $(lvl.ptr)[$pos_stop + 1] - 1
        $(Expr(:block, map(1:lvl.N) do n
            :(resize!($(lvl.tbl[n]), $qos_stop))
        end...))
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualSparseCOOLevel, arch)
    ptr_2 = freshen(ctx, lvl.ptr)
    push_preamble!(ctx, quote
        $ptr_2 = $(lvl.ptr)
        $(lvl.ptr) = $moveto($(lvl.ptr), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.ptr) = $ptr_2
    end)
    tbl_2 = map(lvl.tbl) do idx
        idx_2 = freshen(ctx, idx)
        push_preamble!(ctx, quote
            $idx_2 = $idx
            $idx = $moveto($idx, $(ctx(arch)))
        end)
        push_epilogue!(ctx, quote
            $idx = $idx_2
        end)
        idx_2
    end
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

struct SparseCOOWalkTraversal
    lvl
    R
    start
    stop
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseCOOLevel}, ext, mode::Reader, proto)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    Tp = postype(lvl)
    start = value(:($(lvl.ptr)[$(ctx(pos))]), Tp)
    stop = value(:($(lvl.ptr)[$(ctx(pos)) + 1]), Tp)

    Unfurled(
        arr = fbr,
        body = unfurl(ctx, SparseCOOWalkTraversal(lvl, lvl.N, start, stop), ext, mode, proto)
    )
end

function unfurl(ctx, trv::SparseCOOWalkTraversal, ext, mode::Reader, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, R, start, stop) = (trv.lvl, trv.R, trv.start, trv.stop)
    tag = lvl.ex
    TI = lvl.TI
    Tp = postype(lvl)
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_step = freshen(ctx, tag, :_q_step)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i_stop = freshen(ctx, tag, :_i_stop)

    Thunk(
        preamble = quote
            $my_q = $(ctx(start))
            $my_q_stop = $(ctx(stop))
            if $my_q < $my_q_stop
                $my_i = $(lvl.tbl[R])[$my_q]
                $my_i_stop = $(lvl.tbl[R])[$my_q_stop - 1]
            else
                $my_i = $(TI.parameters[R](1))
                $my_i_stop = $(TI.parameters[R](0))
            end
        end,
        body = (ctx) -> Sequence([
            Phase(
                stop = (ctx, ext) -> value(my_i_stop),
                body = (ctx, ext) ->
                    if R == 1
                        Stepper(
                            seek = (ctx, ext) -> quote
                                if $(lvl.tbl[R])[$my_q] < $(ctx(getstart(ext)))
                                    $my_q = Finch.scansearch($(lvl.tbl[R]), $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                                end
                            end,
                            preamble = :($my_i = $(lvl.tbl[R])[$my_q]),
                            stop =  (ctx, ext) -> value(my_i),
                            chunk = Spike(
                                body = FillLeaf(virtual_level_fill_value(lvl)),
                                tail = instantiate(ctx, VirtualSubFiber(lvl.lvl, my_q), mode),
                            ),
                            next = (ctx, ext) -> :($my_q += $(Tp(1)))
                        )
                    else
                        Stepper(
                            seek = (ctx, ext) -> quote
                                if $(lvl.tbl[R])[$my_q] < $(ctx(getstart(ext)))
                                    $my_q = Finch.scansearch($(lvl.tbl[R]), $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                                end
                            end,
                            preamble = quote
                                $my_i = $(lvl.tbl[R])[$my_q]
                                $my_q_step = $my_q
                                if $(lvl.tbl[R])[$my_q_step] == $my_i
                                    $my_q_step = Finch.scansearch($(lvl.tbl[R]), $my_i + 1, $my_q_step, $my_q_stop - 1)
                                end
                            end,
                            stop = (ctx, ext) -> value(my_i),
                            chunk = Spike(
                                body = FillLeaf(virtual_level_fill_value(lvl)),
                                tail = instantiate(ctx, SparseCOOWalkTraversal(lvl, R - 1, value(my_q, Tp), value(my_q_step, Tp)), mode),
                            ),
                            next = (ctx, ext) -> :($my_q = $my_q_step)
                        )
                    end
            ),
            Phase(
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            )
        ])
    )
end

struct SparseCOOExtrudeTraversal
    lvl
    qos
    fbr_dirty
    coords
    prev_coord
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseCOOLevel}, ext, mode::Updater, proto) =
    unfurl(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto)

function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualSparseCOOLevel}, ext, mode::Updater, proto)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    TI = lvl.TI
    Tp = postype(lvl)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop

    qos = freshen(ctx, tag, :_q)
    prev_coord = freshen(ctx, tag, :_prev_coord)
    Thunk(
        preamble = quote
            $qos = $qos_fill + 1
            $(if issafe(get_mode_flag(ctx))
                quote
                    $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("SparseCOOLevels cannot be updated multiple times"))
                    $prev_coord = ()
                end
            end)
        end,
        body = (ctx) -> unfurl(ctx, SparseCOOExtrudeTraversal(lvl, qos, fbr.dirty, [], prev_coord), ext, mode, proto),
        epilogue = quote
            $(lvl.ptr)[$(ctx(pos)) + 1] = $qos - $qos_fill - 1
            $(if issafe(get_mode_flag(ctx))
                quote
                    if $qos - $qos_fill - 1 > 0
                        $(lvl.prev_pos) = $(ctx(pos))
                    end
                end
            end)
            $qos_fill = $qos - 1
        end
    )
end

function unfurl(ctx, trv::SparseCOOExtrudeTraversal, ext, mode::Updater, proto)#::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, qos, fbr_dirty, coords) = (trv.lvl, trv.qos, trv.fbr_dirty, trv.coords)
    TI = lvl.TI
    Tp = postype(lvl)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    if length(coords) + 1 < lvl.N
        Lookup(
            body = (ctx, i) -> instantiate(ctx, SparseCOOExtrudeTraversal(lvl, qos, fbr_dirty, (i, coords...), trv.prev_coord), mode),
        )
    else
        dirty = freshen(ctx, :dirty)
        Lookup(
            body = (ctx, idx) -> Thunk(
                preamble = quote
                    if $qos > $qos_stop
                        $qos_stop = max($qos_stop << 1, 1)
                        $(Expr(:block, map(1:lvl.N) do n
                            :(Finch.resize_if_smaller!($(lvl.tbl[n]), $qos_stop))
                        end...))
                        $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(qos, Tp), value(qos_stop, Tp)), ctx))
                    end
                    $dirty = false
                end,
                body = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty), mode),
                epilogue = begin
                    coords_2 = map(ctx, (idx, coords...))
                    quote
                        if $dirty
                            $(if issafe(get_mode_flag(ctx))
                                quote
                                    $(trv.prev_coord) < ($(reverse(coords_2)...),) || begin
                                        throw(FinchProtocolError("SparseCOOLevels cannot be updated multiple times"))
                                    end
                                    $(trv.prev_coord) = ($(reverse(coords_2)...),)
                                end
                            end)
                            $(fbr_dirty) = true
                            $(Expr(:block, map(enumerate(coords_2)) do (n, i)
                                :($(lvl.tbl[n])[$qos] = $i)
                            end...))
                            $qos += $(Tp(1))
                        end
                    end
                end
            )
        )
    end
end
