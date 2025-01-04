"""
    ElementLevel{Vf, [Tv=typeof(Vf)], [Tp=Int], [Val]}()

A subfiber of an element level is a scalar of type `Tv`, initialized to `Vf`. `Vf`
may optionally be given as the first argument.

The data is stored in a vector
of type `Val` with `eltype(Val) = Tv`. The type `Tp` is the index type used to
access Val.

```jldoctest
julia> tensor_tree(Tensor(Dense(Element(0.0)), [1, 2, 3]))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: 1.0
   ├─ [2]: 2.0
   └─ [3]: 3.0
```
"""
struct ElementLevel{Vf, Tv, Tp, Val} <: AbstractLevel
    val::Val
end
const Element = ElementLevel

function ElementLevel(d, args...)
    isbits(d) || throw(ArgumentError("Finch currently only supports isbits defaults"))
    ElementLevel{d}(args...)
end
ElementLevel{Vf}() where {Vf} = ElementLevel{Vf, typeof(Vf)}()
ElementLevel{Vf}(val::Val) where {Vf, Val} = ElementLevel{Vf, eltype(Val)}(val)
ElementLevel{Vf, Tv}(args...) where {Vf, Tv} = ElementLevel{Vf, Tv, Int}(args...)
ElementLevel{Vf, Tv, Tp}() where {Vf, Tv, Tp} = ElementLevel{Vf, Tv, Tp}(Tv[])

ElementLevel{Vf, Tv, Tp}(val::Val) where {Vf, Tv, Tp, Val} = ElementLevel{Vf, Tv, Tp, Val}(val)

Base.summary(::Element{Vf}) where {Vf} = "Element($(Vf))"

similar_level(::ElementLevel{Vf, Tv, Tp}, fill_value, eltype::Type, ::Vararg) where {Vf, Tv, Tp} =
    ElementLevel{fill_value, eltype, Tp}()

postype(::Type{<:ElementLevel{Vf, Tv, Tp}}) where {Vf, Tv, Tp} = Tp

function moveto(lvl::ElementLevel{Vf, Tv, Tp}, device) where {Vf, Tv, Tp}
    return ElementLevel{Vf, Tv, Tp}(moveto(lvl.val, device))
end

pattern!(lvl::ElementLevel{Vf, Tv, Tp}) where  {Vf, Tv, Tp} =
    Pattern{Tp}()
set_fill_value!(lvl::ElementLevel{Vf, Tv, Tp}, init) where {Vf, Tv, Tp} =
    ElementLevel{init, Tv, Tp}(lvl.val)
Base.resize!(lvl::ElementLevel) = lvl

function Base.show(io::IO, lvl::ElementLevel{Vf, Tv, Tp, Val}) where {Vf, Tv, Tp, Val}
    print(io, "Element{")
    show(io, Vf)
    print(io, ", $Tv, $Tp}(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.val)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:ElementLevel}) =
    print(io, fbr.lvl.val[fbr.pos])

@inline level_ndims(::Type{<:ElementLevel}) = 0
@inline level_size(::ElementLevel) = ()
@inline level_axes(::ElementLevel) = ()
@inline level_eltype(::Type{<:ElementLevel{Vf, Tv}}) where {Vf, Tv} = Tv
@inline level_fill_value(::Type{<:ElementLevel{Vf}}) where {Vf} = Vf
data_rep_level(::Type{<:ElementLevel{Vf, Tv}}) where {Vf, Tv} = ElementData(Vf, Tv)

(fbr::Tensor{<:ElementLevel})() = SubFiber(fbr.lvl, 1)()
function (fbr::SubFiber{<:ElementLevel})()
    q = fbr.pos
    return fbr.lvl.val[q]
end

countstored_level(lvl::ElementLevel, pos) = pos

mutable struct VirtualElementLevel <: AbstractVirtualLevel
    ex
    Vf
    Tv
    Tp
    val
end

is_level_injective(ctx, ::VirtualElementLevel) = []
is_level_atomic(ctx, lvl::VirtualElementLevel) = ([], false)
function is_level_concurrent(ctx, lvl::VirtualElementLevel)
    return ([], true)
end

lower(ctx::AbstractCompiler, lvl::VirtualElementLevel, ::DefaultStyle) = lvl.ex

function virtualize(ctx, ex, ::Type{ElementLevel{Vf, Tv, Tp, Val}}, tag=:lvl) where {Vf, Tv, Tp, Val}
    sym = freshen(ctx, tag)
    val = freshen(ctx, tag, :_val)
    push_preamble!(ctx, quote
        $sym = $ex
        $val = $ex.val
    end)
    VirtualElementLevel(sym, Vf, Tv, Tp, val)
end

Base.summary(lvl::VirtualElementLevel) = "Element($(lvl.Vf))"

virtual_level_resize!(ctx, lvl::VirtualElementLevel) = lvl
virtual_level_size(ctx, ::VirtualElementLevel) = ()
virtual_level_ndims(ctx, lvl::VirtualElementLevel) = 0
virtual_level_eltype(lvl::VirtualElementLevel) = lvl.Tv
virtual_level_fill_value(lvl::VirtualElementLevel) = lvl.Vf

postype(lvl::VirtualElementLevel) = lvl.Tp

function declare_level!(ctx, lvl::VirtualElementLevel, pos, init)
    init == literal(lvl.Vf) || throw(FinchProtocolError("Cannot initialize Element Levels to non-fill values (have $init expected $(lvl.Vf))"))
    lvl
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualElementLevel, pos)
    push_preamble!(ctx, quote
        resize!($(lvl.val), $(ctx(pos)))
    end)
    return lvl
end

thaw_level!(ctx::AbstractCompiler, lvl::VirtualElementLevel, pos) = lvl

function assemble_level!(ctx, lvl::VirtualElementLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    quote
        Finch.resize_if_smaller!($(lvl.val), $(ctx(pos_stop)))
        Finch.fill_range!($(lvl.val), $(lvl.Vf), $(ctx(pos_start)), $(ctx(pos_stop)))
    end
end

supports_reassembly(::VirtualElementLevel) = true
function reassemble_level!(ctx, lvl::VirtualElementLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    push_preamble!(ctx, quote
        Finch.fill_range!($(lvl.val), $(lvl.Vf), $(ctx(pos_start)), $(ctx(pos_stop)))
    end)
    lvl
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualElementLevel, arch)
    val_2 = freshen(ctx, :val)
    push_preamble!(ctx, quote
        $val_2 = $(lvl.val)
        $(lvl.val) = $moveto($(lvl.val), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.val) = $val_2
    end)
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualElementLevel}, mode::Reader)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    val = freshen(ctx, lvl.ex, :_val)
    return Thunk(
        preamble = quote
            $val = $(lvl.val)[$(ctx(pos))]
        end,
        body = (ctx) -> VirtualScalar(nothing, lvl.Tv, lvl.Vf, gensym(), val)
    )
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualElementLevel}, mode::Updater)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    VirtualScalar(nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))]))
end

function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualElementLevel}, mode::Updater)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    VirtualSparseScalar(nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))]), fbr.dirty)
end

function lower_assign(ctx, fbr::VirtualHollowSubFiber{VirtualElementLevel}, mode::Updater, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    lower_assign(ctx, VirtualSparseScalar(nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))]), fbr.dirty), mode, op, rhs)
end

function lower_assign(ctx, fbr::VirtualSubFiber{VirtualElementLevel}, mode::Updater, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    lower_assign(ctx, VirtualScalar(nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))])), mode, op, rhs)
end