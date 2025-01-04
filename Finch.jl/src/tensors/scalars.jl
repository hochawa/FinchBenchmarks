mutable struct Scalar{Vf, Tv} <: AbstractTensor
    val::Tv
end

Scalar(Vf, args...) = Scalar{Vf}(args...)
Scalar{Vf}(args...) where {Vf} = Scalar{Vf, typeof(Vf)}(args...)
Scalar{Vf, Tv}() where {Vf, Tv} = Scalar{Vf, Tv}(Vf)

@inline Base.ndims(::Type{<:Scalar}) = 0
@inline Base.ndims(::Scalar) = 0
@inline Base.size(::Scalar) = ()
@inline Base.axes(::Scalar) = ()
@inline Base.eltype(::Scalar{Vf, Tv}) where {Vf, Tv} = Tv
@inline Base.eltype(::Type{Scalar{Vf, Tv}}) where {Vf, Tv} = Tv
@inline fill_value(::Type{<:Scalar{Vf}}) where {Vf} = Vf
@inline fill_value(::Scalar{Vf}) where {Vf} = Vf
Base.similar(tns::Scalar{Vf, Tv}) where {Vf, Tv} = Scalar{Vf, Tv}()

(tns::Scalar)() = tns.val
@inline Base.getindex(tns::Scalar) = tns.val

struct VirtualScalar
    ex
    Tv
    Vf
    name
    val
end

lower(ctx::AbstractCompiler, tns::VirtualScalar, ::DefaultStyle) = tns.ex
function virtualize(ctx, ex, ::Type{Scalar{Vf, Tv}}, tag) where {Vf, Tv}
    sym = freshen(ctx, tag)
    val = Symbol(tag, :_val) #TODO hmm this is risky
    push_preamble!(ctx, quote
        $sym = $ex
        $val = $sym.val
    end)
    VirtualScalar(sym, Tv, Vf, tag, val)
end

virtual_moveto(ctx, lvl::VirtualScalar, arch) = lvl

virtual_size(ctx, ::VirtualScalar) = ()

virtual_fill_value(ctx, tns::VirtualScalar) = tns.Vf
virtual_eltype(tns::VirtualScalar, ctx) = tns.Tv

FinchNotation.finch_leaf(x::VirtualScalar) = virtual(x)

function declare!(ctx, tns::VirtualScalar, init)
    push_preamble!(ctx, quote
        $(tns.val) = $(ctx(init))
    end)
    tns
end

function thaw!(ctx, tns::VirtualScalar)
    return tns
end

function freeze!(ctx, tns::VirtualScalar)
    push_preamble!(ctx, quote
        $(tns.ex).val = $(ctx(tns.val))
    end)
    return tns
end

function lower_access(ctx::AbstractCompiler, tns::VirtualScalar, mode)
    return tns.val
end

function lower_assign(ctx, tns::VirtualScalar, mode, op, rhs)
    lhs = value(tns.val, tns.Tv)
    lhs_2 = ctx(simplify(ctx, call(op, lhs, rhs)))
    :($(tns.val) = $lhs_2)
end

function short_circuit_cases(ctx, tns::VirtualScalar, op)
    if isannihilator(ctx, virtual_fill_value(ctx, tns), op)
        [:(tns.val == 0) => Simplify(FillLeaf(Null()))]
    else
        []
    end
end

mutable struct SparseScalar{Vf, Tv} <: AbstractTensor
    val::Tv
    dirty::Bool
end

SparseScalar(Vf, args...) = SparseScalar{Vf}(args...)
SparseScalar{Vf}(args...) where {Vf} = SparseScalar{Vf, typeof(Vf)}(args...)
SparseScalar{Vf, Tv}() where {Vf, Tv} = SparseScalar{Vf, Tv}(Vf, false)
SparseScalar{Vf, Tv}(val) where {Vf, Tv} = SparseScalar{Vf, Tv}(val, true)

@inline Base.ndims(::Type{<:SparseScalar}) = 0
@inline Base.ndims(::SparseScalar) = 0
@inline Base.size(::SparseScalar) = ()
@inline Base.axes(::SparseScalar) = ()
@inline Base.eltype(::SparseScalar{Vf, Tv}) where {Vf, Tv} = Tv
@inline Base.eltype(::Type{SparseScalar{Vf, Tv}}) where {Vf, Tv} = Tv
@inline fill_value(::Type{<:SparseScalar{Vf}}) where {Vf} = Vf
@inline fill_value(::SparseScalar{Vf}) where {Vf} = Vf
Base.similar(tns::SparseScalar{Vf, Tv}) where {Vf, Tv} = SparseScalar{Vf, Tv}()

(tns::SparseScalar)() = tns.val
@inline Base.getindex(tns::SparseScalar) = tns.val

struct VirtualSparseScalar
    ex
    Tv
    Vf
    name
    val
    dirty
end

lower(ctx::AbstractCompiler, tns::VirtualSparseScalar, ::DefaultStyle) = :($SparseScalar{$(tns.Vf), $(tns.Tv)}($(tns.val), $(tns.dirty)))
function virtualize(ctx, ex, ::Type{SparseScalar{Vf, Tv}}, tag) where {Vf, Tv}
    sym = freshen(ctx, tag)
    val = Symbol(tag, :_val) #TODO hmm this is risky
    dirty = Symbol(tag, :_dirty) #TODO hmm this is risky
    push_preamble!(ctx, quote
        $sym = $ex
        $val = $sym.val
        $dirty = $sym.dirty
    end)
    VirtualSparseScalar(sym, Tv, Vf, tag, val, dirty)
end

virtual_size(ctx, ::VirtualSparseScalar) = ()

virtual_fill_value(ctx, tns::VirtualSparseScalar) = tns.Vf
virtual_eltype(tns::VirtualSparseScalar, ctx) = tns.Tv

virtual_moveto(ctx, lvl::VirtualSparseScalar, arch) = lvl

function declare!(ctx, tns::VirtualSparseScalar, init)
    push_preamble!(ctx, quote
        $(tns.val) = $(ctx(init))
        $(tns.dirty) = false
    end)
    tns
end

function thaw!(ctx, tns::VirtualSparseScalar)
    return tns
end

function freeze!(ctx, tns::VirtualSparseScalar)
    push_preamble!(ctx, quote
        $(tns.ex).val = $(ctx(tns.val))
    end)
    return tns
end

function instantiate(ctx, tns::VirtualSparseScalar, mode::Reader)
    Switch(
        tns.dirty => tns,
        true => Simplify(FillLeaf(tns.Vf)),
    )
end

FinchNotation.finch_leaf(x::VirtualSparseScalar) = virtual(x)

function lower_access(ctx::AbstractCompiler, tns::VirtualSparseScalar, mode)
    return tns.val
end

function lower_assign(ctx, tns::VirtualSparseScalar, mode, op, rhs)
    push_preamble!(ctx, quote
        $(tns.dirty) = true
    end)
    lhs = value(tns.val, tns.Tv)
    lhs_2 = ctx(simplify(ctx, call(op, lhs, rhs)))
    :($(tns.val) = $lhs_2)
end


mutable struct ShortCircuitScalar{Vf, Tv} <: AbstractTensor
    val::Tv
end

ShortCircuitScalar(Vf, args...) = ShortCircuitScalar{Vf}(args...)
ShortCircuitScalar{Vf}(args...) where {Vf} = ShortCircuitScalar{Vf, typeof(Vf)}(args...)
ShortCircuitScalar{Vf, Tv}() where {Vf, Tv} = ShortCircuitScalar{Vf, Tv}(Vf)

@inline Base.ndims(::Type{<:ShortCircuitScalar}) = 0
@inline Base.ndims(::ShortCircuitScalar) = 0
@inline Base.size(::ShortCircuitScalar) = ()
@inline Base.axes(::ShortCircuitScalar) = ()
@inline Base.eltype(::ShortCircuitScalar{Vf, Tv}) where {Vf, Tv} = Tv
@inline Base.eltype(::Type{ShortCircuitScalar{Vf, Tv}}) where {Vf, Tv} = Tv
@inline fill_value(::Type{<:ShortCircuitScalar{Vf}}) where {Vf} = Vf
@inline fill_value(::ShortCircuitScalar{Vf}) where {Vf} = Vf
Base.similar(tns::ShortCircuitScalar{Vf, Tv}) where {Vf, Tv} = ShortCircuitScalar{Vf, Tv}()

(tns::ShortCircuitScalar)() = tns.val
@inline Base.getindex(tns::ShortCircuitScalar) = tns.val

struct VirtualShortCircuitScalar
    ex
    Tv
    Vf
    name
    val
end

lower(ctx::AbstractCompiler, tns::VirtualShortCircuitScalar, ::DefaultStyle) = :($ShortCircuitScalar{$(tns.Vf), $(tns.Tv)}($(tns.val)))
function virtualize(ctx, ex, ::Type{ShortCircuitScalar{Vf, Tv}}, tag) where {Vf, Tv}
    sym = freshen(ctx, tag)
    val = Symbol(tag, :_val) #TODO hmm this is risky
    push_preamble!(ctx, quote
        $sym = $ex
        $val = $sym.val
    end)
    VirtualShortCircuitScalar(sym, Tv, Vf, tag, val)
end

virtual_size(ctx, ::VirtualShortCircuitScalar) = ()

virtual_fill_value(ctx, tns::VirtualShortCircuitScalar) = tns.Vf
virtual_eltype(tns::VirtualShortCircuitScalar, ctx) = tns.Tv

FinchNotation.finch_leaf(x::VirtualShortCircuitScalar) = virtual(x)

function declare!(ctx, tns::VirtualShortCircuitScalar, init)
    push_preamble!(ctx, quote
        $(tns.val) = $(ctx(init))
    end)
    tns
end

function thaw!(ctx, tns::VirtualShortCircuitScalar)
    return tns
end

function freeze!(ctx, tns::VirtualShortCircuitScalar)
    push_preamble!(ctx, quote
        $(tns.ex).val = $(ctx(tns.val))
    end)
    return tns
end

function lower_access(ctx::AbstractCompiler, tns::VirtualShortCircuitScalar, mode)
    return tns.val
end

function lower_assign(ctx, tns::VirtualShortCircuitScalar, mode, op, rhs)
    lhs = value(tns.val, tns.Tv)
    lhs_2 = ctx(simplify(ctx, call(op, lhs, rhs)))
    :($(tns.val) = $lhs_2)
end

virtual_moveto(ctx, lvl::VirtualShortCircuitScalar, arch) = lvl

function short_circuit_cases(ctx, tns::VirtualShortCircuitScalar, op)
    [:(Finch.isannihilator($(ctx.algebra), $(ctx(op)), $(tns.val))) => Simplify(FillLeaf(Null()))]
end

mutable struct SparseShortCircuitScalar{Vf, Tv} <: AbstractTensor
    val::Tv
    dirty::Bool
end

SparseShortCircuitScalar(Vf, args...) = SparseShortCircuitScalar{Vf}(args...)
SparseShortCircuitScalar{Vf}(args...) where {Vf} = SparseShortCircuitScalar{Vf, typeof(Vf)}(args...)
SparseShortCircuitScalar{Vf, Tv}() where {Vf, Tv} = SparseShortCircuitScalar{Vf, Tv}(Vf, false)
SparseShortCircuitScalar{Vf, Tv}(val) where {Vf, Tv} = SparseShortCircuitScalar{Vf, Tv}(val, true)

@inline Base.ndims(::Type{<:SparseShortCircuitScalar}) = 0
@inline Base.ndims(::SparseShortCircuitScalar) = 0
@inline Base.size(::SparseShortCircuitScalar) = ()
@inline Base.axes(::SparseShortCircuitScalar) = ()
@inline Base.eltype(::SparseShortCircuitScalar{Vf, Tv}) where {Vf, Tv} = Tv
@inline Base.eltype(::Type{SparseShortCircuitScalar{Vf, Tv}}) where {Vf, Tv} = Tv
@inline fill_value(::Type{<:SparseShortCircuitScalar{Vf}}) where {Vf} = Vf
@inline fill_value(::SparseShortCircuitScalar{Vf}) where {Vf} = Vf
Base.similar(tns::SparseShortCircuitScalar{Vf, Tv}) where {Vf, Tv} = SparseShortCircuitScalar{Vf, Tv}()

(tns::SparseShortCircuitScalar)() = tns.val
@inline Base.getindex(tns::SparseShortCircuitScalar) = tns.val

struct VirtualSparseShortCircuitScalar
    ex
    Tv
    Vf
    name
    val
    dirty
end

lower(ctx::AbstractCompiler, tns::VirtualSparseShortCircuitScalar, ::DefaultStyle) = :($SparseShortCircuitScalar{$(tns.Vf), $(tns.Tv)}($(tns.val), $(tns.dirty)))
function virtualize(ctx, ex, ::Type{SparseShortCircuitScalar{Vf, Tv}}, tag) where {Vf, Tv}
    sym = freshen(ctx, tag)
    val = Symbol(tag, :_val) #TODO hmm this is risky
    dirty = Symbol(tag, :_dirty) #TODO hmm this is risky
    push_preamble!(ctx, quote
        $sym = $ex
        $val = $sym.val
        $dirty = $sym.dirty
    end)
    VirtualSparseShortCircuitScalar(sym, Tv, Vf, tag, val, dirty)
end

virtual_size(ctx, ::VirtualSparseShortCircuitScalar) = ()

virtual_fill_value(ctx, tns::VirtualSparseShortCircuitScalar) = tns.Vf
virtual_eltype(tns::VirtualSparseShortCircuitScalar, ctx) = tns.Tv

virtual_moveto(ctx, lvl::VirtualSparseShortCircuitScalar, arch) = lvl

function declare!(ctx, tns::VirtualSparseShortCircuitScalar, init)
    push_preamble!(ctx, quote
        $(tns.val) = $(ctx(init))
        $(tns.dirty) = false
    end)
    tns
end

function thaw!(ctx, tns::VirtualSparseShortCircuitScalar)
    return tns
end

function freeze!(ctx, tns::VirtualSparseShortCircuitScalar)
    push_preamble!(ctx, quote
        $(tns.ex).val = $(ctx(tns.val))
    end)
    return tns
end

function instantiate(ctx, tns::VirtualSparseShortCircuitScalar, mode::Reader)
    Switch([
        value(tns.dirty, Bool) => tns,
        true => Simplify(FillLeaf(tns.Vf)),
    ])
end

FinchNotation.finch_leaf(x::VirtualSparseShortCircuitScalar) = virtual(x)

function lower_access(ctx::AbstractCompiler, tns::VirtualSparseShortCircuitScalar, mode)
    return tns.val
end

function lower_assign(ctx, tns::VirtualSparseShortCircuitScalar, mode, op, rhs)
    push_preamble!(ctx, quote
        $(tns.dirty) = true
    end)
    lhs = value(tns.val, tns.Tv)
    lhs_2 = ctx(simplify(ctx, call(op, lhs, rhs)))
    :($(tns.val) = $lhs_2)
end

function short_circuit_cases(ctx, tns::VirtualSparseShortCircuitScalar, op)
    [:(Finch.isannihilator($(ctx.algebra), $(ctx(op)), $(tns.val))) => Simplify(FillLeaf(Null()))]
end