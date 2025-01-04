using Base: @propagate_inbounds

struct PlusOneVector{T, A <: AbstractVector{T}} <: AbstractVector{T}
    data::A
end

@propagate_inbounds function Base.getindex(vec::PlusOneVector{T},
                                           index::Int) where {T}
    return vec.data[index] + 0x01
end

@propagate_inbounds function Base.getindex(vec::PlusOneVector{T},
                                           index::Vararg{Int}) where {T}
    return vec.data[index...] + 0x01
end

@propagate_inbounds function Base.setindex!(vec::PlusOneVector{T},
                                            val::T,
                                            index::Int) where {T}
    vec.data[index] = val - 0x01
end

@propagate_inbounds function Base.setindex!(vec::PlusOneVector{T},
                                            val::T,
                                            index::Vararg{Int}) where {T}
    vec.data[index...] = val - 0x01
end

Base.parent(vec::PlusOneVector{T}) where {T} = vec.data
Base.size(vec::PlusOneVector{T}) where {T} = size(vec.data)
Base.axes(vec::PlusOneVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::PlusOneVector{T}, dim) where {T} = resize!(vec.data, dim)

function moveto(vec::PlusOneVector{T}, device) where {T}
    data = moveto(vec.data, device)
    return PlusOneVector{T}(data)
end

struct MinusEpsVector{T, S, A <: AbstractVector{S}} <: AbstractVector{T}
    data::A
end

MinusEpsVector(data::AbstractVector{T}) where {T} = MinusEpsVector{Limit{T}, T, typeof(data)}(data)

@propagate_inbounds function Base.getindex(vec::MinusEpsVector{T},
                                           index::Int) where {T}
    return minus_eps(vec.data[index])
end

@propagate_inbounds function Base.getindex(vec::MinusEpsVector{T},
                                           index::Vararg{Int}) where {T}
    return minus_eps(vec.data[index...])
end

@propagate_inbounds function Base.setindex!(vec::MinusEpsVector{Limit{T}},
                                            val::Limit{T},
                                            index::Int) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_negative()
    end
    vec.data[index] = val.val
end

@propagate_inbounds function Base.setindex!(vec::MinusEpsVector{Limit{T}},
                                            val::Limit{T},
                                            index::Vararg{Int}) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_negative()
    end
    vec.data[index...] = val.val
end

Base.parent(vec::MinusEpsVector{T}) where {T} = vec.data
Base.size(vec::MinusEpsVector{T}) where {T} = size(vec.data)
Base.axes(vec::MinusEpsVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::MinusEpsVector{T}, dim) where {T} = resize!(vec.data, dim)

function moveto(vec::MinusEpsVector{T}, device) where {T}
    data = moveto(vec.data, device)
    return MinusEpsVector{T}(data)
end

struct PlusEpsVector{T, S, A <:AbstractVector{S}} <: AbstractVector{T}
    data::A
end

PlusEpsVector(data::AbstractVector{T}) where {T} = PlusEpsVector{Limit{T}, T, typeof(data)}(data)

@propagate_inbounds function Base.getindex(vec::PlusEpsVector{T},
                                           index::Int) where {T}
    return plus_eps(vec.data[index])
end

@propagate_inbounds function Base.getindex(vec::PlusEpsVector{T},
                                           index::Vararg{Int}) where {T}
    return plus_eps(vec.data[index...])
end

@propagate_inbounds function Base.setindex!(vec::PlusEpsVector{Limit{T}},
                                            val::Limit{T},
                                            index::Int) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_positive()
    end
    vec.data[index] = val.val
end

@propagate_inbounds function Base.setindex!(vec::PlusEpsVector{Limit{T}},
                                            val::Limit{T},
                                            index::Vararg{Int}) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_positive()
    end
    vec.data[index...] = val.val
end

Base.parent(vec::PlusEpsVector{T}) where {T} = vec.data
Base.size(vec::PlusEpsVector{T}) where {T} = size(vec.data)
Base.axes(vec::PlusEpsVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::PlusEpsVector{T}, dim) where {T} = resize!(vec.data, dim)

function moveto(vec::PlusEpsVector{T}, device) where {T}
    data = moveto(vec.data, device)
    return PlusEpsVector{T}(data)
end