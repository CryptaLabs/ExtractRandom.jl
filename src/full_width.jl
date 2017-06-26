full_width_two_universal!{T <: Integer}(at::AbstractMatrix{T},
                                        shifts::AbstractMatrix{T},
                                        v::AbstractVector{T},
                                        out::AbstractVector{T}) = begin
    nbits = 8sizeof(eltype(out))
    @argcheck length(at) ≠ 0

    fill!(out, zero(T))
    length(v) ≠ 0 || return out

    @argcheck length(v) % size(at, 1) == 0
    @argcheck multiples * size(at, 2) == length(out)

    for mult in 0:(multiples - 1), j in 1:size(at, 2)
        i′₀ = mult * size(at, 1) + 1
        x = _ror(at[1, j] & v[i′₀], shifts[1, j])
        for i in 2:size(at, 1)
            x $= _ror(at[i, j] & v[i′₀ + i - 1], shifts[i, j])
        end
        out[mult * size(at, 2) + j] = x
    end
    out
end

immutable FullWidthTwoUniversal{T <: Integer} <: AbstractTwoUniversal
    matrix::Matrix{T}
    shifts::Matrix{T}
end

@inline _ror(x::Integer, shift::Integer) = (x >>> shift) | (x << (8sizeof(x) - shift))
Base.size(func::FullWidthTwoUniversal) = reverse(size(func.matrix))
Base.size(func::FullWidthTwoUniversal, i::Integer) = size(func.matrix, 3 - i)

two_universal!(func::FullWidthTwoUniversal, v::AbstractVector, out::AbstractVector) = begin
    full_width_two_universal!(func.matrix, func.shifts, v, out)
end

(func::FullWidthTwoUniversal)(v::AbstractVector) = begin
    @argcheck length(v) % size(func, 2) == 0
    n = div(length(v), size(func, 2)) * size(func, 1)
    two_universal!(func, v, similar(v, (n,)))
end
(func::FullWidthTwoUniversal)(v::AbstractVector, out::AbstractVector) = begin
    two_universal!(func, v, out)
end
