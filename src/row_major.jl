row_major_two_universal!{T <: Integer}(at::AbstractMatrix{T}, v::AbstractVector{T},
                                       out::AbstractVector{T}) = begin
    nbits = 8sizeof(eltype(out))
    multiples = div(length(v), size(at, 1))
    @argcheck length(at) ≠ 0

    fill!(out, zero(T))
    length(v) ≠ 0 || return out

    @argcheck length(v) % size(at, 1) == 0
    @argcheck round(Int64, (multiples * size(at, 2)) / nbits, RoundUp) == length(out)

    for mult in 0:(multiples - 1), j in 1:size(at, 2)
        i′₀ = mult * size(at, 1) + 1
        x = at[1, j] & v[i′₀]
        for i in 2:size(at, 1)
            x $= at[i, j] & v[i′₀ + i - 1]
        end
        ibit = mult * size(at, 2) + j
        o_index = div(ibit - 1, nbits) + 1
        out[o_index] |= convert(T, xorbits(x)) << (nbits - 1 - (ibit - 1) % nbits)
    end
    out
end

immutable RowMajorTwoUniversal{T <: Integer} <: AbstractTwoUniversal
    matrix::Matrix{T}
end

Base.size(func::RowMajorTwoUniversal) = reverse(size(func.matrix))
Base.size(func::RowMajorTwoUniversal, i::Integer) = size(func.matrix, 3 - i)

two_universal!(func::RowMajorTwoUniversal, v::AbstractVector, out::AbstractVector) = begin
    row_major_two_universal!(func.matrix, v, out)
end
