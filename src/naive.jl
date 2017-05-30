naive_two_universal!{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T},
                                   out::AbstractVector{T}) = begin
    Nbits = 8sizeof(eltype(out))
    multiples = div(length(v), size(a, 2))
    @argcheck length(a) ≠ 0
    if length(v) == 0
        fill!(out, zero(T))
        return out
    end
    @argcheck length(v) % size(a, 2) == 0
    @argcheck round(Int64, (multiples * size(a, 1)) / Nbits, RoundUp) == length(out)

    n, m = size(a)
    nbitsy(i) = min(Nbits, size(a, 1) * multiples  - i * Nbits) - 1
    for i in 0:(length(out) - 1)
        current = zero(T)
        for i′ in 0:nbitsy(i)
            vindex = div(i * Nbits + i′, n) * m + 1
            intermediate = v[vindex] & a[(i * Nbits + i′) % n + 1, 1]
            current |= convert(T, xorbits(intermediate)) << (Nbits - 1 - i′)
        end
        out[i + 1] = current
    end

    for j in 2:m
        for i in 0:(length(out) - 1)
            current = out[i + 1]
            for i′ in 0:nbitsy(i)
                vindex = j + div(i * Nbits + i′, n) * m
                vⱼ = v[vindex]
                intermediate = vⱼ & a[(i * Nbits + i′) % n + 1, (j - 1) % m + 1]
                current $= convert(T, xorbits(intermediate)) << (Nbits - 1 - i′)
            end
            out[i + 1] = current
        end
    end
    out
end

immutable NaiveTwoUniversal{T <: Integer} <: AbstractTwoUniversal
    matrix::Matrix{T}
end

Base.size(func::NaiveTwoUniversal) = size(func.matrix)
Base.size(func::NaiveTwoUniversal, i::Integer) = size(func.matrix, i)

two_universal!(func::NaiveTwoUniversal, v::AbstractVector, out::AbstractVector) = begin
    naive_two_universal!(func.matrix, v, out)
end
