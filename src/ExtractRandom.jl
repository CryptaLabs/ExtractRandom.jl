module ExtractRandom
using ArgCheck
export two_universal, two_universal!, poisson, poisson_min_entropy, bit_representation,
       integer_representation

""" Poisson distribution """
poisson(events::Number, average::Number) = e^-average * average^events / factorial(events)
""" Min entropy for a poisson distribution with mean n """
poisson_min_entropy(average::Number; basis::Real=e) = begin
    -log(poisson(average, average)) / log(basis)
end

"""
```Julia
two_universal!(a::AbstractMatrix{Bool}, v::AbstractVector{Bool}, out::AbstractVector{Bool})
```

Applies the two universal hashing function (parameterized by `a`) to the vector `v`.
"""
two_universal!(a::AbstractMatrix{Bool}, v::AbstractVector{Bool},
               out::AbstractVector{Bool}) = begin
    @argcheck length(v) % size(a, 2) == 0
    @argcheck length(out) % size(a, 1)== 0
    @argcheck size(a, 1) * length(v) == size(a, 2) * length(out)
    n, m = size(a)
    for i in 1:length(out)
        out[i] = v[div(i - 1, n) * m + 1] & a[(i - 1) % n + 1, 1]
    end
    for j in 2:m
        for i in 1:length(out)
            vindex = j + div(i - 1, n) * m
            out[i] $= v[vindex] & a[(i - 1) % n + 1, (j - 1) % m + 1]
        end
    end
    out
end

two_universal{T}(::Type{Val{T}}, a::AbstractMatrix{Bool}, v::AbstractVector{Bool}) = begin
    @argcheck length(v) % size(a, 2) == 0
    n = div(length(v), size(a, 2)) * size(a, 1)
    two_universal!(a, v, similar(v, (n,)))
end

two_universal(a::AbstractMatrix{Bool}, v::AbstractVector{Bool}) = begin
    @argcheck length(v) % size(a, 2) == 0
    n = div(length(v), size(a, 2)) * size(a, 1)
    two_universal!(a, v, similar(v, (n,)))
end

two_universal{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T}) = begin
    @argcheck length(v) % size(a, 2) == 0
    n = round(Int64, (div(length(v), size(a, 2)) * size(a, 1)) / 8sizeof(T), RoundUp)
    two_universal!(a, v, similar(v, (n,)))
end

two_universal{T <: Integer, V}(::Type{Val{V}}, a::AbstractMatrix{T},
                               v::AbstractVector{T}) = begin
    @argcheck length(v) % size(a, 2) == 0
    n = round(Int64, (div(length(v), size(a, 2)) * size(a, 1)) / 8sizeof(T), RoundUp)
    two_universal!(Val{V}, a, v, similar(v, (n,)))
end

two_universal!{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T},
                             out::AbstractVector{T}) = begin
    two_universal!(Val{:naive}, a, v, out)
end

two_universal!{T <: Integer}(::Type{Val{:naive}}, a::AbstractMatrix{T}, v::AbstractVector{T},
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

two_universal!{T <: Integer}(::Type{Val{:work}}, a::AbstractMatrix{T}, v::AbstractVector{T},
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


    work = similar(v, (size(a, 1), ))
    _two_universal!(Val{:work}, a, v, out, work)
end

_two_universal!{T <: Integer}(::Type{Val{:work}}, a::AbstractMatrix{T}, v::AbstractVector{T},
                              out::AbstractVector{T}, work::AbstractVector{T}) = begin
    nbits = 8sizeof(eltype(out))
    n, m = size(a)

    fill!(out, zero(T))
    bit_offset, in_offset = 1, 1
    for multiple in 0:(div(length(v), size(a, 2)) - 1)
        v_view = view(v, in_offset:min(length(v), in_offset - 1 + m))

        out_start = div(bit_offset - 1, nbits) + 1
        out_end = min(length(out), round(Int64, (bit_offset + n - 1)/nbits, RoundUp))
        out_view = view(out, out_start:out_end)
        out_offset = bit_offset - (out_start - 1) * nbits - 1
        _loop_two_universal!(a, v_view, out_view, work, out_offset)

        bit_offset += n
        in_offset += m
    end
    out
end

_loop_two_universal!(a::AbstractMatrix, v::AbstractVector, out::AbstractVector,
                     work::AbstractVector, out_offset) = begin
    nbits = 8sizeof(eltype(out))
    vⱼ = v[1]
    for i in eachindex(work)
        work[i] = vⱼ & a[i, 1]
    end

    for j in drop(eachindex(v), 1)
        vⱼ = v[j]
        for i in eachindex(work)
            work[i] $= vⱼ & a[i, j]
        end
    end

    for ibit in eachindex(work)
        bit = convert(eltype(out), xorbits(work[ibit]))
        work_bit = ibit + out_offset
        shift = nbits - 1 - (work_bit - 1) % nbits
        out[div(work_bit - 1, nbits) + 1] |= bit << shift
    end
    nothing
end

@inline bget(a::Integer, i::Integer) = Bool(a & (1 << i))

xorbits(a::Integer) = begin
    result::Bool = a & 1
    for i in 1:(sizeof(a) * 8 - 1)
        result $= (a & (1 << i)) ≠ 0
    end
    result::Bool
end

"""
    integer_representation{T < Integer}(::Type{T}, bits::AbstractArray{Bool})

Creates array with same bit representation as input bitstring
"""
integer_representation{T <: Integer}(::Type{T}, bits::AbstractArray{Bool}) = begin
    nbits = 8sizeof(T)
    N = round(Int64, length(bits) / nbits, RoundUp)
    result = zeros(T, N)
    for i in eachindex(bits)
        j = div(i - 1, nbits) + 1
        j′ = nbits - 1 - (i - 1) % nbits
        result[j] |= convert(T, bits[i]) << j′
    end
    result
end

"""
    bit_representation(input::AbstractArray)

Creates bit string with same bit representation as input array. The first dimension of the
result is expanded by `8sizeof(eltype(input))`.
"""
bit_representation(input::AbstractArray) = begin
    const T = eltype(input)
    const nbits = 8sizeof(T)
    result = BitArray(size(input, 1) * nbits, size(input)[2:end]...)
    for i in eachindex(input)
        for i′ in 0:(nbits - 1)
            result[(i - 1) * nbits + nbits - i′] = (input[i] & (1 << i′)) ≠ 0
        end
    end
    result
end

end # module
