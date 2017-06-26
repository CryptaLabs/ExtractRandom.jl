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

abstract AbstractTwoUniversal

include("xorbits.jl")
include("bits.jl")
include("naive.jl")
include("work_array.jl")
include("row_major.jl")

for Algo in [NaiveTwoUniversal, WorkArrayTwoUniversal, RowMajorTwoUniversal]
    @eval begin
        (func::$Algo)(v::AbstractVector) = begin
            @argcheck length(v) % size(func, 2) == 0
            T = eltype(v)
            n = round(Int64, (div(length(v), size(func, 2)) * size(func, 1)) / 8sizeof(T),
                      RoundUp)
            two_universal!(func, v, similar(v, (n,)))
        end

        (func::$Algo)(v::AbstractVector, out::AbstractVector) = begin
            two_universal!(func, v, out)
        end
    end
end

two_universal(a::AbstractMatrix{Bool}, v::AbstractVector{Bool}) = begin
    BitTwoUniversal(a)(v)
end
two_universal(algo::Symbol, a::AbstractMatrix{Bool}, v::AbstractVector{Bool}) = begin
    algo == :bit || error("BitArray and arrays of bool should use the bits algo")
    two_universal(a, v)
end
two_universal!(a::AbstractMatrix{Bool}, v::AbstractVector{Bool},
               out::AbstractVector{Bool}) = begin
    BitTwoUniversal(a)(v, out)
end
two_universal!(algo::Symbol, a::AbstractMatrix{Bool}, v::AbstractVector{Bool},
               out::AbstractVector{Bool}) = begin
    algo == :bit || error("BitArray and arrays of bool should use the bits algo")
    two_universal!(a, v, out)
end

two_universal{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T}) = begin
    RowMajorTwoUniversal(transpose(a))(v)
end
two_universal{T <: Integer}(algo::Symbol, a::AbstractMatrix{T},
                            v::AbstractVector{T}) = begin
    if algo == :bits
        error("Incorrect input types. Should be bitarrays.")
    elseif algo == :naive
        NaiveTwoUniversal(a)(v)
    elseif algo == :work_array
        WorkArrayTwoUniversal(a)(v)
    elseif algo == :row_major
        RowMajorTwoUniversal(transpose(a))(v)
    else
        error("Unknown algorithm $algo")
    end
end

two_universal!{T <: Integer}(algo::Symbol, a::AbstractMatrix{T}, v::AbstractVector{T},
                             out::AbstractVector{T}) = begin
    if algo == :bits
        error("Incorrect input types. Should be bitarrays.")
    elseif algo == :naive
        NaiveTwoUniversal(a)(v, out)
    elseif algo == :work_array
        WorkArrayTwoUniversal(a)(v, out)
    elseif algo == :row_major
        RowMajorTwoUniversal(transpose(a))(v, out)
    else
        error("Unknown algorithm $algo")
    end
end

two_universal!{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T},
                             out::AbstractVector{T}) = begin
    RowMajorTwoUniversal(transpose(a))(v, out)
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
    result = BitArray(size(input, 1) * nbits,
                      collect(size(input, u) for u in 2:ndims(input))...)
    for i in eachindex(input)
        for i′ in 0:(nbits - 1)
            result[(i - 1) * nbits + nbits - i′] = (input[i] & (1 << i′)) ≠ 0
        end
    end
    result
end

end # module
