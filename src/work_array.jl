work_two_universal!{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T},
                                  out::AbstractVector{T}, work::AbstractVector{T}) = begin
    Nbits = 8sizeof(eltype(out))
    multiples = div(length(v), size(a, 2))
    @argcheck length(a) ≠ 0
    if length(v) == 0
        fill!(out, zero(T))
        return out
    end
    @argcheck length(v) % size(a, 2) == 0
    @argcheck round(Int64, (multiples * size(a, 1)) / Nbits, RoundUp) == length(out)
    @argcheck length(work) == size(a, 1)

    _work_two_universal!(a, v, out, work)
end

_work_two_universal!{T <: Integer}(a::AbstractMatrix{T}, v::AbstractVector{T},
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

immutable WorkArrayTwoUniversal{T <: Integer} <: AbstractTwoUniversal
    matrix::Matrix{T}
    work::Vector{T}
end

WorkArrayTwoUniversal(a::AbstractMatrix) = begin
    WorkArrayTwoUniversal(convert(Matrix{eltype(a)}, a),
                          zeros(eltype(a), size(a, 1)))
end

Base.size(func::WorkArrayTwoUniversal) = size(func.matrix)
Base.size(func::WorkArrayTwoUniversal, i::Integer) = size(func.matrix, i)

two_universal!(func::WorkArrayTwoUniversal, v::AbstractVector, out::AbstractVector) = begin
    work_two_universal!(func.matrix, v, out, func.work)
end
