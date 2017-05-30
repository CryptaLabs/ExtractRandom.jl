"""
```Julia
two_universal!(a::AbstractMatrix{Bool}, v::AbstractVector{Bool}, out::AbstractVector{Bool})
```

Applies the two universal hashing function (parameterized by `a`) to the vector `v`.
"""
bits_two_universal!(a::AbstractMatrix{Bool}, v::AbstractVector{Bool},
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

immutable BitTwoUniversal{CONTAINER <: AbstractMatrix{Bool}} <: AbstractTwoUniversal
    matrix::CONTAINER
end
Base.size(func::BitTwoUniversal) = size(func.matrix)
Base.size(func::BitTwoUniversal, i::Integer) = size(func.matrix, i)

(func::BitTwoUniversal)(v::AbstractVector) = begin
    @argcheck length(v) % size(func, 2) == 0
    n = div(length(v), size(func, 2)) * size(func, 1)
    two_universal!(func, v, similar(v, (n,)))
end

two_universal!(func::BitTwoUniversal, v::AbstractVector, out::AbstractVector) = begin
    bits_two_universal!(func.matrix, v, out)
end
