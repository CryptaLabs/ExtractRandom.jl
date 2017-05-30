@inline xorbits(a::Integer) = weird_xorbits(a)

naive_xorbits(a::Integer) = begin
    result::Bool = a & one(a)
    for i in 1:(sizeof(a) * 8 - 1)
        result $= (a & (one(a) << i)) ≠ 0
    end
    result::Bool
end

@inline xorbits_number(::UInt64) = 0x1111111111111111
@inline xorbits_number(::UInt32) = 0x11111111
@inline xorbits_number(::UInt16) = 0x1111
@inline xorbits_number(::UInt8) = 0x11
@inline xorbits_shift(::UInt64) = 60
@inline xorbits_shift(::UInt32) = 28
@inline xorbits_shift(::UInt16) = 12
@inline xorbits_shift(::UInt8) = 4

@inline weird_xorbits(x::Signed) = weird_xorbits(unsigned(x))
@inline weird_xorbits(x::Unsigned) = begin
    x $= x >> 1
    x $= x >> 2
    ((((x & xorbits_number(x)) * xorbits_number(x)) >> xorbits_shift(x)) & 1) ≠ 0
end

