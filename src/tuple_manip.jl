using LinearAlgebra

function tplNorm(x::NamedTuple, p::Real=2)
    return norm(norm.(values(x), p), p)   
end


function tplDot(x::NamedTuple, y::NamedTuple, a::NamedTuple)
    s = 0.
    for k = keys(x)
        s += dot(x[k], a[k] .* y[k])
    end
    return s
end


function tplDot(x::NamedTuple, y::NamedTuple, a::Real=1.)
    return sum(dot.(values(x), a .* values(y)))
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::NamedTuple)
    for k = keys(dest)
        copyto!(dest[k], a[k] .* src[k])
    end
    return nothing
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::Real=1.)
    copyto!.(values(dest), a .* values(src))
    return nothing
end


function tplScale!(x::NamedTuple, a::Real)
    for k = keys(x)
        x[k] .= a .* x[k]
    end
    return nothing
end