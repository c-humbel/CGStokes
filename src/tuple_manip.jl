using LinearAlgebra

function tplNorm(x::NamedTuple, p::Real=2)
    return norm(norm.(values(x), p), p)   
end


function tplDot(x::NamedTuple, y::NamedTuple, a::NamedTuple)
    mult(a1, a2, a3) = a1*a2*a3
    s = 0.
    for k = keys(x)
        s += mapreduce(mult, +, x[k], a[k], y[k])
    end
    return s
end


function tplDot(x::NamedTuple, y::NamedTuple, a::Real=1.)
    return a * sum(dot.(values(x), values(y)))
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::NamedTuple)
    for k = keys(dest)
        copyto!(dest[k], src[k])
        dest[k] .*= a[k]
    end
    return nothing
end


function tplSet!(dest::NamedTuple, src::NamedTuple)
    copyto!.(values(dest), values(src))
    return nothing
end

function tplFill!(dest::NamedTuple, value::Real)
    fill!.(values(dest), value)
    return nothing
end


function tplScale!(x::NamedTuple, a::Real)
    for k = keys(x)
        x[k] .*= a
    end
    return nothing
end


function tplAdd!(this::NamedTuple, other::NamedTuple)
    for k = keys(this)
        this[k] .+= other[k]
    end
    return nothing
end

function tplSub!(this::NamedTuple, other::NamedTuple)
    for k = keys(this)
        this[k] .-= other[k]
    end
    return nothing
end