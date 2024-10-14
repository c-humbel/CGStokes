using Enzyme
using LinearAlgebra


function f(s, u, a, b, c)
    @. s.x  = a * u.x + b * u.y + c
    @. s.y  = a * u.y - b * u.x + 5c
end

u = (x=ones(3,3), y=ones(3,3))
p = (x=zeros(3,3), y=zeros(3,3))
p.x[1,1] = 1
s = (x=zeros(3,3), y=zeros(3,3))
t = (x=zeros(3,3), y=zeros(3,3))
autodiff(Forward, f, Duplicated(s, t), Duplicated(u, p), Const(3), Const(2), Const(10000))