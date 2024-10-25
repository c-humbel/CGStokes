using Enzyme
using LinearAlgebra


function f(s, u, a, b, c)
    for i = axes(u.x, 1)[1:end-1]
        s.x[i] = a * (u.x[i+1] - u.x[i]) + b * (u.y[i+1] - u.y[i]) + c
        s.y[i] = a * (u.y[i+1] - u.y[i]) + b * (u.x[i+1] - u.x[i]) + 5c
    end
end

u = (x=rand(10), y=rand(10))
p = (x=ones(10), y=ones(10))
p.x[1] = 0; p.x[end] = 0
p.y[1] = 0; p.y[end] = 0
s = (x=zeros(9), y=zeros(9))
t = (x=zeros(9), y=zeros(9))
autodiff(Forward, f, Duplicated(s, t), Duplicated(u, p), Const(3), Const(2), Const(10000))