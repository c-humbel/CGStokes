using KernelAbstractions
using Enzyme
using Metal


# ##### Measure runtime for different workgroup sizes & backends #####
@kernel inbounds=true function do_stuff(a)
    I = @index(Global)
    for _ in 1:100
        a[I] = atan(a[I])
    end
end

backend = MetalBackend()
n = 2^10

a = KernelAbstractions.zeros(backend, Float32, n, n)

copyto!(a, rand(Float32, n, n))


function measure_mykernel(a, backend, wg) 
    mykernel = do_stuff(backend, wg, size(a))

    mykernel(a)

    @timev begin
        mykernel(a)
        KernelAbstractions.synchronize(backend)
    end
end

measure_mykernel(a, backend, 64)


# ##### Using Enzyme to differentiate kernels #####
@kernel inbounds=true function set_sq(a, @Const(b))
    I = @index(Global)
    a[I] = b[I]*b[I]
end


backend = MetalBackend()
a = KernelAbstractions.zeros(backend, Float32, 100, 100)
b  = deepcopy(a)
da = deepcopy(a)
db = deepcopy(a)

b  .= 1f0
db .= 1f0

# this does not work using Metal Backend (ie. takes longer to compile/evaluate than I have patience)

ker = set_sq(backend, 64, size(a))
autodiff(Forward, ker, DuplicatedNoNeed(a, da), Duplicated(b, db))


# ##### Call kernel with 1D ndrange #####
@kernel function set_bc_x_zero(A)
    i, = @index(Global, NTuple)
    A[i, 1]   = 0
    A[i, end] = 0
end

backend = MetalBackend()

a = KernelAbstractions.zeros(backend, Float32, 100, 3)
fill!(a, 3)

ker = set_bc_x_zero(backend, 64, 100)

ker(a)