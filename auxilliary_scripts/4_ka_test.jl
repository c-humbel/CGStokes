using KernelAbstractions
using Metal

@kernel inbounds=true function do_stuff(a)
    I = @index(Global)
    for _ in 1:100
        a[I] = atan(a[I])
    end
end

backend = CPU()
n = 2^10

a = KernelAbstractions.zeros(backend, Float32, n, n)

copyto!(a, rand(Float32, n, n))

function measure_mykernel(a, backend, wg) 
    mykernel = do_stuff(backend, wg, size(a))

    mykernel(a, ndrange=size(a))

    @timev begin
        mykernel(a, ndrange=size(a))
        KernelAbstractions.synchronize(backend)
    end
end

measure_mykernel(a, backend, 64)