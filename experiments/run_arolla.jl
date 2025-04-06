include("ISMIP_Arolla.jl")

args = Dict(:n              => 1000,
            :aspect         => 0.3,
            :γ_factor       => 1e5,
            :niter          => 5000000,
            :ϵ_ph           => 1e-8,
            :ϵ_cg           => 1e-10,
            :ϵ_newton       => 1e-8,
            :freq_recompute => 50,
            :verbose        => false,
            :backend        => CUDABackend(),
            :workgroup      => (32, 8))

stats = @timed run("data/arolla100.dat"; args...)
dat = extract_data(stats.value..., save=true)
create_summary_plots(dat...; savefig=true)

# output timing summary
println("total time (s), ", stats.time)
println("compilation (s), ", stats.compile_time + stats.recompile_time)
println("garbage collection (s), ", stats.gctime)
println("runtime (including gc), ", stats.time - stats.compile_time - stats.recompile_time)