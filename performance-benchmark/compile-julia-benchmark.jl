using PackageCompiler

# Create the sysimage 'julia-benchmark.so'
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-compile if you change anything.
create_sysimage(; sysimage_path="julia-benchmark.so",
                precompile_execution_file="_internal-precompile-script.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3`, # Assume if we are precompiling we want an optimized, production build
               )

