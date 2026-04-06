using CairoMakie
using TeeStreams

# Data stored in two nested Dicts, one for factorisation timings and the other for solve
# timings. The keys at each level are:
#   1) matrix size
#   2) nproc = distributed_nproc * shared_nproc
#   3) shared_nproc
#   4) tile size
#   5) imat
# and the value is the timing.

const results_dir = "results-plots"

function get_entry(d, key, default)
    if key ∉ keys(d)
        d[key] = default
    end
    return d[key]
end

function read_timings(filename)
    factorisation_timings = Dict{Int64,Any}()
    solve_timings = Dict{Int64,Any}()
    tile_sizes = Int64[]
    for line in eachline(filename)
        valstrings = split(line)
        matsize = parse(Int64, valstrings[1])
        tile_size = parse(Int64, valstrings[2])
        imat = parse(Int64, valstrings[3])
        distributed_nproc = parse(Int64, valstrings[4])
        shared_nproc = parse(Int64, valstrings[5])
        t_factorisation = parse(Float64, valstrings[6])
        t_trisolve_min = parse(Float64, valstrings[7])
        t_trisolve_mean = parse(Float64, valstrings[8])
        t_trisolve_max = parse(Float64, valstrings[9])

        nproc = distributed_nproc * shared_nproc

        level2 = get_entry(factorisation_timings, matsize, Dict{Int64,Any}())
        level3 = get_entry(level2, nproc, Dict{Int64,Any}())
        level4 = get_entry(level3, shared_nproc, Dict{Int64,Any}())
        level5 = get_entry(level4, tile_size, Dict{Int64,Any}())
        level5[imat] = t_factorisation

        level2 = get_entry(solve_timings, matsize, Dict{Int64,Any}())
        level3 = get_entry(level2, nproc, Dict{Int64,Any}())
        level4 = get_entry(level3, shared_nproc, Dict{Int64,Any}())
        level5 = get_entry(level4, tile_size, Dict{Int64,Any}())
        level5[imat] = t_trisolve_mean

        if tile_size ∉ tile_sizes
            push!(tile_sizes, tile_size)
        end
    end

    sort!(tile_sizes)

    return factorisation_timings, solve_timings, tile_sizes
end

function convert_timings2_to_Array(timings2, tile_sizes)
    nproc_array = sort(collect(keys(timings2)))
    max_nproc = maximum(nproc_array)

    # NaN will stand for missing entries, so initialise with NaN.
    timings_array = fill(Float64(NaN), max_nproc, max_nproc, length(tile_sizes))
    for (nproc, timings3) ∈ pairs(timings2)
        for (n_shared, timings4) ∈ pairs(timings3)
            for (tile_size, timings5) ∈ pairs(timings4)
                # Keys in timings5 are just matrix labels. The matrices are random, so we
                # just average the timings over all of them.
                tile_size_index = searchsortedfirst(tile_sizes, tile_size)
                this_timing = sum(values(timings5)) / length(timings5)
                timings_array[nproc,n_shared,tile_size_index] = this_timing
            end
        end
    end

    return timings_array, nproc_array
end

function plot_for_matrix_size(timings2, tile_sizes, mat_size, operation_label, solver_label, tee)
    timings_array, nproc_array = convert_timings2_to_Array(timings2, tile_sizes)
    max_nproc = maximum(nproc_array)

    # Find the best times for any tile_size or n_shared for each nproc.
    timings_missing_is_inf = copy(timings_array)
    timings_missing_is_inf[isnan.(timings_missing_is_inf)] .= Inf
    optimal_times = Float64[]
    optimal_n_shared = Int64[]
    optimal_tile_size = Int64[]
    for nproc ∈ 1:max_nproc
        imin = argmin(timings_missing_is_inf[nproc,:,:])
        push!(optimal_times, timings_missing_is_inf[nproc,imin])
        push!(optimal_n_shared, imin[1])
        push!(optimal_tile_size, tile_sizes[imin[2]])
    end

    println(tee, "Best parameters for each nproc")
    println(tee, "nproc\tshared\ttile_size")
    for nproc ∈ 1:max_nproc
        println(tee, nproc, "\t", optimal_n_shared[nproc], "\t", optimal_tile_size[nproc])
    end

    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="nproc", ylabel="$operation_label time", title="$solver_label $mat_size", yscale=log10)
    for (i, tile_size) ∈ enumerate(tile_sizes)
        for t ∈ eachcol(timings_array[:,:,i])
            scatter!(ax, t; color=i, colormap=:tab10, colorrange=(1,10), label="tile_size=$tile_size")
            #scatter!(ax, t; color=i, colorrange=(1,10), label="tile_size=$tile_size")
        end
        lines!(ax, timings_array[:,1,i]; linestyle=:dash, color=i, colormap=:tab10, colorrange=(1,10), label="tile_size=$tile_size, n_shared=1")
        lines!(ax, [timings_array[n,n,i] for n ∈ 1:max_nproc]; linestyle=:solid, color=i, colormap=:tab10, colorrange=(1,10), label="tile_size=$tile_size, n_shared=nproc")
    end

    # Ensure the first row width is 3/4 of the column width so that the plot does not get
    # squashed by the legend
    rowsize!(fig.layout, 1, Aspect(1, 3/4))

    Legend(fig[2,1], ax; tellwidth=false, tellheight=true, unique=true)

    resize_to_layout!(fig)

    save(joinpath(results_dir, "$solver_label-$operation_label-$mat_size.pdf"), fig)

    return optimal_times
end

function plot_timings(timings1, tile_sizes, operation_label, solver_label, tee)
    sorted_keys = sort(collect(keys(timings1)))
    optimal_times = Tuple{Vector{Float64},String}[]
    println(tee, operation_label)
    println(tee, "-" ^ length(operation_label))
    for mat_size ∈ sorted_keys
        println(tee, "mat_size = $mat_size")
        println(tee, "^^^^^^^^^^^^^^^^")
        timings2 = timings1[mat_size]
        this_optimal_times =
            plot_for_matrix_size(timings2, tile_sizes, mat_size, operation_label,
                                 solver_label, tee)
        push!(optimal_times, (this_optimal_times, "$solver_label $mat_size"))
        println(tee)
    end
    return optimal_times
end

function plot_optimal_times(optimal_times, operation_label)
    linestyles = [:solid, :dash, :dot, :dashdot]
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="nproc", ylabel="optimal $operation_label time",
              title=operation_label, yscale=log10)

    for (i, (this_optimal_times, solver_label)) ∈ enumerate(optimal_times)
        for (j, (times, label)) ∈ enumerate(this_optimal_times)
            lines!(ax, times; label=label, linestyle=linestyles[i], color=j,
                   colormap=:tab20, colorrange=(1,20))
        end
    end

    # Ensure the first row width is 3/4 of the column width so that the plot does not get
    # squashed by the legend
    rowsize!(fig.layout, 1, Aspect(1, 3/4))

    Legend(fig[2,1], ax; tellwidth=false, tellheight=true, unique=true)

    resize_to_layout!(fig)

    save(joinpath(results_dir, "compare-solvers-$operation_label.pdf"), fig)

    return nothing
end

function plot_benchmarks()
    julia_file = "timings-julia.log"
    optimal_factorisation_times = Tuple{Vector{Tuple{Vector{Float64},String}},String}[]
    optimal_solve_times = Tuple{Vector{Tuple{Vector{Float64},String}},String}[]
    mkpath(results_dir)
    io = open(joinpath(results_dir, "optimal_parameters.log"), "w")
    tee = TeeStream(stdout, io)
    for (filename, label) ∈ (("timings-julia.log", "DenseLUs"),
                             ("timings-fortran.log", "ScaLAPACK"),
                             ("timings-linearalgebra.log", "LinearAlgebra"),
                            )
        if isfile(filename)
            println(tee, label)
            println(tee, "=" ^ length(label))
            factorisation_timings, solve_timings, tile_sizes = read_timings(filename)
            push!(optimal_factorisation_times,
                  (plot_timings(factorisation_timings, tile_sizes, "factorisation", label, tee),
                   label))
            push!(optimal_solve_times,
                  (plot_timings(solve_timings, tile_sizes, "solve", label, tee),
                   label))
            println(tee, "********************************************************************")
            println(tee)
        end
    end
    plot_optimal_times(optimal_factorisation_times, "factorisation")
    plot_optimal_times(optimal_solve_times, "solve")

    close(io)

    return nothing
end

plot_benchmarks()
