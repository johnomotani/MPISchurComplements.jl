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
    dbr_values = Int64[]
    nproc_array = Int64[]
    tile_sizes = Int64[]
    for line in eachline(filename)
        valstrings = split(line)
        valcounter = 0
        matsize = parse(Int64, valstrings[valcounter+=1])
        tile_size = parse(Int64, valstrings[valcounter+=1])
        imat = parse(Int64, valstrings[valcounter+=1])
        distributed_nproc = parse(Int64, valstrings[valcounter+=1])
        shared_nproc = parse(Int64, valstrings[valcounter+=1])
        distributed_block_rows = parse(Int64, valstrings[valcounter+=1])
        t_factorisation = parse(Float64, valstrings[valcounter+=1])
        t_trisolve_min = parse(Float64, valstrings[valcounter+=1])
        t_trisolve_mean = parse(Float64, valstrings[valcounter+=1])
        t_trisolve_max = parse(Float64, valstrings[valcounter+=1])
        t_total = parse(Float64, valstrings[valcounter+=1])
        if valcounter != length(valstrings)
            error("valcounter=$valcounter not the same as number of variables in log file $(length(valstrings))")
        end

        nproc = distributed_nproc * shared_nproc

        level2 = get_entry(factorisation_timings, matsize, Dict{Int64,Any}())
        level3 = get_entry(level2, distributed_block_rows, Dict{Int64,Any}())
        level4 = get_entry(level3, nproc, Dict{Int64,Any}())
        level5 = get_entry(level4, shared_nproc, Dict{Int64,Any}())
        level6 = get_entry(level5, tile_size, Dict{Int64,Any}())
        level6[imat] = t_factorisation

        level2 = get_entry(solve_timings, matsize, Dict{Int64,Any}())
        level3 = get_entry(level2, distributed_block_rows, Dict{Int64,Any}())
        level4 = get_entry(level3, nproc, Dict{Int64,Any}())
        level5 = get_entry(level4, shared_nproc, Dict{Int64,Any}())
        level6 = get_entry(level5, tile_size, Dict{Int64,Any}())
        level6[imat] = t_trisolve_mean

        if distributed_block_rows ∉ dbr_values
            push!(dbr_values, distributed_block_rows)
        end
        if nproc ∉ nproc_array
            push!(nproc_array, nproc)
        end
        if tile_size ∉ tile_sizes
            push!(tile_sizes, tile_size)
        end
    end

    sort!(tile_sizes)
    sort!(nproc_array)

    return factorisation_timings, solve_timings, dbr_values, nproc_array, tile_sizes
end

function convert_timings2_to_Array(timings2, dbr_values, nproc_array, tile_sizes)
    max_nproc = maximum(nproc_array)
    n_distributed_block_rows = length(dbr_values)

    # NaN will stand for missing entries, so initialise with NaN.
    timings_array = fill(Float64(NaN), max_nproc, max_nproc, n_distributed_block_rows, length(tile_sizes))
    for (idbr, (distributed_block_rows, timings3)) ∈ enumerate(pairs(timings2))
        for (nproc, timings4) ∈ pairs(timings3)
            for (n_shared, timings5) ∈ pairs(timings4)
                for (tile_size, timings6) ∈ pairs(timings5)
                    # Keys in timings5 are just matrix labels. The matrices are random, so we
                    # just average the timings over all of them.
                    tile_size_index = searchsortedfirst(tile_sizes, tile_size)
                    this_timing = sum(values(timings6)) / length(timings6)
                    timings_array[nproc,n_shared,idbr,tile_size_index] = this_timing
                end
            end
        end
    end

    return timings_array, nproc_array
end

function plot_for_matrix_size(timings2, dbr_values, nproc_array, tile_sizes, mat_size,
                              operation_label, solver_label, tee)
    timings_array, nproc_array = convert_timings2_to_Array(timings2, dbr_values,
                                                           nproc_array, tile_sizes)
    max_nproc = maximum(nproc_array)

    # Find the best times for any tile_size or n_shared for each nproc.
    timings_missing_is_inf = copy(timings_array)
    timings_missing_is_inf[isnan.(timings_missing_is_inf)] .= Inf
    optimal_times = Float64[]
    optimal_dbr = Int64[]
    optimal_n_shared = Int64[]
    optimal_tile_size = Int64[]
    for nproc ∈ 1:max_nproc
        imin = argmin(timings_missing_is_inf[nproc,:,:,:])
        push!(optimal_times, timings_missing_is_inf[nproc,imin])
        push!(optimal_n_shared, imin[1])
        push!(optimal_dbr, dbr_values[imin[2]])
        push!(optimal_tile_size, tile_sizes[imin[3]])
    end

    println(tee, "Best parameters for each nproc")
    println(tee, "nproc\tshared\tdbr\ttile_size")
    for nproc ∈ 1:max_nproc
        if isfinite(optimal_times[nproc])
            println(tee, nproc, "\t", optimal_n_shared[nproc], "\t", optimal_dbr[nproc], "\t", optimal_tile_size[nproc])
        end
    end

    fig = Figure()
    #ax = Axis(fig[1,1]; xlabel="nproc", ylabel="$operation_label time", title="$solver_label $mat_size", yscale=log10)
    ax = Axis(fig[1,1]; xlabel="nproc", ylabel="$operation_label time", title="$solver_label $mat_size")

    marker_styles = (:circle, :cross, :xcross, :utriangle, :dtriangle, :ltriangle, :rtriangle)
    linestyle_pairs = ((:solid, :dash),
                       (:dot, :dashdot),
                       ((:dot, :dense), (:dashdot, :dense)),
                       ((:dot, :loose), (:dashdot, :loose)),
                      )

    nvals = 1:max_nproc
    for (i, (dbr, ms, ls)) ∈ enumerate(zip(dbr_values, marker_styles, linestyle_pairs))
        if dbr == 0
            dbr_label = ""
        else
            dbr_label = "dbr=$dbr, "
        end
        for (j, tile_size) ∈ enumerate(tile_sizes)
            for t ∈ eachcol(timings_array[:,:,i,j])
                scatter!(ax, t; marker=ms, color=j, colormap=:tab10, colorrange=(1,10), label="$(dbr_label)tile_size=$tile_size")
                #scatter!(ax, t; marker=ms, color=j, colorrange=(1,10), label="dbr=$dbr, tile_size=$tile_size")
            end
            all_shared_times = [timings_array[n,n,i,j] for n ∈ nvals]
            valid_ninds_all_shared = @. isfinite(all_shared_times)
            no_shared_times = timings_array[:,1,i,j]
            valid_ninds_no_shared = @. isfinite(no_shared_times)
            lines!(ax, nvals[valid_ninds_all_shared], all_shared_times[valid_ninds_all_shared]; linestyle=ls[1], color=j, colormap=:tab10, colorrange=(1,10), label="$(dbr_label)tile_size=$tile_size, n_shared=nproc")
            lines!(ax, nvals[valid_ninds_no_shared], no_shared_times[valid_ninds_no_shared]; linestyle=ls[2], color=j, colormap=:tab10, colorrange=(1,10), label="$(dbr_label)tile_size=$tile_size, n_shared=1")
        end
    end

    # Ensure the first row width is 3/4 of the column width so that the plot does not get
    # squashed by the legend
    rowsize!(fig.layout, 1, Aspect(1, 3/4))

    Legend(fig[2,1], ax; tellwidth=false, tellheight=true, unique=true)

    resize_to_layout!(fig)

    save(joinpath(results_dir, "$solver_label-$operation_label-$mat_size.pdf"), fig)

    return optimal_times
end

function plot_timings(timings1, dbr_values, nproc_array, tile_sizes, operation_label, solver_label, tee)
    sorted_keys = sort(collect(keys(timings1)))
    optimal_times = Tuple{Vector{Float64},String}[]
    println(tee, operation_label)
    println(tee, "-" ^ length(operation_label))
    for mat_size ∈ sorted_keys
        println(tee, "mat_size = $mat_size")
        println(tee, "^^^^^^^^^^^^^^^^")
        timings2 = timings1[mat_size]
        this_optimal_times =
            plot_for_matrix_size(timings2, dbr_values, nproc_array, tile_sizes, mat_size,
                                 operation_label, solver_label, tee)
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
            nvals = 1:length(times)
            valid_ninds = @. isfinite(times)
            lines!(ax, nvals[valid_ninds], times[valid_ninds]; label=label,
                   linestyle=linestyles[i], color=j, colormap=:tab20, colorrange=(1,20))
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

function plot_benchmarks(suffix="")
    optimal_factorisation_times = Tuple{Vector{Tuple{Vector{Float64},String}},String}[]
    optimal_solve_times = Tuple{Vector{Tuple{Vector{Float64},String}},String}[]
    mkpath(results_dir)
    io = open(joinpath(results_dir, "optimal_parameters$suffix.log"), "w")
    tee = TeeStream(stdout, io)
    for (filename, label) ∈ (("timings-julia$suffix.log", "DenseLUs"),
                             ("timings-fortran.log", "ScaLAPACK"),
                             ("timings-linearalgebra$suffix.log", "LinearAlgebra"),
                            )
        if isfile(filename)
            println(tee, label)
            println(tee, "=" ^ length(label))
            factorisation_timings, solve_timings, dbr_values, nproc_array, tile_sizes = read_timings(filename)
            push!(optimal_factorisation_times,
                  (plot_timings(factorisation_timings, dbr_values, nproc_array, tile_sizes, "factorisation", label, tee),
                   label))
            push!(optimal_solve_times,
                  (plot_timings(solve_timings, dbr_values, nproc_array, tile_sizes, "solve", label, tee),
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

if length(ARGS) > 0
    plot_benchmarks(ARGS[1])
else
    plot_benchmarks()
end
