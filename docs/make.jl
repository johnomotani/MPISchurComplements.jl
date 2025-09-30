using Pkg

Pkg.instantiate()

using Documenter
using MPISchurComplements

makedocs(
    sitename = "MPISchurComplements",
    format = Documenter.HTML(),
    modules = [MPISchurComplements]
)

if get(ENV, "CI", nothing) == "true"
    # Documenter can also automatically deploy documentation to gh-pages.
    # See "Hosting Documentation" and deploydocs() in the Documenter manual
    # for more information.
    deploydocs(
        repo = "github.com/johnomotani/MPISchurComplements.jl",
        push_preview = true,
    )
end
