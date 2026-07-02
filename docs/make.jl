using Documenter, SinCosLUT

makedocs(
    sitename = "SinCosLUT.jl",
    modules  = [SinCosLUT],
    authors  = "Soeren Schoenbrod and contributors",
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages    = [
        "Home"                       => "index.md",
        "Usage guide"                => "guide.md",
        "Fused, array-free"          => "fused.md",
        "Accuracy & drift-free phase" => "accuracy.md",
        "Benchmarks"                 => "benchmarks.md",
        "API reference"              => "api.md",
    ],
    checkdocs = :exports,   # fail the build if an exported symbol loses its docstring
)

deploydocs(
    repo = "github.com/JuliaGNSS/SinCosLUT.jl.git",
    push_preview = true,
)
