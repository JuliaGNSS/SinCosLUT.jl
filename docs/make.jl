using Documenter, SinCosLUT

makedocs(
    sitename = "SinCosLUT.jl",
    modules  = [SinCosLUT],
    authors  = "Soeren Schoenbrod and contributors",
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages    = ["Home" => "index.md"],
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/JuliaGNSS/SinCosLUT.jl.git",
    push_preview = true,
)
