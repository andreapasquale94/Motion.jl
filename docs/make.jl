using Documenter
using Motion 

include("generate.jl")

const CI = get(ENV, "CI", "false") == "true"

makedocs(
    modules = [Motion],
    sitename = "Motion.jl",
    authors="Andrea Pasquale <andrea.pasquale@outlook.it>",
    format=Documenter.HTML(; prettyurls=CI, highlights=["yaml"], ansicolor=true),
    pages = [
        "Home" => "index.md",
        "Tutorials" => GENERATED_FILES
    ],
    clean=true,
    checkdocs=:none
);

if CI
    deploydocs(;
        repo="github.com/andreapasquale94/Motion.jl", branch="gh-pages"
    )
end
