using Documenter
using Motion

include("generate.jl")

const CI = get(ENV, "CI", "false") == "true"

makedocs(
	modules = [
		Motion,
		Motion.CR3BP,
		Motion.ImpulsiveShooting,
		Motion.ConstThrustShooting,
	],
	sitename = "Motion.jl",
	authors = "Andrea Pasquale <andrea.pasquale@outlook.it>",
	format = Documenter.HTML(; 
		prettyurls = CI, 
		highlights = ["yaml"], 
		ansicolor = true,
		size_threshold = nothing
	),
	pages = [
		"Home" => "index.md",
		"Tutorials" => GENERATED_FILES,
		"API" => "api.md",
	],
	clean = true,
	checkdocs = :none,
);

if CI
	deploydocs(;
		repo = "github.com/andreapasquale94/Motion.jl", branch = "gh-pages",
	)
end
