using Literate

const DOCS_DIR     = @__DIR__
const LITERATE_DIR = joinpath(DOCS_DIR, "tutorials")
const GEN_DIR      = joinpath(DOCS_DIR, "src", "gen")

isdir(LITERATE_DIR) || return String[]

files = sort(filter(f -> endswith(f, ".jl"), readdir(LITERATE_DIR; join = true)))
GENERATED_FILES = String[]

for src in files
	# Output markdown filename
	base = splitext(basename(src))[1]
	# Generate markdown in docs/src/gen
	Literate.markdown(
		src,
		GEN_DIR;
		documenter = true,
		execute = true,
	)

	# Return relative path from docs/src
	push!(GENERATED_FILES, joinpath("gen", base * ".md"))
end

