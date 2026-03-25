using JuliaC

rm("build"; force=true, recursive=true)

img = ImageRecipe(
    output_type    = "--output-exe",
    file           = "src/MyApp.jl",
    trim_mode      = "safe",
    add_ccallables = false,
    verbose        = true,
)

link = LinkRecipe(
    image_recipe = img,
    outname      = "build/my_app",
    rpath        = nothing, # set automatically when bundling
)

bun = BundleRecipe(
    link_recipe = link,
    output_dir  = "build" # or `nothing` to skip bundling
)

compile_products(img)
link_products(link)
bundle_products(bun)
