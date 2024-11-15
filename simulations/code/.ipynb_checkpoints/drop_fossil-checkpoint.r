# -*- coding: utf-8 -*-
# # Package and configuration import

args = commandArgs(trailingOnly=TRUE)
source("/workspace/coniferas/simulations/config_sim.r")

require(ape)
require(phytools)
require(geiger)

# # Load and delete extant

# +
file_path <- paste(raw_path, args[1], sep = "")
lines <- readLines(file_path)

out_lines <- list()

# Header line
out_lines[1] = lines[1]
for (i in 2:length(lines)) {
    # Get last element from line
    elems <- strsplit(lines[i], "\\|")[[1]]
    tree <- elems[length(elems)]

    # Read tree of corresponding line
    tree <- read.tree(text = tree)
    
    # Delete extant tips
    tree <- drop.extinct(tree)
    
    # Change line with new tree without fossils
    elems[length(elems)] <- write.tree(tree)
    
    # Generate new line
    out_lines[i] <- paste(elems, collapse = "|")
}

print(paste(length(lines) - 1, " trees cleaned"))
# -

# # Write output file

out_file_path <- paste(no_fossil_path, args[1], sep = "")
out_file_path <- sub("_sim_", "_sim_no_fossil_", out_file_path)
out_lines = as.character(out_lines)
writeLines(out_lines, out_file_path)
