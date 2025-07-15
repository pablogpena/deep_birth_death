# +
#!/bin/bash

log_file="/workspace/phylo_estimation/simulations/674_10k/timing_log.txt"
mkdir -p "$(dirname "$log_file")"

echo "Timing log - $(date)" > "$log_file"

for arg in "$@"
do
    echo "Running $arg" | tee -a "$log_file"
    
    { 
        time Rscript sim_phylogeny.r "$arg"
    } 2>> "$log_file"
    
    echo "-----" >> "$log_file"
done


# +
#for arg in "$@"
#do
#	Rscript sim_phylogeny.r $arg
#done
