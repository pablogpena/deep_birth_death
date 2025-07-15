# +
ARG_PATH="$1"

ID=$(basename "$ARG_PATH")

{ time Rscript run_DDD.r "$ARG_PATH" ; } 2> "/workspace/phylo_estimation/MLE/inference_data/time/DDD_time_${ID}.txt"
