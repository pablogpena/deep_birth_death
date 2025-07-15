# +
ARG_PATH="$1"

ID=$(basename "$ARG_PATH")

{ time Rscript run_DDD.r "$ARG_PATH" ; } 2> "/workspace/deep_birth_death/MLE/inference_data/time/DDD_time_${ID}.txt"
