#!/bin/bash

# # Quick test run 
#   python "run_tests_isu.py" \
#     --sut "moondream" \
#     --algorithm "rs" \
#     --population_size 2 \
#     --max_time "00:20:00" \
#     --features_config "configs/isu_features.json" #\
#     #--no_wandb 

#set -x

# Define the list of suts to iterate over
suts=("gpt5") #"gemini-2.5" "gpt5" "isu-bmw" "dummy"
algorithms=("nsga2") #"nsga2" "nsga2d"
budget="03:00:00"
script="run_tests_isu.py"
config_file="configs/isu_features.json"
repeat=1 #6

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for sut in "${suts[@]}"
do
  #repeat experiments
  for ((i=1; i<=repeat; i++))
  do
    seed=$i

    echo "=== Experiment Run: $i for SUT: $sut, seed: $seed ==="

    for algorithm in "${algorithms[@]}"
    do
    echo "--- Running experiments for $sut with $algorithm ---"

    #skip nsga2 for gpt5
    # if [ "$sut" == "gpt5" ]; then
    #   continue
    # fi

    ts=$(date +%Y%m%d-%H%M%S)
    out_log="$LOG_DIR/${sut}${algorithm}_out$ts.log"
    err_log="$LOG_DIR/${sut}${algorithm}_err$ts.log"
    
    python $script \
      --sut "$sut" \
      --algorithm "$algorithm" \
      --population_size 20 \
      --n_generations 30 \
      --max_time $budget \
      --features_config "$config_file" \
      --seed $seed \
      > >(tee "$out_log") \
      2> >(tee "$err_log" >&2)
    echo "Experiment for $sut with $algorithm Completed"
    done

    echo "--- Running experiments for $sut with rs ---"

    ts=$(date +%Y%m%d-%H%M%S)
    out_log="$LOG_DIR/${sut}rs_out$ts.log"
    err_log="$LOG_DIR/${sut}rs_err$ts.log"

    python $script \
      --sut "$sut" \
      --algorithm "rs" \
      --population_size 2000 \
      --max_time $budget \
      --features_config "$config_file" \
      --seed $seed \
      > >(tee "$out_log") \
      2> >(tee "$err_log" >&2)
    echo "Experiment for $sut with rs Completed"

  done

done

  #PYTHONWARNINGS="error" 
  # python $script \
  #   --sut "$sut" \
  #   --algorithm "nsga2" \
  #   --population_size 10 \
  #   --n_generations 15 \
  #   --features_config "configs/isu_features.json" #\
  #   --max_time $budget \
  #   --no_wandb
