suts=("moondream") 
script="run_tests_isu.py"

echo "--- Running experiments for $sut with gs ---"
python $script \
  --sut "$sut" \
  --algorithm "gs" \
  --population_size 50 \
  --features_config "configs/isu_features_validation.json" \
  --no_wandb 
echo "Experiment for $sut with gs Completed"
