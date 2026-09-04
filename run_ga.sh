python scripts/generate_isu_render_and_canny.py \
      --population-size 2 \
      --n-generations 2 \
      --scene isu/blender/scenes/scene_v3.blend \
      --features-config configs/isu_challenge_features.json \
      --output-dir results/scene_v3_ga/ \
      --no-wandb \
      --sut dummy