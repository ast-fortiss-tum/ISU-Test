from isu.model.run_blender import run_blender

json_path = "test_outputs/20251125_161144.json"
image_path_sim = run_blender(json_path=json_path, image_output_dir="test_outputs")