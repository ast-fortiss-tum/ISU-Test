import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isu.simulation.sut_simulator import ISUSimulator
import os
from isu.eval.critical import CriticalMerged
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# os.environ["PROJECT_ROOT"] = "/Users/id/Documents"
# PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

#string = "/Users/id/Library/CloudStorage/OneDrive-BMWGroup/OpenSBT-ISU-Integration/isu/results/gpt5-chat_5n_1i_RS_configs_isu_features/rs"
base_result_dir = PROJECT_ROOT / "data_val" 
image_folder = base_result_dir / "images"
gt_params_folder = base_result_dir / "json_inputs"

image_folder_reproduced = base_result_dir / "images_reproduced"
image_folder_sim2real = base_result_dir / "images_sim2real"
image_folder_sim = base_result_dir / "images_sim"

suts = ["isu-bmw", "gpt5-chat", "gemini-2.5"] 
domains = ["sim", "reproduced"] 

for sut in suts:

    for domain in domains:
        res_folder = base_result_dir / "wrong_predictions" / sut / domain
        res_folder.mkdir(parents=True, exist_ok=True)

        # for image_name in os.listdir(image_folder):
        #     if (image_name.endswith(f"_sim.png") and domain == "sim") or \
        #         (image_name.endswith(f"_sim2real.png") and domain == "sim2real") or \
        #         (image_name.endswith(f"_reproduced.png") and domain == "reproduced"):

        image_folder = image_folder_reproduced if domain == "reproduced" else \
                       image_folder_sim2real if domain == "sim2real" else \
                       image_folder_sim if domain == "sim" else None

        for image_name in os.listdir(image_folder):
                print(f"Processing {image_name} for {sut} in {domain}, image_folder: {image_folder}")

                image_path = os.path.join(image_folder, image_name)
                if not image_path.endswith(".png"):
                    continue
                predicts_dict, raw_result, last_error = ISUSimulator.evaluate_image(image_path, sut=sut)

                if sut == "isu-bmw":
                    wait_time = 5
                    print(f"Waiting for {wait_time} seconds to let ISU process the image...")
                    import time
                    time.sleep(wait_time)

                print(f"Evaluated {image_name} for {sut} in {domain}")
                print(f"Predictions: {predicts_dict}")
                base_name = image_name.replace("_sim2real", "").replace("_sim", "").replace("_reproduced", "").replace(".png", "")

                gt_params_path = os.path.join(gt_params_folder, f"{base_name}.json")
                
                with open(gt_params_path, "r") as f:
                    params_dict = json.load(f)

                wrong_predicts = CriticalMerged.compare_dict(predicts_dict, params_dict)
                print(f"Wrong predictions: {wrong_predicts}")
                if wrong_predicts:

                    diffs = [
                        {"key": k, "predicted": v[0], "actual": v[1]}
                        for k, v in wrong_predicts.items()
                    ]

                    out_file = res_folder / image_name.replace(".png", "_wrong.json")
                    out_file.write_text(json.dumps(diffs, ensure_ascii=False, indent=2))
                    print(f"Wrote wrong predictions to {out_file}")


                    # pred = [
                    #     {"key": k, "predicted": v}
                    #     for k, v in predicts_dict.items()
                    # ]

                    # out_file = res_folder / image_name.replace(".png", "_pred.json")
                    # out_file.write_text(json.dumps(pred, ensure_ascii=False, indent=2))