import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isu.simulation.sut_simulator import ISUSimulator, feature_weight_map
import json
from PIL import Image
from isu.eval.critical import CriticalMerged
import os

# sut = "gemini-2.5"
# image_path = "/Users/id/Library/CloudStorage/OneDrive-BMWGroup/OpenSBT-ISU-Integration/isu/results_s/strict-isu-bmw_2000n_1i_03_00_00t_RS_configs_isu_features/rs/critical_images/20251119_045509_sim.png"
# predicts_dict, raw_result, last_error = ISUSimulator.evaluate_image(image_path, sut=sut)
# print("Raw result:", raw_result)
# print("Predictions:", predicts_dict)

def fitness_correct_predicts(predicts_dict, params_dict):
        correct_predicts = 0
        for k, v in predicts_dict.items():   
            if k in params_dict:
                if str(v) == str(params_dict[k]):
                    correct_predicts += feature_weight_map[k]
                elif k == "suitcase_location" and params_dict[k] in [x.strip() for x in v.split(",")]:
                    print(f"suitcase location matched: pred {v}, actual {params_dict[k]}")
                    correct_predicts += feature_weight_map[k]
        correct_predicts = round(correct_predicts, 2)
        print(f"Total correct predicts score: {correct_predicts}")
        return correct_predicts

sut = "gemini-2.5"
BASE_PATH = "/Users/id/Documents/OpenSBT-ISU-Integration/isu/results/gemini-2.5_2000n_1i_03_00_00t_RS_configs_isu_features_seed6/rs"
threshold = 1.0
image_folder = BASE_PATH + "/images"

for image_path in Path(image_folder).glob("*.png"):
    # load params
    params_file = Path(BASE_PATH) / "json_inputs" / (image_path.stem.replace("_sim", "") + ".json")
    params_dict = json.load(open(params_file, "r"))

    # load or evaluate predicts
    target_folder = "output_predictions"
    predicts_dict_file = Path(BASE_PATH) / target_folder / (image_path.stem + ".txt")
    # if not predicts_dict_file.exists():
    #     print("Evaluating image:", image_path)
    #     predicts_dict, raw_result, last_error = ISUSimulator.evaluate_image(str(image_path), sut=sut)
    #     ISUSimulator.log_simulation_output(raw_result, image_path, tartget_folder=target_folder)
    # else:
    print("Using cached predictions for image:", image_path)
    with open(predicts_dict_file, "r", encoding="utf-8") as f:
        text = f.read()
    clean = text.replace("```json", "").replace("```", "").strip()
    predicts_dict = json.loads(clean)

    # evaluate correctness
    correct_predicts = fitness_correct_predicts(predicts_dict, params_dict)
    if_critical = correct_predicts < threshold

    if if_critical:
        critical_save_folder = Path(image_path).parent.parent / f"critical_images_threshold_{threshold}"
        critical_save_folder.mkdir(parents=True, exist_ok=True)

        critical_image = Image.open(image_path)
        path = critical_save_folder / os.path.basename(image_path)
        critical_image.save(path)

        # Save wrong predictions
        wrong_predicts = CriticalMerged.compare_dict(predicts_dict, params_dict)
        if wrong_predicts:
            diffs = [
                {"key": k, "predicted": v[0], "actual": v[1]}
                for k, v in wrong_predicts.items()
            ]
            with open(str(path).replace(".png", "_wrong.json"), "w") as f:
                f.write(json.dumps(diffs, ensure_ascii=False, indent=2))
            