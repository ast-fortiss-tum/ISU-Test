import os
import json
import numpy as np
from pathlib import Path

RESULTS_FOLDER = "/Users/id/Library/CloudStorage/OneDrive-BMWGroup/OpenSBT-ISU-Integration/isu/results/isu_bmw_20n_30i_03_00_00t_NSGA2_configs_isu_features/nsga2"

def calculate_failures_found(folder):
    folder_critical_images = os.path.join(folder, "critical_images")
    folder_all_evaluations = os.path.join(folder, "output_predictions")
    folder_all_instances = os.path.join(folder, "images")

    failure_count = 0
    total_instances = 0
    total_evaluations = 0

    # failure_count_duplicatefree = 0

    # failureset = set()
    # for filename in os.listdir(folder_critical_images):
    #     if filename.endswith("full.json"):
    #         path = os.path.join(folder_critical_images, filename)
    #         with open(path, "r") as f:
    #             data = json.load(f)[:15]
    #         for entry in data:
    #             if entry in failureset:
    #                 continue
    #             failureset.add(entry)
    #             failure_count_duplicatefree += 1
            

    for filename in os.listdir(folder_critical_images):
        if filename.endswith("wrong.json"):
            failure_count += 1
    for filename in os.listdir(folder_all_evaluations):
        if filename.endswith(".txt"):
            total_evaluations += 1
    for filename in os.listdir(folder_all_instances):
        if filename.endswith(".png"):
            total_instances += 1
    ratio = failure_count / total_evaluations 
    ratio = round(ratio, 2)

    output = {
        "failures_found": failure_count,
        "total_evaluations": total_evaluations,
        "failure_ratio": ratio,
        "total_instances": total_instances
    }

    path = os.path.join(folder, "failure_summary.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=4)
    return failure_count, total_evaluations, ratio, total_instances

if __name__ == "__main__":
    failures, total_evaluations, ratio, total_instances = calculate_failures_found(RESULTS_FOLDER)
    print(f"Failures found: {failures}")
    print(f"Total instances: {total_instances}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Failure ratio: {ratio:.2f}")

