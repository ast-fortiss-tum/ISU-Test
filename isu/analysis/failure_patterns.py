import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

def print_failure_patterns(sut, algo, base_result_dir, folder_stem = "critical_images", target_path="failure_heatmap.png"):

    # For post evaluation after runs
    # if algo == "RS":
    #     base_result_dir = PROJECT_ROOT / "results" / f"{sut}_2000n_1i_{time}t_{algo}_configs_isu_features" / f"{algo.lower()}"
    # else:
    #     base_result_dir = PROJECT_ROOT / "results" / f"{sut}_20n_30i_{time}t_{algo}_configs_isu_features" / f"{algo.lower()}"
    folder = base_result_dir + os.sep + folder_stem

    # For validations
    #folder = PROJECT_ROOT / "data_val" / "wrong_predictions" / sut / domain

    failure_groups = defaultdict(int)
    all_failures = 0

    for filename in os.listdir(folder):
        if filename.endswith("wrong.json"):
            all_failures += 1
            with open(os.path.join(folder, filename), "r") as f:
                data = json.load(f)

            # Extract failures only
            failures = []
            for item in data:
                failures.append((item["key"], item["predicted"], item["actual"]))

            failure_signature = tuple(sorted([(key, pred, act) for key, pred, act in failures]))
            failure_groups[failure_signature] += 1

    result = {k: v / all_failures for k, v in failure_groups.items()}

    df = pd.DataFrame.from_dict(result, orient="index", columns=["percentage"])
    df["count"] = [failure_groups[k] for k in df.index]  # 添加原始计数列
    df = df.sort_values(by="percentage", ascending=False)

    annot_matrix = df.apply(lambda row: f"{row['percentage']:.1%} ({row['count']})", axis=1).values.reshape(-1, 1)

    df.index = [
        "\n".join(str(k) for k in pattern if str(k).lower() != "nan") 
        if len(pattern) > 0 
        else "(no failures)"
        for pattern in df.index
    ]

    plt.figure(figsize=(7, len(df)*1.5))
    # sns.heatmap(
    #     f"{df}",
    #     annot=True,
    #     cmap="Blues",
    #     #fmt=".02f",
    #     fmt=".1%",
    #     linewidths=.5,
    #     cbar=False,
    #     annot_kws={"fontsize": 14, "fontweight": "bold"}   # << bigger + bold numbers
    # )

    sns.heatmap(
        df[["percentage"]],  
        annot=annot_matrix,  
        cmap="Blues",
        fmt="",  
        linewidths=.5,
        cbar=False,
        annot_kws={"fontsize": 14, "fontweight": "bold"}
    )

    plt.title(f"Failures {sut} {algo}", fontsize=16, fontweight='bold')

    heatmap_outfile = base_result_dir + os.sep + target_path
    plt.tight_layout()
    # plt.show()
    plt.savefig(heatmap_outfile, dpi=200, bbox_inches="tight")

    # Feature-level heatmap
    # all_results = defaultdict(dict)   # {pattern_signature: {sut: count}} 
    # for sut in suts:
    #     # For post evaluation after runs
    #     if algo == "RS":
    #         base_result_dir = PROJECT_ROOT / "results" / f"{sut}_2000n_1i_02_00_00t_{algo}_configs_isu_features" / f"{algo.lower()}"
    #     else:
    #         base_result_dir = PROJECT_ROOT / "results" / f"{sut}_20n_30i_02_00_00t_{algo}_configs_isu_features" / f"{algo.lower()}"
    #     folder = base_result_dir / "critical_images"

    #     # For validations
    #     #folder = PROJECT_ROOT / "data_val" / "wrong_predictions" / sut / domain

    #     failure_groups = defaultdict(int)

    #     for filename in os.listdir(folder):
    #         if filename.endswith("wrong.json") or filename.endswith("wrong.txt") :
    #             with open(os.path.join(folder, filename), "r") as f:
    #                 data = json.load(f)

    #             # Extract failures only
    #             failures = []
    #             for item in data:
    #                 failures.append((item["key"]))

    #             # failure_signature = tuple(sorted(failures))
    #             failure_signature = tuple(sorted(failures))

    #             failure_groups[failure_signature] += 1

    #     # Store counts into global matrix
    #     for signature, count in failure_groups.items():
    #         all_results[signature][sut] = count


    # feature_counts = defaultdict(lambda: defaultdict(int))  # {feature: {sut: count}}
    # for signature, sut_counts in all_results.items():
    #     if not signature:
    #         continue
    #     for feat in signature:
    #         feat_str = str(feat).strip()
    #         if feat_str and feat_str.lower() not in ("nan", "none"):
    #             for sut, cnt in sut_counts.items():
    #                 feature_counts[feat_str][sut] += cnt

    # fdf = pd.DataFrame.from_dict(feature_counts, orient="index").fillna(0)

    # fdf = fdf.reindex(columns=suts).fillna(0)
    # fdf = fdf.reindex(feature_order).fillna(0)


    # plt.figure(figsize=(10, max(6, len(fdf) * 0.6)))
    # sns.heatmap(
    #     fdf,
    #     annot=True,
    #     cmap="Blues",
    #     fmt=".0f",
    #     linewidths=.5,
    #     cbar=False,
    #     annot_kws={"fontsize": 14, "fontweight": "bold"}
    # )

    # plt.xlabel("SUT", fontsize=16, fontweight='bold')
    # plt.ylabel("Feature", fontsize=16, fontweight='bold')
    # plt.title(f"Failure Features {algo.upper()}", fontsize=18, fontweight='bold')

    # plt.xticks(fontsize=14, fontweight='bold', rotation=0)
    # plt.yticks(fontsize=14, fontweight='bold')

    # feature_heatmap_outfile = PROJECT_ROOT / "results" / f"feailure_feature_heatmap_{sut}_{algo}_divided.png"
    # plt.tight_layout()
    # plt.savefig(feature_heatmap_outfile, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    print_failure_patterns("gemini-2.5", 
                           "rs", 
                           "isu/results/gemini-2.5_2000n_1i_00_15_00t_1seed_RS_configs_isu_features/rs")