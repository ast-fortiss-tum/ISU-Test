import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

mode = "budget"  # "time" or "budget"

def get_color(label):
    for prefix, color in color_map.items():
        if label.startswith(prefix):
            return color
    return "black"  # fallback

color_map = {
    "gpt5": "tab:purple",
    "isu-bmw": "tab:red",
    "strict-isu-bmw": "tab:orange",
    "gemini-25": "tab:green",
}

res_files_folder = "isu/results"
nsga2_files = sorted(glob.glob(f"{res_files_folder}/**/nsga2/failures_over_time.csv")) 
#nsga2d_files = sorted(glob.glob(f"{res_files_folder}/**/nsga2d/failures_over_time.csv"))
rs_files = sorted(glob.glob(f"{res_files_folder}/**/rs/failures_over_time.csv"))

plt.figure(figsize=(10,6))

for fp in nsga2_files:
    df = pd.read_csv(fp)
    # if fp has "2.0" in its path, label it as "gemini-20", similarly for "2.5"
    if "2.0" in fp:
        label = "gemini-20"
    elif "2.5" in fp:
        label = "gemini-25"
    else:
        label = Path(fp).parent.parent.stem  # filename (without .csv) becomes legend label
        label = label.split("_")[0]  
    # if label == "moondream":
    #     continue
    if mode == "time":
        plt.plot(df["Time_s"], df["FailuresFound"], color=get_color(label), marker="", linestyle="-", linewidth=2, label="NSGA2-"+label)
    else:
        plt.plot(df["Budget_%"], df["FailuresFound"], color=get_color(label), marker="", linestyle="-", linewidth=2, label="NSGA2-"+label)

# for fp in nsga2d_files:
#     df = pd.read_csv(fp)
#     if "2.0" in fp:
#         label = "gemini-20"
#     elif "2.5" in fp:
#         label = "gemini-25"
#     else:
#         label = Path(fp).parent.parent.stem  # filename (without .csv) becomes legend label
#         label = label.split("_")[0]  
#     # if label == "moondream":
#     #     continue
#     if mode == "time":
#         plt.plot(df["Time_s"], df["FailuresFound"], color=get_color(label), marker="", linestyle="--", linewidth=2, label="NSGA2D-"+label)
#     else:
#         plt.plot(df["Budget_%"], df["FailuresFound"], color=get_color(label), marker="", linestyle="--", linewidth=2, label="NSGA2D-"+label)

for fp in rs_files:
    df = pd.read_csv(fp)
    if "2.0" in fp:
        label = "gemini-20"
    elif "2.5" in fp:
        label = "gemini-25"
    else:
        label = Path(fp).parent.parent.stem  # filename (without .csv) becomes legend label
        label = label.split("_")[0]  
    # if label == "moondream":
    #     continue
    if mode == "time":
        plt.plot(df["Time_s"], df["FailuresFound"], color=get_color(label), marker="", linestyle=":", linewidth=2, label="Random-"+label)
    else:
        plt.plot(df["Budget_%"], df["FailuresFound"], color=get_color(label), marker="", linestyle=":", linewidth=2, label="Random-"+label)

if mode == "time":
    plt.xlabel("Search Time Used (s)")
else:
    plt.xlabel("Search Budget Used (%)")
plt.ylabel("Cumulative Failures Found")
plt.title("Failure Discovery")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
#plt.show()
path = Path(res_files_folder) / "failure_discovery_plot.png"
plt.savefig(path)
