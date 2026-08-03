from opensbt.utils.wandb import download_run_artifacts, download_run_artifacts_relative, get_run_table, get_summary
from opensbt.visualization.utils import AnalysisPlots, AnalysisException, AnalysisTables, AnalysisDiversity
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from collections import defaultdict
import pandas as pd
from typing import List, Tuple, Literal, Optional
import pickle
import tqdm

# from llm.utils.embeddings_local import get_embedding as get_embedding_local
# from llm.utils.embeddings_openai import get_embedding as get_embedding_openai
from isu.model.models import Scenario

def filter_isu(runs):
    res = []
    #take = False
    for run in runs:
        # if run.id == "wqpwm2wj":
        #     take = True
        #if take and run.state == "finished":
        if run.state == "finished":
            res.append(run)
    return res


run_filters = {"OpenSBT-ISU": filter_isu}


def convert_name(run_name: str) -> str:
    """
    Convert a run name into a standardized SUT string by scanning each
    word (separated by '_') in order and including all matching identifiers.
    """
    sut_keywords = {
        "ipa": "",
        "yelp": "",
        "gpt-4o": "GPT-4o",
        "gpt-5-chat": "GPT-5-Chat",
        "deepseek-v3-0324": "DeepSeek-V3",
        "chatbmw": "",
        "mistral": "Mistral-7B",
        "qwen3" : "Qwen3-8B",
        "deepseek-v2" : "DeepSeek-V2-16B"
    }

    words = run_name.split("_")
    sut_parts = []
    for w in words:
        w_lower = w.lower()
        if w_lower in sut_keywords:
            print(w_lower)
            kwd = sut_keywords[w_lower]
            if kwd != "":
                sut_parts.append(kwd)
    
    return "_".join(sut_parts) if sut_parts else "unknown_sut"


metric_names = {
    "failures": "Number of Failures",
    "critical_ratio": "Critical Ratio",
}

algo_names_map = {
    "rs": "Random",
    "nsga2": "NSGA2"
}

algorithms = [
    "NSGA2",
    "Random"
]


def get_algo_name(algo, features):
    algo_name = algo_names_map.get((algo, features), None)
    if algo_name is None:
        algo_name = algo_names_map[algo]
    return algo_name


def plot_metric_vs_time(
    project="OpenSBT-ISU",
    size=(18, 6),
    metric="failures",
    time_in_minutes=120,
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    fig, axes = plt.subplots(1, len(artifact_paths), sharey="row", sharex="all")
    fig.set_size_inches(*size)
    fig.supxlabel("Time, min")
    fig.supylabel(metric_names[metric])
    tick_count = time_in_minutes // 30 + 1
    ticks_kwargs = {"xticks": np.linspace(0, time_in_minutes, tick_count)}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
    plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
            AnalysisPlots.plot_with_std(axes[i], dfs, label=algo_name, metric=metric, target_time=time_in_minutes)
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.85) 
    legend_handles = {}
    for label, col in AnalysisPlots.label_colors.items():
        legend_handles[label] = plt.Line2D([0], [0], color=col, lw=10)
    # fig.legend(legend_handles.values(), legend_handles.keys(), title="Labels", loc="upper right")
    plt.tight_layout()
    plt.savefig(file_name, format="pdf")


def plot_metric_vs_time_separate(
    project="SafeLLM",
    size=(12, 6),
    metric="failures",
    time_in_minutes=120,
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    fig = plt.figure()
    fig.set_size_inches(size[0], size[1])
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
            AnalysisPlots.plot_with_std(plt, dfs, label=algo_name, metric=metric, target_time=time_in_minutes)
        plt.title(convert_name(sut.capitalize()))
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.85) 
        plt.tight_layout()
        tick_count = time_in_minutes // 30 + 1
        plt.xticks(np.linspace(0, time_in_minutes, tick_count))
        plt.xlim((0, time_in_minutes))
        plt.xlabel("Time [min]")
        if metric.lower() == "critical_ratio":
            plt.ylim(0, 1)
        plt.ylabel(metric_names[metric])
        plt.tight_layout()
        path = file_name + f"/{convert_name(sut.capitalize())}.pdf"
        os.makedirs(file_name, exist_ok=True)
        plt.savefig(path, format="pdf")
        plt.cla()

def boxplots(
    project="SafeLLM",
    size=(18, 6),
    metric="failures",
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    if metric == "critical_ratio":
        fig, axes = plt.subplots(1, len(artifact_paths), sharey="row")
    else:
        fig, axes = plt.subplots(1, len(artifact_paths))
    fig.set_size_inches(*size)
    # fig.supylabel(metric_names[metric])
    # Ensure axes is always a list
    if len(artifact_paths) == 1:
        axes = [axes]
    ticks_kwargs = {}
    if metric == "critical_ratio":
        ticks_kwargs["yticks"] = np.linspace(0.0, 1.0, 6)
        ticks_kwargs["ylim"] = (0.0, 1.0)
        plt.setp(axes, **ticks_kwargs)
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        name_to_dfs = {}
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
            name_to_dfs[algo_name] = dfs
        algo_names = []
        dfs_list = []
        for algorithm in algorithms:
            if algorithm in name_to_dfs:
                algo_names.append(algorithm)
                dfs_list.append(name_to_dfs[algorithm])
        AnalysisPlots.boxplot(axes[i], dfs_list, algo_names, metric=metric)
        axes[i].set_title(convert_name(sut.capitalize()))
        axes[i].set_box_aspect(1)
        axes[i].set_xticklabels(algo_names, rotation='vertical')
    
    # Put y-label only next to first subplot
    axes[0].set_ylabel(metric_names[metric], labelpad=10, fontsize = 20)

    legend_handles = {}
    for label, col in AnalysisPlots.label_colors.items():
        legend_handles[label] = plt.Line2D([0], [0], color=col, lw=10)
    
    # fig.legend(legend_handles.values(), legend_handles.keys(), title="Labels", loc="upper right")
    fig.tight_layout()
    plt.savefig(file_name, format="pdf")


def boxplots_separate(
    project="SafeLLM",
    size=(12, 6),
    metric="failures",
    file_name="plot",
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    fig = plt.figure()
    fig.set_size_inches(*size)
    # fig.supylabel(metric_names[metric])
    # Ensure axes is always a list
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        name_to_dfs = {}
        for (algo, features), paths in algos.items():
            algo_name = get_algo_name(algo, features)
            dfs = [get_run_table(path) for path in paths]
            name_to_dfs[algo_name] = dfs
        algo_names = []
        dfs_list = []
        for algorithm in algorithms:
            if algorithm in name_to_dfs:
                algo_names.append(algorithm)
                dfs_list.append(name_to_dfs[algorithm])
        AnalysisPlots.boxplot(plt, dfs_list, algo_names, metric=metric)
        plt.title(convert_name(sut.capitalize()))
        plt.xticks(ticks=np.arange(len(algo_names)) + 1, labels=algo_names, rotation='vertical')
        if metric.lower() == "critical_ratio":
            plt.ylim(0, 1)
        plt.ylabel(metric_names[metric])
        plt.tight_layout()
        path = file_name + f"/{convert_name(sut.capitalize())}.pdf"
        os.makedirs(file_name, exist_ok=True)
        plt.savefig(path, format="pdf")
        plt.cla()


def statistics_table(
    project="OpenSBT-ISU",
    metric="failures",
    path="table.csv"
):
    if project not in run_filters:
        raise AnalysisException(
            "Plesase implement runs filter for yout project in opensbt.visualization.llm_figures"
        )
    artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])
    statistics = {}
    suts = []
    for i, (sut, algos) in enumerate(artifact_paths.items()):
        suts.append(sut)
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        if metric == "failures":
            values = [
                [float(get_summary(path)["Number Critical Scenarios (duplicate free)"]) for path in paths] for paths in algos.values()
            ]
        elif metric == "critical_ratio":
            values = [
                [float(get_summary(path)["Number Critical Scenarios (duplicate free)"]) / float(get_summary(path)["Number All Scenarios"]) for path in paths] for paths in algos.values()
            ]
        else:
            raise AnalysisException("Unknown metric")
        statistics[sut] = AnalysisTables.statistics(values, algo_names)
    data = defaultdict(list)
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            data["Algorithm 1"].append(algorithms[i])
            data["Algorithm 2"].append(algorithms[j])
            for sut in suts:
                stats = statistics[sut][algorithms[i]][algorithms[j]]
                if len(stats) > 0:
                    data[f"{sut.capitalize()}.P-Value"].append(stats[0])
                    data[f"{sut.capitalize()}.Effect Size"].append(stats[1])
                else:
                    data[f"{sut.capitalize()}.P-Value"].append(None)
                    data[f"{sut.capitalize()}.Effect Size"].append(None)
    df = pd.DataFrame(data)
    # df = df.dropna()
    df.to_csv(path, index=False)

def last_values_table(
    project: str = "SafeLLM",
    metrics: list | str = "failures",
    path: str = "table.csv",
):
    save_path = path

    if run_filters is not None and project not in run_filters:
        raise AnalysisException(
            "Please implement runs filter for your project in opensbt.visualization.llm_figures"
        )

    artifact_paths = download_run_artifacts(f"opentest/{project}", run_filters[project])

    if isinstance(metrics, str):
        metrics = [metrics]

    summary_data = []

    for sut, algos in artifact_paths.items():
        algo_names = [get_algo_name(*key) for key in algos.keys()]
        print("algos:", algo_names)

        for algo_name, paths in zip(algo_names, algos.values()):
            row = {"Algorithm": algo_name,"SUT": sut}

            for metric in metrics:
                if metric == "failures":
                    values = [int(get_summary(path)["Number Critical Scenarios (duplicate free)"]) for path in paths]
                elif metric == "critical_ratio":
                    values = []
                    for path in paths:
                        ratio = float(get_summary(path)["Number Critical Scenarios (duplicate free)"]) / float(get_summary(path)["Number All Scenarios"])
                        values.append(ratio)
                else:
                    raise AnalysisException(f"Unknown metric: {metric}")

                row[f"{metric}_mean"] = np.mean(values) if values else np.nan
                row[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0

            summary_data.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    algorithm_order = ["NSGAII", "Random", "T-wise", "ASTRAL"]
    summary_df = pd.DataFrame(summary_data)
    summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=algorithm_order, ordered=True)
    summary_df = summary_df.sort_values(by=["SUT", "Algorithm"]).reset_index(drop=True)

    summary_df.to_csv(save_path, index=False)
    print(f"\nSummary table of mean/std saved to: {save_path}")

    # Determine LaTeX column alignment dynamically
    n_metrics = len(metrics)
    col_format = "ll" + "cc" * n_metrics  # 2 left + 2 per metric (mean/std)
    
    latex_path = os.path.splitext(save_path)[0] + ".tex"
    summary_df.to_latex(
        latex_path,
        index=False,
        float_format="%.3f",
        caption=f"Summary of metrics per SUT and algorithm",
        label="tab:metric_summary",
        column_format=col_format
    )