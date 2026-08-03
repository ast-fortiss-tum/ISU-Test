import io
import time
import os
import json
from typing import Callable, Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
from pymoo.core.algorithm import Algorithm

import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.wandb_run import Run
from wandb.apis.public import Api
import re

def wandb_log_csv(filename):
    # log only if file is not empty
    if (
        Path(filename).exists()
        and Path(filename).is_file()
        and os.stat(filename).st_size > 0
    ):
        try:
            df = pd.read_csv(filename)
            metric_table = wandb.Table(dataframe=df)
            metric_table_artifact = wandb.Artifact("metric_history", type="dataset")
            metric_table_artifact.add(metric_table, "metric_table")
            metric_table_artifact.add_file(filename)
            wandb.log({"log": metric_table})
            wandb.log_artifact(metric_table_artifact, name=filename)
        except io.UnsupportedOperation:
            print(f"Cannot log {filename}. Check if it is in .csv format.")
    else:
        print(f"{filename} does not exist or it is empty.")



def logging_callback(algorithm: Algorithm):
    all_population = algorithm.pop
    critical_all, _ = all_population.divide_critical_non_critical()
    wandb.log(
        {
            "population_size": len(all_population),
            "failures": len(critical_all),
            "critical_ratio": len(critical_all) / len(all_population),
            "timestamp": time.time()
        }
    )

def logging_callback_archive(algorithm: Algorithm):
    if hasattr(algorithm, "archive") and algorithm.archive is not None:
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        wandb.log(
            {
                "test_size": len(all_population),
                "failures": len(critical_all),
                "critical_ratio": len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
                "timestamp": time.time()
            }
        )

class TableCallback:
    def __init__(self):
        self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")

    def log(self, algorithm: Algorithm):
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        self.table.add_data(
            len(all_population),
            len(critical_all),
            len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
            time.time()
        )
        wandb.log({"Summary Table": self.table})

    def __getstate__(self):
        """Return state for pickling (skip unpicklable parts)."""
        state = self.__dict__.copy()
        state["table"] = None
        return state

    def __setstate__(self, state):
        """Recreate skipped attributes after unpickling."""
        self.__dict__.update(state)
        if self.table is None:
            self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")
            

def get_sut(project, run, tags):
    if project == "SafeLLM":
        return tags.get("sut", "unknown sut")
    elif project == "OpenSBT-ISU":
        return tags.get("sut", "unknown sut")
    else:
        # Regex to match everything up to "_<number>n"
        match = re.match(r"^(.*)_\d+n", run.name)
        if match:
            return match.group(1)
        else:
            raise Exception("SUT name could not be extracted from run name")
        
def download_run_artifacts(path: str, 
                           filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None) -> Dict[str, Dict[str, List[str]]]:
    runs = Api().runs(
        path,
        per_page = 1000
    )
    project = path.split("/")[-1]
    if filter_runs is not None:
        runs = filter_runs(runs)

    all_paths = defaultdict(lambda: defaultdict(list))
    missing_runs = []
    for run in runs:
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        print("run:", run.name, "tags:", tags)  
        local_path = os.path.join(
            "results",
            "artifacts",
            path,
            tags.get("sut", "unknown sut"),
            tags.get("algorithm", "unknown algo"),
            tags.get("seed", "no seed"),
            run.name,
            run.id,
        )
        os.makedirs(local_path, exist_ok=True)
        if not os.path.exists(os.path.join(local_path, "run_history.csv")):
            run.history().to_csv(os.path.join(local_path, "run_history.csv"), index=False)
        all_paths[get_sut(project, run, tags)][(tags.get("algorithm", "unknown algo"), tags.get("features", "default"))].append(local_path)
        if not os.path.exists(os.path.join(local_path, "config.txt")):
            missing_runs.append((run, local_path))
    print(f"missing_runs: {missing_runs}")
    for run, path in tqdm.tqdm(missing_runs):
        artifacts = run.logged_artifacts()
        # added tags
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        if len(artifacts) > 5:
            previous = artifacts[0]
            for a in artifacts:
                if a.type == "output":
                    a.download(path)
                    previous.download(path)
                    break
                else:
                    previous = a
            with open(os.path.join(path, "tags.json"), "w") as f:
                json.dump(tags, f)
            run.history().to_csv(os.path.join(local_path, "run_history.csv"), index=False)
    return all_paths

def download_run_artifacts_relative(
    wb_project_path: str,
    local_root: str = "./opentest",
    filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None,
    one_per_name: bool = False
) -> Dict[str, str]:
    """
    Download W&B run artifacts.
    
    Parameters
    ----------
    wb_project_path : str
        The "entity/project" path in W&B.
    local_root : str, default="./opentest"
        Local root directory for downloaded artifacts.
    filter_runs : callable, optional
        Function to filter runs before download.
    one_per_name : bool, default=False
        If True, only one run (latest) per unique run.name is downloaded.
        If False, all runs are downloaded.
    """
    api = Api()
    runs = api.runs(wb_project_path, per_page=1000)
    project_name = wb_project_path.split("/")[-1]

    if filter_runs is not None:
        runs = filter_runs(runs)

    runs = list(runs)

    print(f"Found {len(runs)} runs before grouping.")

    # Optionally group by run.name
    if one_per_name:
        grouped = defaultdict(list)
        for run in runs:
            grouped[run.name].append(run)
        selected_runs = [max(group, key=lambda r: r.created_at) for group in grouped.values()]
        print(f"Reduced to {len(selected_runs)} unique run names.")
    else:
        selected_runs = runs
        print(f"Downloading all {len(selected_runs)} runs.")

    all_paths = defaultdict(lambda: defaultdict(list))
    
    for run in tqdm.tqdm(selected_runs):
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        sut_name = get_sut(project_name, run, tags)

        local_path = os.path.join(
            local_root,
            project_name,
            "artifacts",
            sut_name,
            tags.get("algorithm", "unknown_algo"),
            tags.get("seed", "no_seed"),
            run.name,
            run.id
        )
        os.makedirs(local_path, exist_ok=True)

        all_paths[sut_name][(tags.get("algorithm", "unknown_algo"), 
                             tags.get("features", "default"))].append(local_path)

        # Skip if directory already contains files
        if any(os.scandir(local_path)):
            continue

        # Download all logged artifacts
        for artifact in run.logged_artifacts():
            artifact.download(local_path)

        # Save tags metadata
        with open(os.path.join(local_path, "tags.json"), "w") as f:
            json.dump(tags, f)

        # Save full run history
        history_df = run.history(keys=None, pandas=True)
        history_df.to_csv(os.path.join(local_path, "run_history.csv"), index=False)

    print(f"Downloaded {len(all_paths)} run directories.")
    print(f"all_paths:", all_paths)
    return all_paths

def get_summary(artifact_directory_path: str):
    summary_table = pd.read_csv(os.path.join(artifact_directory_path, "summary_results.csv"))
    summary = {
        attr: value for attr, value in
        zip(summary_table.Attribute, summary_table.Value)
    }
    return summary


def get_run_table(artifact_directory_path: str, freq: str = "1min", interpolate_duplicates: bool = True):
    if os.path.exists(os.path.join(artifact_directory_path, "Summary Table.table.json")):
        with open(os.path.join(artifact_directory_path, "Summary Table.table.json")) as f:
            table_dict = json.load(f)
            df = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
    else:
        df = pd.read_csv(os.path.join(artifact_directory_path, "run_history.csv"))
        df = df[["failures", "test_size", "critical_ratio", "timestamp"]].dropna()
    summary = get_summary(artifact_directory_path)
    if interpolate_duplicates:
        no_duplicates_ratio = float(summary["Number Critical Scenarios (duplicate free)"]) \
            / float(summary["Number Critical Scenarios"])
        df["failures"] = df.failures * no_duplicates_ratio
        df["critical_ratio"] = df["critical_ratio"] * no_duplicates_ratio
    df["time"] = df.timestamp.mul(1e9).apply(pd.Timestamp)
    df.time = df.time - df.time.iloc[0]
    df = df.set_index("time")
    df = pd.concat([df, df.resample(freq).asfreq()]).sort_index().interpolate("index").resample(freq).first()
    return df
