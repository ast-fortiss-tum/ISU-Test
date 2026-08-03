from dataclasses import fields
#from llm.eval.diversity import cluster_utterances_vars
#from llm.model.qa_problem import QAProblem
from opensbt.model_ga.individual import IndividualSimulated
#from llm.feature_discretization import get_features
#from llm.model.models import LOS
from pymoo.core.result import Result
from opensbt.utils.duplicates import duplicate_free
from opensbt.model_ga.population import PopulationExtended
import csv
from itertools import product
from pydantic import BaseModel
import shutil
#from llm.utils.math import euclid_distance
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
import os
import json
from typing import List, Dict, Optional, Any
# from llm.model.models import Utterance, ContentInput
# from llm.model.qa_problem import QAProblem
# from llm.eval.utterances_distance import UtterancesDistance
from isu.model.problem import ISUProblem
from isu.model.models import Scenario
from isu.eval.similarity import ScenarioDistance 

def write_novelty_archive_to_json(res, save_folder: str, filename = "novelty_archive"):
    algorithm = res.algorithm
    
    class ArchiveTestReport(TestReport):
        size: int | None = None

    if hasattr(algorithm, "archive_novelty"):
        archive = algorithm.archive_novelty
        problem = res.problem
        all_scenarios: List[ArchiveTestReport] = []

        for idx, ind in enumerate(archive):
            scenario: Scenario = ind.get("X")[0]  # Assuming "X" holds vars and is a list of lists
            #poi_exists = ind.get("SO").poi_exists
            #other = ind.get("SO").other

            fitness_names = list(problem.fitness_function.name)
            scores_dict = {name: float(score) for name, score in zip(fitness_names, ind.get("F").tolist())}
            is_critical = bool(ind.get("CB"))

            if hasattr(problem, "feature_handler"):
                feature_handler = problem.feature_handler
                features_dict = feature_handler.get_feature_values_dict(
                    scenario.continuous_vars,
                    scenario.categorical_vars,
                )
            else:
                features_dict = {}

            all_scenarios.append(
                ArchiveTestReport(
                    size=len(all_scenarios),
                    scenario=scenario,
                    features_dict=features_dict,
                    fitness=scores_dict,
                    is_critical=is_critical,
                    #other = other,
                    #poi_exists=poi_exists,
                ).model_dump(serialize_as_any=True)
            )
            
        filename = os.path.join(save_folder, f"{filename}.json")

        # Write JSON file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_scenarios, f, ensure_ascii=False, indent=4)
            print(f"Novelty archive saved to {filename}")
    else:
        print("No novelty archive in algorithm class. ")
   

class SimoutReport(BaseModel):
    simout: Dict[str, Any] = dict()
    
def write_simout_to_json(res, save_folder: str, critical_only: bool = False, filename: Optional[str] = None):
    all_population = res.archive
    problem: ISUProblem = res.problem

    if critical_only:
        all_population, _ = all_population.divide_critical_non_critical()

    if filename is None:
        filename = "simout_critical" if critical_only else "simout"

    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    all_scenarios: List[SimoutReport] = []

    for idx, ind in enumerate(all_population):
        simout = ind.get("SO")
        all_scenarios.append(
            SimoutReport(
                simout = simout.to_dict()
            ).model_dump(serialize_as_any=True)
        )
    # Construct a filename: e.g., utterance_0001.json
    filename = os.path.join(save_folder, f"{filename}.json")

    # Write JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=4)

def write_tests_to_json(res, save_folder: str, critical_only: bool = False, filename: Optional[str] = None):
    all_population = res.archive
    problem: ISUProblem = res.problem
    if critical_only:
        all_population, _ = all_population.divide_critical_non_critical()

    if filename is None:
        filename = "critical_scenarios" if critical_only else "all_scenarios"

    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    all_scenarios: List[TestReport] = []

    for idx, ind in enumerate(all_population):
        scenario: Scenario = ind.get("X")[0]  # Assuming "X" holds vars and is a list of lists
        #poi_exists = ind.get("SO").poi_exists
        #other = ind.get("SO").other

        fitness_names = list(problem.fitness_function.name)
        scores_dict = {name: float(score) for name, score in zip(fitness_names, ind.get("F").tolist())}
        is_critical = bool(ind.get("CB"))

        if hasattr(problem, "feature_handler"):
            feature_handler = problem.feature_handler
            features_dict = feature_handler.get_feature_values_dict(
                scenario.continuous_vars,
                scenario.categorical_vars,
            )
        else:
            features_dict = {}

        all_scenarios.append(
            TestReport(
                scenario=scenario,
                features_dict=features_dict,
                fitness=scores_dict,
                is_critical=is_critical,
                #other = other,
                #poi_exists=poi_exists,
            ).model_dump(serialize_as_any=True)
        )
    # Construct a filename: e.g., utterance_0001.json
    filename = os.path.join(save_folder, f"{filename}.json")

    # Write JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=4)

def compute_pairwise_dissimilarity(scenarios, **kwargs):

    vars_dissimilarities = []

    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            distance = ScenarioDistance.calculate(
                scenarios[i].get("X")[0],
                scenarios[j].get("X")[0],
                **kwargs
            )

            vars_dissimilarities.append(distance.vars_distance)

    avg_vars_dissim = -1 if not vars_dissimilarities else np.mean(vars_dissimilarities)
    return avg_vars_dissim

def compute_average_max_dissimilarity(scenarios, **kwargs):

    max_var = -1

    # Compute all pairwise distances for this test
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            distance = ScenarioDistance.calculate(
                scenarios[i].get("X")[0],
                scenarios[j].get("X")[0],
                **kwargs
            )
            max_var = max(max_var, distance.vars_distance)


    max_vars = [max_var] if max_var >= 0 else []

    avg_max_vars = np.mean(max_vars) if max_vars else -1

    return avg_max_vars

class TestReport(BaseModel):
    scenario: Scenario
    features_dict: Dict[str, Any] = dict()
    fitness: Dict[str, float] = dict()
    is_critical: bool = False
    #poi_exists: bool = False
    #other: Dict = None

def write_tests_to_json(res, save_folder: str, critical_only: bool = False, filename: Optional[str] = None):
    all_population = res.archive
    problem: ISUProblem = res.problem
    if critical_only:
        all_population, _ = all_population.divide_critical_non_critical()

    if filename is None:
        filename = "critical_scenarios" if critical_only else "all_scenarios"

    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    all_scenarios: List[TestReport] = []

    for idx, ind in enumerate(all_population):
        scenario: Scenario = ind.get("X")[0]  # Assuming "X" holds vars and is a list of lists
        #poi_exists = ind.get("SO").poi_exists
        #other = ind.get("SO").other

        fitness_names = list(problem.fitness_function.name)
        scores_dict = {name: float(score) for name, score in zip(fitness_names, ind.get("F").tolist())}
        is_critical = bool(ind.get("CB"))

        if hasattr(problem, "feature_handler"):
            feature_handler = problem.feature_handler
            features_dict = feature_handler.get_feature_values_dict(
                scenario.ordinal_vars,
                scenario.categorical_vars,
            )
        else:
            features_dict = {}

        all_scenarios.append(
            TestReport(
                scenario=scenario,
                features_dict=features_dict,
                fitness=scores_dict,
                is_critical=is_critical,
                #other = other,
                #poi_exists=poi_exists,
            ).model_dump(serialize_as_any=True)
        )
    # Construct a filename: e.g., utterance_0001.json
    filename = os.path.join(save_folder, f"{filename}.json")

    # Write JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=4)

def calculate_diversity(res, save_folder, **kwargs):
    """
    Calculate and store diversity statistics:
    - Input: Question diversity (semantic dissimilarity based on question embeddings)
    - Output: Answer diversity (semantic dissimilarity based on answer embeddings / output diversity)
    - Input: Variable-level diversity
    For both critical and all scenarios.
    """
    all_population = res.obtain_all_population()
    critical, _ = all_population.divide_critical_non_critical()
    critical_clean = duplicate_free(critical)
    
    # --- Average pairwise dissimilarities ---
    avg_vars_critical = compute_pairwise_dissimilarity(critical_clean, **kwargs)
    avg_vars_all = compute_pairwise_dissimilarity(all_population, **kwargs)

    # --- Average max dissimilarities ---
    avg_max_vars_critical= compute_average_max_dissimilarity(critical_clean, **kwargs)
    avg_max_vars_all = compute_average_max_dissimilarity(all_population, **kwargs)

    # # --- Cluster-level diversity ---
    # cluster_critical = cluster_utterances_vars([ind.get("X")[0] for ind in critical_clean])
    # cluster_all = cluster_utterances_vars([ind.get("X")[0] for ind in all_population])

    # --- Write results to CSV ---
    with open(save_folder + 'diversity.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attribute', 'Value'])

        writer.writerow(['Average Dissimilarity Critical (Vars)', avg_vars_critical])        
        writer.writerow(['Average Dissimilarity All (Vars)', avg_vars_all])

        writer.writerow(['Average Max Dissimilarity Critical (Vars)', avg_max_vars_critical])         
        writer.writerow(['Average Max Dissimilarity All (Vars)', avg_max_vars_all])
        
        # # Cluster-based metrics
        # writer.writerow(['Critical Clusters: Best k', cluster_critical['best_k']])
        # writer.writerow(['Critical Clusters: Avg Medoid Distance', cluster_critical['avg_medoid_distance']])
        # writer.writerow(['Critical Clusters: Max Medoid Distance', cluster_critical['max_medoid_distance']])
        # writer.writerow(['Critical Clusters: Avg Max Medoid Distance', cluster_critical['avg_max_medoid_distance']])

        # writer.writerow(['All Clusters: Best k', cluster_all['best_k']])
        # writer.writerow(['All Clusters: Avg Medoid Distance', cluster_all['avg_medoid_distance']])
        # writer.writerow(['All Clusters: Max Medoid Distance', cluster_all['max_medoid_distance']])
        # writer.writerow(['All Clusters: Avg Max Medoid Distance', cluster_all['avg_max_medoid_distance']])

def write_failures_over_time(res, save_folder, interval = 100):
    """
    Compute cumulative failures over time from a pymoo result archive, preserving execution order.

    Parameters
    ----------
    res : pymoo Result
        Result object containing archive.
    save_folder : str
        Folder to save the interpolated CSV.
    total_search_time : int
        Total search time in seconds.
    interval : int
        Interpolation step in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with Time_s, Budget_%, FailuresFound.
    """
    os.makedirs(save_folder, exist_ok=True)

    total_search_time = res.exec_time

    all_population = res.archive  # Keep all individuals in order
    
    all_population = duplicate_free(all_population)

    # Store critical labels in order
    critical_labels = []
    for ind in all_population:
        # Assuming 'CB' is 1 for critical (failure) and 0 for non-critical
        critical_labels.append(ind.get("CB"))

    # Total number of individuals
    n_tests = len(all_population)
    time_per_test = total_search_time / n_tests

    # Discovery times
    discovery_times = [(i + 1) * time_per_test for i in range(n_tests)]

    # Cumulative failures
    cumulative_failures = np.cumsum(critical_labels)

    # Interpolation points
    time_points = np.arange(0, total_search_time + interval, interval)
    failures_over_time = np.interp(time_points, discovery_times, cumulative_failures)

    # Create result DataFrame
    result = pd.DataFrame({
        "Time_s": time_points,
        "Budget_%": (time_points / total_search_time) * 100,
        "FailuresFound": failures_over_time
    })

    # Save CSV
    csv_path = os.path.join(save_folder, "failures_over_time.csv")
    result.to_csv(csv_path, index=False)
    print(f"Interpolated failures saved to {csv_path}")

    return result

# def copy_prompts(save_folder, source_path = "./llm/prompts.py"):
#     destination_path = save_folder + "/prompts.txt"
#     shutil.copy2(source_path, destination_path)
#     return destination_path

def copy_config(save_folder, source_path = "./isu/config.py"):
    destination_path = save_folder + "/config.txt"
    shutil.copy2(source_path, destination_path)
    return destination_path