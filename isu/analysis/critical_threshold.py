#read all_testcases.csv and filter on the fitness value on the first column,count how many are below a certain threshold
from typing import List

def critical_threshold_analysis(csv_path, threshold_list: List[float]) -> dict:
    total_cases = 0
    critical_cases_per_threshold = {threshold: 0 for threshold in threshold_list}
    with open(csv_path, "r") as f:
        lines = f.readlines()
        header = lines[0]
        for line in lines[1:]:
            total_cases += 1
            parts = line.strip().split(",")
            fitness_value = float(parts[1])
            for threshold in threshold_list:
                if fitness_value < threshold:
                    critical_cases_per_threshold[threshold] += 1
    critical_ratios = {threshold: critical_cases_per_threshold[threshold] / total_cases for threshold in threshold_list}
    #only take 2 decimal places
    critical_ratios = {k: round(v, 2) for k, v in critical_ratios.items()}
    return {
        "total_cases": total_cases,
        "critical_cases_per_threshold": critical_cases_per_threshold,
        "critical_ratios": critical_ratios
    }
if __name__ == "__main__":
    csv_path = "/path/to/your/all_testcases.csv"
    threshold_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    analysis_result = critical_threshold_analysis(csv_path, threshold_list)
    print("Total Cases:", analysis_result["total_cases"])
    print("Critical Cases per Threshold:", analysis_result["critical_cases_per_threshold"])
    #print("Critical Ratios:", analysis_result["critical_ratios"])
    for threshold, ratio in analysis_result["critical_ratios"].items():
        print(f"Threshold: {threshold}, Critical Ratio: {ratio}")