from typing import Tuple, List, Literal, Iterable
import numpy as np
from opensbt.evaluation.critical import Critical
from isu.simulation.sut_simulator import ISUSimulationOutput
from PIL import Image
import os
import json

class CriticalByFitnessThreshold(Critical):
    def __init__(self, 
                 var_name: str = "",
                 score: float = 1.0, 
                 mode: str = "<"):
        super().__init__()

        self.mode = mode
        self.score = score
        self.var_name = var_name

    def name(self) -> str:
        return f"{self.var_name} {self.mode} {self.score}"

    def eval(self, vector_fitness: np.ndarray, simout: ISUSimulationOutput) -> bool:
        value = abs(vector_fitness[0])

        if self.mode == "<":
            print( f"Evaluating Critical: {value} < {self.score} is {value < self.score}" )
            return value < self.score
        elif self.mode == "<=":
            return value <= self.score
        elif self.mode == ">":
            return value > self.score
        elif self.mode == ">=":
            return value >= self.score
        elif self.mode == "==":
            return value == self.score
        elif self.mode == "!=":
            return value != self.score
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose from: <, <=, >, >=, ==, !=")


class CriticalMerged(Critical):
    def __init__(self,
                 fitness_names: Iterable[str],
                 criticals: List[Tuple[Critical, List[str]]],
                 mode: Literal["and", "or"] = "or",
                 critical_save_folder: str=None):
        """
        Apply multiple critical functions with shared or distinct fitness inputs.

        :param fitness_names: All dimension names of the fitness vector.
        :param criticals: List of tuples (critical, required_fitness_names).
        :param mode: "or" → True if any critical is triggered, "and" → all must be triggered.
        """
        if mode not in {"or", "and"}:
            raise ValueError(f"Invalid mode: {mode}. Use 'or' or 'and'.")

        self.names_dict = {name: i for i, name in enumerate(fitness_names)}
        self.criticals = criticals
        self.mode = mode
        self.critical_save_folder = critical_save_folder
        if self.critical_save_folder:
            os.makedirs(self.critical_save_folder, exist_ok=True)

    def name(self) -> str:
        crit_names = []
        for critical, fitness_names in self.criticals:
            feat_str = ", ".join(fitness_names) if fitness_names else "no fitness names"
            crit_names.append(f"{feat_str} {critical.name()}")
        return f"CriticalMerged[{self.mode.upper()}]({'; '.join(crit_names)})"

    def eval(self, vector_fitness: np.ndarray, simout: ISUSimulationOutput) -> bool:
        results = []
        for critical, fitness_names in self.criticals:
            indices = [self.names_dict[name] for name in fitness_names]
            subvector = vector_fitness[indices]
            result = critical.eval(subvector, simout)
            print(f"Evaluating {critical.name()} with fitness values {subvector} → {result}")

            # Save critical images and their information
            if result:
                # critical_image_sim = Image.open(simout.scenario.image_path_sim)
                # path_sim = self.critical_save_folder + f"/{os.path.basename(simout.scenario.image_path_sim)}"
                # critical_image_sim.save(path_sim)

                # critical_image = Image.open(simout.scenario.image_path_sim2real)
                # path = self.critical_save_folder + f"/{os.path.basename(simout.scenario.image_path_sim2real)}"
                # critical_image.save(path)

                critical_image = Image.open(simout.scenario.image_path_sim)
                path = self.critical_save_folder + f"/{os.path.basename(simout.scenario.image_path_sim)}"
                critical_image.save(path)

                # Save wrong predictions
                wrong_predicts = CriticalMerged.compare_dict(simout.scenario.predicts_dict, simout.scenario.params_dict)
                if wrong_predicts:
                    diffs = [
                        {"key": k, "predicted": v[0], "actual": v[1]}
                        for k, v in wrong_predicts.items()
                    ]
                    with open(path.replace(".png", "_wrong.json"), "w") as f:
                        f.write(json.dumps(diffs, ensure_ascii=False, indent=2))
                    
                    # with open(path.replace(".png", "_full.json"), "w") as f:
                    #     f.write(json.dumps(simout.scenario.params_dict, ensure_ascii=False, indent=2))


                # log simulation details
                with open(path.replace(".png", "_sim_log.txt"), "w") as f:
                    f.write(f"Raw Result:\n{simout.raw_result}\n")
                    f.write(f"Last_error:\n{simout.last_error}\n")
                    f.write("Predicts:\n")
                    for key, value in simout.scenario.predicts_dict.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("Params:\n")
                    for key, value in simout.scenario.params_dict.items():
                        f.write(f"  {key}: {value}\n")

            results.append(result)

        return any(results) if self.mode == "or" else all(results)
    
    @staticmethod
    def compare_dict(predicts_dict, params_dict):
        wrong_predicts = {}

        for k, v in predicts_dict.items():   
            if k in params_dict and str(v) != str(params_dict[k]):
                #print(f"Wrong prediction for {k}: predicted={v}, actual={params_dict[k]}")
                # to relax bmw_isu suitcase location matching
                if not (k == "suitcase_location" and params_dict[k] in [x.strip() for x in v.split(",")]):
                    wrong_predicts[k] = (v, params_dict[k])  # (predicted, actual)

        return wrong_predicts