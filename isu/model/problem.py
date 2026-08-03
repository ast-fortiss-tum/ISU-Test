from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
import logging as log
from functools import partial
from isu.model.scenario_generation import ScenarioGenerator
from isu.features import FeatureHandler
import wandb
from llm.utils.seed import set_seed


@dataclass
class ISUProblem(Problem):
    def __init__(self,
        # xl: List[float],
        # xu: List[float],
        fitness_function: Fitness,
        simulate_function,
        critical_function: Critical,
        simulation_variables: Optional[List[str]] = None,
        design_names: List[str] = None,
        objective_names: List[str] = None,
        problem_name: str = None,
        context: str = None,
        feature_handler_config_path: str = None,
        scenario_generator: Optional[ScenarioGenerator] = None,
        seed: int = 0
        ):

        super().__init__(n_obj=len(fitness_function.name),
                         n_var=1)
        
        self.seed = seed
 
        set_seed(seed)

        assert fitness_function is not None
        assert simulate_function is not None
        assert len(fitness_function.min_or_max) == len(fitness_function.name)

        self.problem_name = problem_name
        self.names_dim_scenario = ["scenario"]

        self.fitness_function = fitness_function
        self.simulate_function = partial(simulate_function, name=problem_name)
        self.critical_function = critical_function
        self.simulation_variables = simulation_variables
        if design_names is not None:
            self.design_names = design_names
        else:
            self.design_names = simulation_variables

        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = fitness_function.name
        self.feature_handler_config_path = feature_handler_config_path

        if feature_handler_config_path is not None:
            self.feature_handler = FeatureHandler.from_json(feature_handler_config_path)
            # wandb.log(self.feature_handler.model_dump())

        self.scenario_generator = scenario_generator
        if self.scenario_generator is not None:
            self.scenario_generator.feature_handler = self.feature_handler
        self.context = context

        self.counter = 0
        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")
            
    def _evaluate(self, x, out, *args, **kwargs):
        self.counter = self.counter + 1
        log.info(f"Running evaluation number {self.counter}")
        try:
            # log.info(f"x: {x}")
            simout_list = self.simulate_function(self.simulation_variables, x)
        except Exception as e:
            log.info("Exception during simulation ocurred: ")
            raise e
        out["SO"] = []
        vector_list = []
        label_list = []

        for i, simout in enumerate(simout_list):
            kwargs["individual"] = x[i]

            out["SO"].append(simout)
            vector_fitness = np.asarray(
                self.signs) * np.array(self.fitness_function.eval(simout, **kwargs))
            vector_list.append(np.array(vector_fitness))
            label_list.append(self.critical_function.eval(vector_fitness, simout = simout))

        out["F"] = np.vstack(vector_list)
        out["CB"] = label_list

    def is_simulation(self):
        return True
