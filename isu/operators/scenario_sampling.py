from isu.model.models import Scenario
from isu.model.problem import ISUProblem
from .base_classes import ScenarioSamplingBase

from typing import List, Tuple, Optional
import random
import multiprocessing
from testflows.combinatorics import Covering
from isu.features.models import DiscreteFeature


class ScenarioSampling(ScenarioSamplingBase):
    def __init__(
        self,
        variable_length=False,  
        render=False
    ):
        super().__init__()
        self.variable_length = variable_length
        self.render = render

    def _sample_instances(
        self, problem: ISUProblem, n_samples: int, **kwargs
    ) -> List[Scenario]:
        result = []

        for i in range(n_samples):
            vars = problem.feature_handler.sample_feature_scores()
            result.append(problem.scenario_generator.generate(
                ordinal_vars=vars.ordinal,
                categorical_vars=vars.categorical,
                continuous_vars=vars.continuous,
                render=self.render
            ))
        return result

def calculate_covering_size(params, t, result_dict):
    covering = Covering(params, strength=t)
    result_dict[t] = len(list(o for o in covering))
class ScenarioSamplingGrid(ScenarioSamplingBase):
    def __init__(
        self,
        covering_search_time: int = 30,
        total_samples: Optional[int] = None,
        t: int = None
    ):
        super().__init__()
        self.covering_search_time = covering_search_time
        self.total_samples = total_samples
        self.covering_variables = []
        self.t = t

    @staticmethod
    def _timeout(*args, **kwargs):
        raise TimeoutError

    def _get_covering(self, n_samples, features: List[DiscreteFeature]) -> List[Tuple[int]]:
        # Create parameter dictionary for features
        params = {f.name: list(range(f.num_values)) for f in features}
        

        if self.t is None:
            # Binary search for the minimal strength t that covers enough samples
            max_t, min_t = len(features) + 1, 0

            m = multiprocessing.Manager()
            covering_size_dict = m.dict()
            while max_t - min_t > 1:
                t = (max_t + min_t) // 2
                p = multiprocessing.Process(target=calculate_covering_size, args=[params, t, covering_size_dict])
                p.start()
                p.join(self.covering_search_time)
                if p.is_alive():
                    max_t = t
                    p.kill()
                    p.join()
                else:
                    if covering_size_dict[t] > n_samples:
                        max_t = t
                    else:
                        min_t = t
            t = max_t
        else:
            t = self.t

        print(f"Using t={t} for covering sampling")
            
        covering_raw = Covering(params, strength=t)

        # Convert to list of tuples
        covering = [tuple(c[f.name] for f in features) for c in covering_raw]

        # Calculate max samples per combination
        max_samples_per_combination = n_samples // len(covering) + 1

        # Initialize combination counts
        combination_to_count = {c: max_samples_per_combination for c in covering}

        # Adjust counts by randomly removing excess samples
        total_samples = len(covering) * max_samples_per_combination
        excess_samples = total_samples - n_samples
        for c in random.sample(covering, excess_samples):
            combination_to_count[c] -= 1

        # Expand combinations into the result list
        result = [list(c) for c, count in combination_to_count.items() for _ in range(count)]
        print(f"For {n_samples} generated a covering of size {len(result)} with t={t}")
        random.shuffle(result)
        return result
    
    def sample_covering(self, problem: ISUProblem, n_samples: int, **kwargs):
        # Gather categorical and ordinal features
        categorical_features = list(problem.feature_handler.categorical_features.values())
        ordinal_features = list(problem.feature_handler.ordinal_features.values())
        features = categorical_features + ordinal_features

        # Get covering variables
        self.covering_variables = self._get_covering(n_samples, features)

    def _sample_instances(self, problem: ISUProblem, n_samples: int, **kwargs) -> List[Scenario]:
        # Gather categorical and ordinal features
        categorical_features = list(problem.feature_handler.categorical_features.values())
        ordinal_features = list(problem.feature_handler.ordinal_features.values())
        vars = problem.feature_handler.sample_feature_scores()

        result = []
        for i in range(n_samples):
            if len(self.covering_variables) == 0:
                to_sample = self.total_samples or n_samples - i
                self.sample_covering(problem, to_sample)
            covering_variable = self.covering_variables.pop()

            # Split into categorical and ordinal variables
            categorical_vars = covering_variable[:len(categorical_features)]
            ordinal_vars = covering_variable[len(categorical_features):]

            # Normalize ordinal variables
            ordinal_vars = [(value + 0.5) / feature.num_values for value, feature in zip(ordinal_vars, ordinal_features)]

            scenario = problem.scenario_generator.generate(
                ordinal_vars=ordinal_vars,
                categorical_vars=categorical_vars,
                continuous_vars=vars.continuous,
                render=False
            )
            result.append(scenario)

        return result
