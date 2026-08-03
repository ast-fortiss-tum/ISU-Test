from typing import List

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.pm import PolynomialMutation

from isu.model.models import Scenario
from isu.model.problem import ISUProblem

from .base_classes import ScenarioMutationBase

class ChoiceMutation(Mutation):
    def _do(self, problem: ISUProblem, X, **kwargs):
        prob_var = min(0.5, 1 / len(problem.feature_handler.categorical_features))
        for k, feature in enumerate(problem.feature_handler.categorical_features.values()):
            mut = np.where(np.random.random(len(X)) < prob_var)[0]
            X[mut, k] = np.random.choice(feature.num_values, len(mut))

        return X

class ScenarioMutation(ScenarioMutationBase):
    def __init__(
        self,
        mut_prob=0.9,
        temperature=0.3
    ):  # the last length/2 values are seg length values
        super().__init__()
        self.mut_prob = mut_prob
        self.temperature = temperature
        self.poly = PolynomialMutation(prob=self.mut_prob, eta=30)
        self.rm = ChoiceMutation(prob=self.mut_prob)

    def _instance_mutation(
        self, problem: ISUProblem, scenarios: List[Scenario]
    ) -> List[Scenario]:
        new_ordinal_vars = self._ordinal_vars_mutation(problem, scenarios, self.poly)
        new_categorical_vars = self._categorical_vars_mutation(problem, scenarios, self.rm)
        new_continuous_vars = self._continuous_vars_mutation(problem, scenarios, self.poly)

        result = []
        for i in range(len(scenarios)):
            result.append(problem.scenario_generator.generate(
                ordinal_vars=new_ordinal_vars[i],
                categorical_vars=new_categorical_vars[i],
                continuous_vars=new_continuous_vars[i],
                render=False
            ))

        return result
