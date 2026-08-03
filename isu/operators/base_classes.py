import copy
import dataclasses
from typing import List, Union, final, Optional
from abc import ABC, abstractmethod
from isu.model.problem import ISUProblem

import numpy as np
import pydantic
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from opensbt.operators import (
    CustomObjectCrossoverBase,
    CustomObjectDuplicateEliminationBase,
    CustomObjectMutationBase,
    CustomObjectOperatorBase,
    CustomObjectSamplingBase,
    CustomObjectRepairBase
)

from isu.model.models import Scenario
@dataclasses.dataclass
class ScenarioOperatorBase(CustomObjectOperatorBase, ABC):
    @staticmethod
    def _has_ordinal_features(scenario: Scenario) -> bool:
        return len(scenario.ordinal_vars) > 0

    @staticmethod
    def _has_categorical_features(scenario: Scenario) -> bool:
        return len(scenario.categorical_vars) > 0
    
    @staticmethod
    def _has_continuous_features(scenario: Scenario) -> bool:
        return len(scenario.continuous_vars) > 0

    @staticmethod
    def _validate_instance(obj):
        assert isinstance(obj, Scenario), "Population must be made of Scenario instances"


class ScenarioCrossoverBase(CustomObjectCrossoverBase, ScenarioOperatorBase, ABC):
    def _ordinal_vars_crossover(
        self,
        problem: ISUProblem,
        matings: List[List[Scenario]],
        crossover: Crossover,
    ) -> List[List[List[float]]]:
        if self._has_ordinal_features(matings[0][0]):
            return self._vars_crossover(
                problem, matings, crossover, attribute_name="ordinal_vars")
        else:
            return self._empty_crossover(len(matings))

    def _categorical_vars_crossover(
        self,
        problem: ISUProblem,
        matings: List[List[Scenario]],
        crossover: Crossover,
    ) -> List[List[List[int]]]:
        if self._has_categorical_features(matings[0][0]):
            return self._vars_crossover(
                problem, matings, crossover, attribute_name="categorical_vars")
        else:
            return self._empty_crossover(len(matings))
        
    def _continuous_vars_crossover(
        self,
        problem: ISUProblem,
        matings: List[List[Scenario]],
        crossover: Crossover,
    ) -> List[List[List[float]]]:
        if self._has_continuous_features(matings[0][0]):
            feature_handler = problem.feature_handler
            # extract xl and xu from feature handler
            xl = np.array([f.lb for f in feature_handler.continuous_features.values()])
            xu = np.array([f.ub for f in feature_handler.continuous_features.values()])
            return self._vars_crossover(
                problem, matings, crossover, attribute_name="continuous_vars", xl=xl, xu=xu)
        else:
            return self._empty_crossover(len(matings))


class ScenarioMutationBase(CustomObjectMutationBase, ScenarioOperatorBase, ABC):
    def _ordinal_vars_mutation(
        self,
        problem: ISUProblem,
        scenarios: List[Scenario],
        mutation: Mutation,
    ) -> List[List[float]]:
        if self._has_ordinal_features(scenarios[0]):
            return self._vars_mutation(problem, scenarios, mutation, "ordinal_vars")
        else:
            return self._empty_mutation(len(scenarios))

    def _categorical_vars_mutation(
        self,
        problem: ISUProblem,
        scenarios: List[Scenario],
        mutation: Mutation,
    ) -> List[List[int]]:
        if self._has_categorical_features(scenarios[0]):
            return self._vars_mutation(problem, scenarios, mutation, "categorical_vars")
        else:
            return self._empty_mutation(len(scenarios))
    def _continuous_vars_mutation(
        self,
        problem: ISUProblem,
        scenarios: List[Scenario],
        mutation: Mutation,
    ) -> List[List[float]]:
        if self._has_continuous_features(scenarios[0]):
            feature_handler = problem.feature_handler
            # TODO: extract xl and xu from feature handler
            xl = np.array([f.lb for f in feature_handler.continuous_features.values()])
            xu = np.array([f.ub for f in feature_handler.continuous_features.values()])
            return self._vars_mutation(problem, scenarios, mutation, "continuous_vars", xl=xl, xu=xu)
        else:
            return self._empty_mutation(len(scenarios))


class ScenarioSamplingBase(CustomObjectSamplingBase, ScenarioOperatorBase, ABC):
    pass


class ScenarioDuplicateEliminationBase(
    CustomObjectDuplicateEliminationBase, ScenarioOperatorBase, ABC
):
    pass

class ScenarioRepairBase(
    CustomObjectRepairBase, ScenarioOperatorBase, ABC
):
    pass