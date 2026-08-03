from typing import List

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX

from isu.model.models import Scenario
from isu.model.problem import ISUProblem

from .base_classes import ScenarioCrossoverBase
class ScenarioCrossover(ScenarioCrossoverBase):
    def __init__(
        self, crossover_rate=0.7, temperature=0.3
    ):
        super().__init__(2, 1)
        self.crossover_rate = crossover_rate
        self.temperature = temperature
        self.sbx = SBX(
            prob=self.crossover_rate, prob_var=self.crossover_rate, eta=30, vtype=float
        )
        self.ux = UX()

    def _instance_crossover(
        self, problem: ISUProblem, matings: List[List[Scenario]]
    ) -> List[List[Scenario]]:

        #print("Crossover matings:", len(matings), matings)
        offspring_ordinal_vars = self._ordinal_vars_crossover(problem, matings, self.sbx)
        #print("Crossover offspring ordinal vars:", offspring_ordinal_vars)
        offspring_categorical_vars = self._categorical_vars_crossover(problem, matings, self.ux)
        #print("Crossover offspring categorical vars:", offspring_categorical_vars)
        offspring_continuous_vars = self._continuous_vars_crossover(problem, matings, self.sbx)
        #print("Crossover offspring continuous vars:", offspring_continuous_vars)

        result: List[List[Scenario]] = []
        for _ in matings:
            result.append([])
        for i in range(self.n_offsprings):
            for j in range(len(matings)):
                #print("Vars used:", offspring_ordinal_vars[j][i], offspring_categorical_vars[j][i], offspring_continuous_vars[j][i])
                result[j].append(problem.scenario_generator.generate(
                    ordinal_vars=offspring_ordinal_vars[j][i],
                    categorical_vars=offspring_categorical_vars[j][i],
                    continuous_vars=offspring_continuous_vars[j][i],
                    render=False
                ))
        # print("Crossover result:", result)
        return result
