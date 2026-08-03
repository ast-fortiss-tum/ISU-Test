from .base_classes import ScenarioRepairBase
from isu.model.problem import ISUProblem
from isu.model.models import Scenario
class RepairScenarioGenerator(ScenarioRepairBase):
    def __init__(
        self,
    ):
        super().__init__()
        self.generate_scenario = True
    
    def _repair_instance(self, problem: ISUProblem, instance: Scenario, **kwargs):
        if instance.predicts_dict is not None:
            return instance
        return problem.scenario_generator.generate(
                ordinal_vars=instance.ordinal_vars,
                categorical_vars=instance.categorical_vars,
                continuous_vars=instance.continuous_vars,
                render=True
            )


class NoScenarioRepair(ScenarioRepairBase):
    def _repair_instance(self, problem: ISUProblem, instance: Scenario, **kwargs):
        return instance