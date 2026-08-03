from isu.model.models import Scenario
from isu.utils.math import euclid_distance, mae, mse

from .base_classes import ScenarioDuplicateEliminationBase


class ScenarioDuplicateElimination(ScenarioDuplicateEliminationBase):
    def _instances_equal(self, a: Scenario, b: Scenario) -> bool:
        # return (
        #     euclid_distance(
        #     a.ordinal_vars, b.ordinal_vars
        # ) < 1 and
        #     a.categorical_vars == b.categorical_vars and
        #     euclid_distance(
        #         a.continuous_vars, b.continuous_vars
        #     ) < 1
        # )
        return a.categorical_vars == b.categorical_vars
    

class ScenarioDuplicateEliminationDiscreteMSE(ScenarioDuplicateEliminationBase):
    def _instances_equal(self, a: Scenario, b: Scenario) -> bool:
        #  return (
        #     mse(a.ordinal_vars, b.ordinal_vars) < 1 and
        #     a.categorical_vars == b.categorical_vars and
        #     mse(a.continuous_vars, b.continuous_vars) < 1
        # )
        return a.categorical_vars == b.categorical_vars

