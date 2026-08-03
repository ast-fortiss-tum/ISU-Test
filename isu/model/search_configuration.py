from opensbt.experiment.search_configuration import SearchConfiguration, SearchOperators

from isu.operators.base_classes import (
    ScenarioCrossoverBase,
    ScenarioMutationBase,
    ScenarioSamplingBase,
    ScenarioDuplicateEliminationBase,
    ScenarioRepairBase
)

from isu.operators.scenario_crossover import ScenarioCrossover
from isu.operators.scenario_sampling import ScenarioSampling, ScenarioSamplingGrid
from isu.operators.scenario_mutator import ScenarioMutation
from isu.operators.scenario_duplicates import ScenarioDuplicateElimination
from isu.operators.scenario_repair import RepairScenarioGenerator

class ISUSearchOperators(SearchOperators):
    crossover: ScenarioCrossoverBase = ScenarioCrossover()
    sampling: ScenarioSamplingBase = ScenarioSampling()
    duplicate_elimination: ScenarioDuplicateEliminationBase = ScenarioDuplicateElimination()
    mutation: ScenarioMutationBase = ScenarioMutation()
    repair: ScenarioRepairBase = RepairScenarioGenerator()
    
class ISUSearchConfiguration(SearchConfiguration):
    operators: SearchOperators = ISUSearchOperators()