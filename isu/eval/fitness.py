from typing import Tuple, List
from opensbt.evaluation.fitness import Fitness
from isu.simulation.sut_simulator import ISUSimulationOutput
from isu.eval.similarity import get_similarity_individual
from isu.simulation.sut_simulator import feature_weight_map
from pymoo.core.individual import Individual
from functools import partial 

class FitnessCorrectPredictions(Fitness):
    @property
    def min_or_max(self):
        return "min", 

    @property
    def name(self):
        return "correct_predicts", 

    def eval(self, simout: ISUSimulationOutput, **kwargs) -> Tuple[float]:
        predicts_dict = simout.scenario.predicts_dict
        params_dict = simout.scenario.params_dict

        correct_predicts = 0
        for k, v in predicts_dict.items():   
            if k in params_dict:
                if str(v) == str(params_dict[k]):
                    correct_predicts += feature_weight_map[k]
                elif k == "suitcase_location" and params_dict[k] in [x.strip() for x in v.split(",")]:
                    print(f"suitcase location matched: pred {v}, actual {params_dict[k]}")
                    correct_predicts += feature_weight_map[k]
        correct_predicts = round(correct_predicts, 2)
        print(f"Total correct predicts score: {correct_predicts}")
        return (correct_predicts,)

class FitnessDiverse(Fitness):
    def __init__(self, bounds=None, diversify=False) -> None:
        super().__init__()
        self.bounds = bounds

    @property
    def min_or_max(self):
        return "max",
    
    @property
    def name(self):
        return "distance",

    def eval(self, simout: ISUSimulationOutput, **kwargs) -> Tuple[float]:  
        distance_archive = 0
        if "algorithm" in kwargs:
            algorithm = kwargs["algorithm"]
            
            if not hasattr(algorithm, 'archive_novelty'):
                print("no archive novelty, using archive")
                _, distance_archive = self.closest_individual_from_vars(
                                                kwargs["individual"], 
                                                algorithm.archive,
                                                dist_fnc = partial(get_similarity_individual, bounds = self.bounds)
                                            )
  
            else:
                print("using archive novelty")
                _, distance_archive = algorithm.archive_novelty.closest_individual_from_vars(
                                                kwargs["individual"], 
                                                dist_fnc = partial(get_similarity_individual, bounds = self.bounds)
                                            )
                
        print("archive distance:", distance_archive)
        f_vector = (distance_archive,)
        return f_vector
    
    def closest_individual_from_ind(self, ind, archive, dist_fnc):
        if len(archive) == 0:
            return None, 0
        else:
            closest_ind = None
            closest_dist = 10000
            for ind_other in archive:
                dist = dist_fnc(ind, ind_other)
                # print("calculated dist:", dist)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ind = ind_other
            return (closest_ind, closest_dist)
        
    def closest_individual_from_vars(self, variables, archive, dist_fnc):
        # print("variables:", variables)
        ind = Individual()
        ind.set("X", variables)
        return self.closest_individual_from_ind(ind, archive, dist_fnc)

    
class FitnessMerged(Fitness):
    def __init__(self, fitnesses: List[Fitness]) -> None:
        super().__init__()
        self.fitnesses = fitnesses

    @property
    def min_or_max(self):
        res = ()
        for fitness in self.fitnesses:
            res += fitness.min_or_max
        return res
    
    @property
    def name(self):
        res = ()
        for fitness in self.fitnesses:
            res += fitness.name
        return res

    def eval(self, simout: ISUSimulationOutput, **kwargs) -> Tuple[float]:  
        res = ()
        for fitness in self.fitnesses:
            res += fitness.eval(simout, **kwargs)
        
        return res