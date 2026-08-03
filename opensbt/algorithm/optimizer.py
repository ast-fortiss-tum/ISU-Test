import os

from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult
from pymoo.optimize import minimize
from pymoo.core.problem import Problem  
from pymoo.core.algorithm import Algorithm

import dill
from opensbt.config import BACKUP_FOLDER, RESULTS_FOLDER, EXPERIMENTAL_MODE, BACKUP_ITERATIONS
from opensbt.visualization.visualizer import create_save_folder, backup_object, delete_backup
from pymoo.termination import get_termination

class Optimizer(ABC):
    """ Base class for all optimizers in OpenSBT.  Subclasses need to   
        implement the __init__ method. The run method has to be overriden when non pymoo implemented algorithms are used.
        For reference consider the implementation of the NSGA-II-DT optimizer in opensbt/algorithm/nsga2dt_optimizer.py
    """
    
    algorithm_name: str
    parameters: Dict
    config: SearchConfiguration
    problem: Problem
    algorithm: Algorithm
    termination: object
    save_history: bool
    
    parameters: str
    save_folder: str = None
    
    @abstractmethod
    def __init__(self, problem: SimulationProblem, config: SearchConfiguration, **kwargs):
        """Initialize here the Optimization algorithm to be used for search-based testing.

        :param problem: The testing problem to be solved.
        :type problem: SimulationProblem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        """
        pass

    def run(self) -> SimulationResult:
        if self.termination is None:
            ''' Prioritize max search time over set maximal number of generations'''
            if self.config.maximal_execution_time is not None:
                self.termination = get_termination("time", self.config.maximal_execution_time)
                print(f"[Optimizer] Using time-based termination: {self.config.maximal_execution_time}.")
            else:
                self.termination = get_termination("n_gen", self.config.n_generations)
                print(f"[Optimizer] Using generation-based termination: {self.config.n_generations} generations.")
        while(self.termination.do_continue()):
            self.algorithm.next()
            self.termination.update(self.algorithm)
            print(f"[Optimizer] Iteration {self.algorithm.n_iter} termination progress: {self.termination.perc:.2f}")
            if BACKUP_ITERATIONS:
                n_iter = self.algorithm.n_iter - 1
                dill.detect.trace(True)   # will show what dill is trying to pickle
                # store backup for current iteration
                backup_object(self.algorithm, 
                            self.save_folder, 
                            name = f"algorithm_iteration_{n_iter}")
                backup_object(self.termination, 
                            self.save_folder, 
                            name = f"termination_iteration_{n_iter}")                
                # delete old backup
                delete_backup(save_folder = self.save_folder + BACKUP_FOLDER,
                            name = f"algorithm_iteration_{n_iter - 1}")
                delete_backup(save_folder = self.save_folder + BACKUP_FOLDER,
                            name = f"termination_iteration_{n_iter - 1}")
                                                
        res = SimulationResult.from_result(self.algorithm.result())
        return res
    
    def resume(self, save_folder: str):
        """ Resume an optimization from a backup folder.
        :param save_folder: The folder where the backup is stored.
        :return: The optimizer instance ready to run from the last backuped iteration.
        :rtype: Optimizer
        """
        # if save folder already contains a backup folder, load from there
        if os.path.exists(os.path.join(save_folder, BACKUP_FOLDER)):
            try:
                #find backup files, the names should be "algorithm_iteration_{n}.pkl" and "termination_iteration_{n}.pkl"
                print("[Optimizer] Resuming from folder/path:", os.path.join(save_folder, BACKUP_FOLDER))
                ckpt_files = [f for f in os.listdir(os.path.join(save_folder, BACKUP_FOLDER))]
                for f in ckpt_files:
                    print("[Optimizer] Found backup files:", f)
                    if f.startswith("algorithm_iteration_"):
                        alg_files = f
                    if f.startswith("termination_iteration_"):
                        term_files = f
                self.algorithm = dill.load(open(os.path.join(save_folder, BACKUP_FOLDER, alg_files), "rb"))
                self.termination = dill.load(open(os.path.join(save_folder, BACKUP_FOLDER, term_files), "rb"))
            
                print("[Optimizer] Resumed from backup files:", alg_files, term_files)
            except Exception as e:
                print("[Optimizer] Failed to load backup files, starting from scratch. Error:", e)
                self.algorithm.setup(problem = self.problem, 
                            termination = self.termination,
                            save_history = self.save_history)

        else:    
            print("[Optimizer] Initialized from scratch, no resume path provided.")       
            self.algorithm.setup(problem = self.problem, 
                            termination = self.termination,
                            save_history = self.save_history)
        return self