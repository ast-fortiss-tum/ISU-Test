import os
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.experiment.search_configuration import SearchConfiguration, SearchOperators
from isu.model.problem import ISUProblem
from isu.eval.fitness import FitnessMerged, FitnessDiverse, FitnessCorrectPredictions
from isu.eval.critical import CriticalMerged, CriticalByFitnessThreshold

from opensbt.utils.log_utils import log, setup_logging, disable_pymoo_warnings
from opensbt.config import RESULTS_FOLDER, LOG_FILE
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer

from opensbt.algorithm.ps_rand import PureSampling
from opensbt.algorithm.optimizer import Optimizer

from isu.model.scenario_generation import BlenderScenarioGenerator

import argparse
from datetime import datetime
#import weave
import wandb
import warnings
from opensbt.visualization.visualizer import create_save_folder

from opensbt.utils.wandb import logging_callback_archive, TableCallback
from opensbt.utils.callback import merged_callbacks

from isu.simulation.sut_simulator import ISUSimulator
from isu.model.problem import ISUProblem
from functools import partial   

from isu.model.search_configuration import ISUSearchOperators
from isu.operators.scenario_sampling import ScenarioSampling, ScenarioSamplingGrid
from opensbt.config import RESULTS_FOLDER, EXPERIMENTAL_MODE


# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Seed"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["rs", "gs", "nsga2", "nsga2d"],
        default="nsga2",
        help="Algorithm.",
    )
    parser.add_argument(
        "--sut",
        type=str,
        default="gpt5-chat",
        help="The ISU tested.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=4,
        help="Population size for GA (default: 4)",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1,
        help="Number of generations (default: 1)",
    )
    parser.add_argument(
        "--max_time",
        type=str,
        default=None,
        help="Maximal execution time as string 'hh:mm:ss'",
    )
    parser.add_argument(
        "--features_config",
        type=str,
        default="configs/isu_features.json",
        help="Path to the file with feature config",
    )
    parser.add_argument(
            "--no_wandb",
            action="store_true",
            help="Turn off wanbd logging"
        )
    return parser.parse_args() 

if __name__ == "__main__":
    args = parse_args()

    os.chmod(os.getcwd(), 0o777)
    logger = log.getLogger(__name__)
    setup_logging(LOG_FILE)
    disable_pymoo_warnings()

    datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seed = args.seed

    problem_name = (
        f"{args.sut}"
        + f"_{args.population_size}n"
        + (f"_{args.n_generations}i" if args.n_generations is not None else "")
        + (
            f"_{args.max_time.replace(':', '_')}t"
            if args.max_time is not None
            else ""
        )
        + f"_{args.seed}seed"
        + f"_{args.algorithm.upper()}"
        + f"_{args.features_config.split('.')[0].replace('/', '_')}"
    )
    save_folder = create_save_folder(problem_name, 
                                RESULTS_FOLDER,
                                algorithm_name=args.algorithm)
    print(f"Results will be saved to: {save_folder}")

    os.makedirs(save_folder + os.sep + "critical_images", exist_ok=True)
    os.makedirs(save_folder + os.sep + "json_inputs", exist_ok=True)
    os.makedirs(save_folder + os.sep + "images", exist_ok=True)

    search_operators = ISUSearchOperators()
    search_operators.sampling = (
            ScenarioSamplingGrid(
                total_samples=args.population_size,
                t = 3
            )
            if args.algorithm == "gs"
            else ScenarioSampling(render=True)  
            if args.algorithm == "rs"
            else ScenarioSampling(render=False)
            if args.algorithm in ["nsga2", "nsga2d"]
            else None
        )

    config = SearchConfiguration(
        operators=search_operators,
        population_size=args.population_size,
        n_generations=args.n_generations,
        maximal_execution_time=args.max_time,
        results_folder=RESULTS_FOLDER,
        n_repopulate_max=0,
        archive_threshold=0.04 #1/24
    )

    fitness = FitnessMerged([
        FitnessCorrectPredictions(),
        FitnessDiverse(),
    ])

    critical = CriticalMerged(
        fitness_names=fitness.name,
        criticals=[
            (CriticalByFitnessThreshold(mode="<", score=1.0), ["correct_predicts"]),
        ],
        mode="or",
        critical_save_folder=save_folder + os.sep + "critical_images"
    )

    tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "features_config"]

    if not args.no_wandb:
        # weave.init("dev")
        wandb.init(
            entity="opentest",                  # team
            project="OpenSBT-ISU",                  # the project name
            name=problem_name,                  # run name
            group=datetime.now().strftime("%d-%m-%Y"),  # group by date
            tags=tags,
        )
    else:
        wandb.init(mode="disabled")

    problem = ISUProblem(
                    problem_name=problem_name,
                    objective_names=["correct_predicts", "diversity"],
                    simulation_variables=["scenario"],
                    fitness_function=fitness,  
                    critical_function=critical,
                    simulate_function=partial(ISUSimulator.simulate, 
                                                sut=args.sut,
                                                name=problem_name
                                            ),
                    feature_handler_config_path=args.features_config,
                    scenario_generator=BlenderScenarioGenerator(
                        problem_name=problem_name,
                        json_output_dir=os.path.join(save_folder, "json_inputs"),
                        image_output_dir=os.path.join(save_folder, "images")
                    ),
                    seed = seed
                    )
    norm_bounds = {
        "continuous_vars": [
            [f.lb for f in problem.feature_handler.continuous_features.values()],
            [f.ub for f in problem.feature_handler.continuous_features.values()]
        ]
    }

    fitness.fitnesses[1].bounds = norm_bounds #pass bounds to diversity fitness

    callback = merged_callbacks(
        logging_callback_archive, TableCallback().log
    )

    optimizer_map = {
        "rs": PureSampling,
        "gs": PureSampling,
        "nsga2": NsgaIIOptimizer,
        "nsga2d": NSGAIIDOptimizer,
    }
    if args.algorithm not in optimizer_map:
        raise ValueError("Algorithm not supported")
    
    optimizer_class: type[Optimizer] = optimizer_map[args.algorithm]
    optimizer = optimizer_class(problem, config, callback=callback, algorithm_name=args.algorithm)
    
    optimizer.save_folder = create_save_folder(optimizer.problem.problem_name, 
                        config.results_folder,
                        algorithm_name=optimizer.algorithm_name,
                        is_experimental=EXPERIMENTAL_MODE)

    optimizer = optimizer.resume(optimizer.save_folder)
    res = optimizer.run()

    res.write_results(
        results_folder=optimizer.save_folder, params=optimizer.parameters, search_config=config, norm_bounds=norm_bounds
    )

    from isu.analysis.failure_patterns import print_failure_patterns
    print_failure_patterns(
        sut=args.sut,
        algo=args.algorithm,
        base_result_dir=optimizer.save_folder
    )

    log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")


    # # post-process: translate critical images from sim to sim2real, evaluate again, save wrong predictions
    # critical_images_folder = os.path.join(save_folder, "critical_images")
    # translated_folder = os.path.join(save_folder, "critical_images_sim2real")
    # os.makedirs(translated_folder, exist_ok=True)

    # #translate all critical images with sim2real, save to new folder
    # for img_file in os.listdir(critical_images_folder):
    #     if img_file.endswith("_sim.png"):
            
    #         image_path = os.path.join(critical_images_folder, img_file)
    #         gt_json_path = os.path.join(save_folder, "json_inputs", img_file.replace("_sim.png", ".json"))
    #         with open(gt_json_path, "r") as f:
    #             gt_json = json.load(f)
            
    #         translated_image_path = BlenderScenarioGenerator.call_sim2real(image_path, dummy=False, gt=gt_json) 
    #         dst_path = os.path.join(translated_folder, img_file.replace("_sim.png", "_sim2real.png"))
    #         os.rename(translated_image_path, dst_path)
    #         print(f"Translated {img_file} to sim2real and saved to {dst_path}")

    # #run evaluation again on translated images, save wrong predictions to new json files
    # for img_file in os.listdir(translated_folder):
    #     if img_file.endswith("_sim2real.png"):

    #         image_path = os.path.join(translated_folder, img_file)
    #         gt_json_path = os.path.join(save_folder, "json_inputs", img_file.replace("_sim2real.png", ".json"))
    #         with open(gt_json_path, "r") as f:
    #             gt_json = json.load(f)

    #         predicts_dict, raw_result, last_error = ISUSimulator.evaluate_image(image_path, sut=args.sut)

    #         wrong_predicts = CriticalMerged.compare_dict(predicts_dict, gt_json)
    #         if wrong_predicts:
    #             diffs = [
    #                 {"key": k, "predicted": v[0], "actual": v[1]}
    #                 for k, v in wrong_predicts.items()
    #             ]
    #             with open(os.path.join(translated_folder, img_file.replace(".png", "_wrong.json")), "w") as f:
    #                 f.write(json.dumps(diffs, ensure_ascii=False, indent=2))