"""
Run random sampling or genetic search for ISU scenarios and save Blender outputs.

Example:
    python scripts/generate_isu_data.py \
        --population-size 4 \
        --n-generations 2 \
        --scene isu/blender/scenes/scene_v3.blend \
        --features-config configs/isu_challenge_features.json \
        --output-dir results/scene_v3_ga \
        --sut dummy

    # Random sampling mode:
    python scripts/generate_isu_data.py \
        --algorithm rs --population-size 100 \
        --output-dir results/scene_v3_rs --sut dummy
"""

import argparse
import json
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

# Keep the script runnable directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymoo
import wandb
from PIL import Image

from opensbt.model_ga.individual import IndividualSimulated
from opensbt.model_ga.population import PopulationExtended
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult

pymoo.core.individual.Individual = IndividualSimulated
pymoo.core.population.Population = PopulationExtended
pymoo.core.problem.Problem = SimulationProblem
pymoo.core.result.Result = SimulationResult

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand import PureSampling
from opensbt.algorithm.optimizer import Optimizer
from opensbt.config import EXPERIMENTAL_MODE
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.utils.log_utils import disable_pymoo_warnings, log, setup_logging
from opensbt.visualization.visualizer import create_save_folder

from isu.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from isu.eval.fitness import FitnessCorrectPredictions, FitnessDiverse, FitnessMerged
from isu.model.problem import ISUProblem
from isu.model.scenario_generation import BlenderScenarioGenerator
from isu.model.search_configuration import ISUSearchOperators
from isu.simulation.sut_simulator import ISUSimulator
from isu.operators.scenario_sampling import ScenarioSampling


SEG_PALETTE = {
    0: (20, 20, 20), 1: (220, 30, 60), 2: (30, 100, 200),
    3: (210, 160, 0), 4: (30, 150, 80), 5: (220, 100, 0),
    6: (160, 30, 80), 7: (90, 60, 200), 8: (50, 130, 180),
    9: (130, 50, 160), 10: (100, 60, 30),
}
SEG_CLASS_NAMES = {
    0: "unlabeled", 1: "human", 2: "phone", 3: "suitcase",
    4: "baby_seat", 5: "safety_belt", 6: "beverage",
    7: "car_interior", 8: "exterior", 9: "blanket", 10: "seat",
}
INST_PALETTE = {
    0: (1, 1, 1), 1: (220, 30, 60), 2: (30, 100, 200),
    3: (210, 160, 0), 4: (30, 150, 80), 5: (220, 100, 0),
    6: (130, 50, 160), 7: (140, 80, 255), 8: (50, 130, 180),
    9: (160, 30, 80), 10: (200, 100, 0), 11: (0, 160, 100),
    12: (0, 140, 200), 13: (200, 0, 90), 14: (0, 100, 60),
    15: (200, 200, 0), 16: (38, 15, 3), 17: (28, 15, 150),
    18: (8, 63, 122),
}
INST_CLASS_NAMES = {
    0: "background", 1: "driver", 2: "passenger_codriver",
    3: "passenger_rear_left", 4: "passenger_rear_right",
    5: "driver_seatbelt", 6: "codriver_seatbelt",
    7: "rear_left_seatbelt", 8: "rear_right_seatbelt",
    9: "phone_codriver", 10: "cola_bottle", 11: "cola_can",
    12: "suitcase", 13: "baby_seat", 14: "baby", 15: "driver_phone",
    16: "seat", 17: "car_interior", 18: "exterior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RS or NSGA-II search for ISU scenarios.")
    parser.add_argument(
        "--algorithm",
        choices=["nsga2", "rs"],
        default="nsga2",
        help="Search algorithm: nsga2 for genetic search or rs for random sampling.",
    )
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--n-generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sut", default="dummy", help="SUT name, for example dummy, gpt5, or gpt4o.")
    parser.add_argument("--features-config", default="configs/isu_challenge_features.json")
    parser.add_argument("--scene", default="isu/blender/scenes/scene_v3.blend")
    parser.add_argument("--output-dir", default="results/scene_v3_ga")
    parser.add_argument("--timeout-sec", type=int, default=220)
    parser.add_argument("--max-time", default=None, help="Optional duration in hh:mm:ss format.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


def make_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "critical_images": output_root / "critical_images",
        "json": output_root / "json_inputs",
        "images": output_root / "images",
        "depth": output_root / "depth",
        "seg": output_root / "seg",
        "canny": output_root / "canny",
        "instance_seg": output_root / "instance_seg",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def rotate_canny_outputs(canny_dir: Path) -> None:
    for canny_path in canny_dir.glob("*_canny.png"):
        with Image.open(canny_path) as image:
            rotated = image.rotate(180)
            rotated.save(canny_path)


def convert_depth_outputs(dirs: dict[str, Path]) -> None:
    """Convert Blender depth EXRs to grayscale and colorized PNG images."""
    try:
        import OpenEXR
    except ImportError as error:
        print(f"[WARN] EXR to PNG conversion skipped: {error}")
        return

    magma_stops = np.array([
        [0.001, 0.000, 0.014],
        [0.094, 0.039, 0.224],
        [0.251, 0.039, 0.416],
        [0.478, 0.016, 0.439],
        [0.706, 0.094, 0.369],
        [0.906, 0.306, 0.196],
        [0.988, 0.616, 0.149],
        [0.988, 0.992, 0.749],
    ], dtype=np.float32)

    for depth_exr_path in sorted(dirs["depth"].joinpath("exr").glob("*_depth.exr")):
        depth_png_path = dirs["depth_png"] / f"{depth_exr_path.stem}.png"
        depth_vis_path = dirs["depth_vis"] / f"{depth_exr_path.stem}_vis.png"
        try:
            exr_file = OpenEXR.InputFile(str(depth_exr_path))
            data_window = exr_file.header()["dataWindow"]
            width = data_window.max.x - data_window.min.x + 1
            height = data_window.max.y - data_window.min.y + 1
            channels = exr_file.header()["channels"]
            channel_name = "Z" if "Z" in channels else next(iter(channels))
            depth_array = np.frombuffer(
                exr_file.channel(channel_name), dtype=np.float32
            ).reshape(height, width)

            finite_depth = depth_array[np.isfinite(depth_array)]
            if finite_depth.size == 0:
                raise ValueError("depth image contains no finite values")
            depth_min = finite_depth.min()
            depth_max = finite_depth.max()
            if depth_max > depth_min:
                depth_normalized = (
                    (np.nan_to_num(depth_array, nan=depth_max) - depth_min)
                    / (depth_max - depth_min) * 65535
                ).clip(0, 65535).astype(np.uint16)
            else:
                depth_normalized = np.zeros_like(depth_array, dtype=np.uint16)

            Image.fromarray(depth_normalized, mode="I;16").save(str(depth_png_path))
            depth_float = depth_normalized.astype(np.float32) / 65535.0
            positions = depth_float * (len(magma_stops) - 1)
            lower = np.floor(positions).astype(np.int32)
            upper = np.minimum(lower + 1, len(magma_stops) - 1)
            fraction = (positions - lower)[..., None]
            depth_rgb = (
                magma_stops[lower] * (1.0 - fraction)
                + magma_stops[upper] * fraction
            )
            depth_rgb = (depth_rgb * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(depth_rgb, mode="RGB").save(str(depth_vis_path))
        except Exception as error:
            print(f"[WARN] Depth visualization failed for {depth_exr_path.name}: {error}")


def write_output_metadata(output_root: Path, scene_path: Path, features_config: Path,
                          population_size: int, n_generations: int, seed: int) -> None:
    def relative(path: Path) -> str | None:
        return str(path.relative_to(output_root)) if path.exists() else None

    records = []
    for json_path in sorted((output_root / "json_inputs").glob("*.json")):
        with open(json_path, encoding="utf-8") as stream:
            params = json.load(stream)
        stem = json_path.stem
        records.append({
            "sample_name": stem,
            "json_path": relative(json_path),
            "image_path": relative(output_root / "images" / f"{stem}_sim.png"),
            "depth_path": relative(output_root / "depth" / "exr" / f"{stem}_depth.exr"),
            "seg_path": relative(output_root / "seg" / f"{stem}_seg.png"),
            "canny_path": relative(output_root / "canny" / f"{stem}_canny.png"),
            "instance_seg_path": relative(output_root / "instance_seg" / f"{stem}_instance_seg.png"),
            "params": params,
        })

    manifest = {
        "summary": {
            "num_success": len(records),
            "scene": scene_path.name,
            "features_config": features_config.name,
            "population_size": population_size,
            "n_generations": n_generations,
            "seed": seed,
        },
        "records": records,
    }
    with open(output_root / "sample_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    seg_map = [
        {"id": cid, "name": SEG_CLASS_NAMES[cid], "color_rgb": list(SEG_PALETTE[cid])}
        for cid in sorted(SEG_CLASS_NAMES)
    ]
    with open(output_root / "seg_class_map.json", "w", encoding="utf-8") as stream:
        json.dump(seg_map, stream, indent=2)

    instance_map = [
        {"id": cid, "name": INST_CLASS_NAMES[cid], "color_rgb": list(INST_PALETTE[cid])}
        for cid in sorted(INST_CLASS_NAMES)
    ]
    with open(output_root / "instance_seg_class_map.json", "w", encoding="utf-8") as stream:
        json.dump(instance_map, stream, indent=2)


def main() -> None:
    args = parse_args()
    features_config = Path(args.features_config)
    scene_path = Path(args.scene)
    output_root = Path(args.output_dir)
    if not features_config.is_absolute():
        features_config = REPO_ROOT / features_config
    if not scene_path.is_absolute():
        scene_path = REPO_ROOT / scene_path
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    output_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    if not features_config.exists():
        raise FileNotFoundError(f"Feature config not found: {features_config}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    dirs = make_dirs(output_root)
    algorithm_label = "rs" if args.algorithm == "rs" else "ga"
    problem_name = f"isu_{algorithm_label}_{args.population_size}n_{args.n_generations}g_{args.seed}seed"
    setup_logging(str(output_root / "ga.log"))
    disable_pymoo_warnings()

    tags = [f"{key}:{value}" for key, value in vars(args).items() if key != "features_config"]
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            entity="opentest",
            project="OpenSBT-ISU",
            name=problem_name,
            group=args.algorithm,
            tags=tags,
        )

    operators = ISUSearchOperators()
    operators.sampling = ScenarioSampling(render=True)
    config = SearchConfiguration(
        operators=operators,
        population_size=args.population_size,
        n_generations=args.n_generations,
        maximal_execution_time=args.max_time,
        results_folder=str(output_root),
        seed=args.seed,
        n_repopulate_max=0,
        archive_threshold=0.04,
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
        critical_save_folder=str(output_root / "critical_images"),
    )

    problem = ISUProblem(
        problem_name=problem_name,
        objective_names=["correct_predicts", "diversity"],
        simulation_variables=["scenario"],
        fitness_function=fitness,
        critical_function=critical,
        simulate_function=partial(ISUSimulator.simulate, sut=args.sut),
        feature_handler_config_path=str(features_config),
        scenario_generator=BlenderScenarioGenerator(
            problem_name=problem_name,
            json_output_dir=str(dirs["json"]),
            image_output_dir=str(dirs["images"]),
            depth_output_dir=str(dirs["depth"]),
            seg_output_dir=str(dirs["seg"]),
            canny_output_dir=str(dirs["canny"]),
            instance_seg_output_dir=str(dirs["instance_seg"]),
            blend_file=str(scene_path),
        ),
        seed=args.seed,
    )

    norm_bounds = {
        "continuous_vars": [
            [feature.lb for feature in problem.feature_handler.continuous_features.values()],
            [feature.ub for feature in problem.feature_handler.continuous_features.values()],
        ]
    }
    fitness.fitnesses[1].bounds = norm_bounds

    if args.algorithm == "rs":
        optimizer_class: type[Optimizer] = PureSampling
        optimizer = optimizer_class(problem, config, algorithm_name="rs")
        result = optimizer.run()
    else:
        optimizer_class = NsgaIIOptimizer
        optimizer = optimizer_class(problem, config, algorithm_name="nsga2")
        optimizer.save_folder = create_save_folder(
            problem.problem_name,
            config.results_folder,
            algorithm_name="nsga2",
            is_experimental=EXPERIMENTAL_MODE,
        )
        optimizer = optimizer.resume(optimizer.save_folder)
        result = optimizer.run()
    result.write_results(
        results_folder=optimizer.save_folder,
        params=optimizer.parameters,
        search_config=config,
        norm_bounds=norm_bounds,
    )
    rotate_canny_outputs(dirs["canny"])
    write_output_metadata(
        output_root=output_root,
        scene_path=scene_path,
        features_config=features_config,
        population_size=args.population_size,
        n_generations=args.n_generations,
        seed=args.seed,
    )
    log.info("GA search finished in %.2f sec", result.exec_time)
    print(f"Results written to: {optimizer.save_folder}")


if __name__ == "__main__":
    main()
