<div align="center">
  <h1>ISU-Test: Search-based Testing of Vision Language Models for In-Car Scene Understanding</h1>
  <p><a href="https://doi.org/10.1145/3832783.3834506"><img src="https://img.shields.io/badge/DOI-10.1145%2F3832783.3834506-blue" alt="DOI" /></a></p>
  <p>Dynamic test case generation and search-based testing for vision-language in-car scene understanding.</p>
</div>

## Overview

This project provides search-based generation and testing of in-car scene-understanding scenarios. The current entry point is [scripts/generate_isu_data.py](scripts/generate_isu_data.py). It supports both genetic search and random sampling, renders Blender scenes, evaluates the selected SUT, and writes the generated artifacts and metadata to a timestamped result folder.

## Example outputs

The following channel examples are from the random-sampling run in
`docs/readme_assets/scene_v3_rs/`:

<table align="center">
  <tr>
    <td><strong>RGB</strong><br /><img src="docs/readme_assets/scene_v3_rs/rgb.png" alt="Rendered RGB in-car scene" width="360" /></td>
    <td><strong>Canny edges</strong><br /><img src="docs/readme_assets/scene_v3_rs/canny.png" alt="Canny edge channel" width="360" /></td>
  </tr>
  <tr>
    <td><strong>Semantic segmentation</strong><br /><img src="docs/readme_assets/scene_v3_rs/semantic_segmentation.png" alt="Semantic segmentation channel" width="360" /></td>
    <td><strong>Instance segmentation</strong><br /><img src="docs/readme_assets/scene_v3_rs/instance_segmentation.png" alt="Instance segmentation channel" width="360" /></td>
  </tr>
</table>

For each successful sample, the pipeline can produce these image channels:

| Channel | Output | Description |
| --- | --- | --- |
| RGB | `images/<sample>_sim.png` | Rendered Blender scene used as the primary input. |
| Semantic segmentation | `seg/<sample>_seg.png` | Pixel colors identify semantic classes such as human, phone, suitcase, baby seat, safety belt, beverage, car interior, exterior, blanket, and seat. |
| Canny | `canny/<sample>_canny.png` | Geometry-edge image produced by the Blender Freestyle pass and rotated 180 degrees during post-processing. |
| Instance segmentation | `instance_seg/<sample>_instance_seg.png` | Pixel colors identify individual scene instances, including occupants, phones, belts, beverages, suitcase, baby seat, baby, seats, and car regions. |

The channel color definitions are written to `seg_class_map.json` and `instance_seg_class_map.json`. `sample_manifest.json` records the relative path and scenario parameters for every successful sample, as well as failed samples when applicable.

## Repository structure

- [scripts/generate_isu_data.py](scripts/generate_isu_data.py): Unified random-sampling and genetic-search entry point.
- [requirements.txt](requirements.txt): Python dependencies.
- [configs/isu_challenge_features.json](configs/isu_challenge_features.json): Default feature configuration used by the unified runner.

Directories:

- `isu/` — ISU-specific modules (config, output handling, analysis, blender helpers, evaluation, simulation and SUT adapters).
  - `isu/analysis/` — analysis and plotting utilities.
  - `isu/eval/` — evaluation functions (fitness, similarity, critical checks).
  - `isu/model/` — model and scenario generation code.
  - `isu/simulation/` — simulators and prompts for SUTs.

- `llm/` — LLM integration layer, clients and adapters for different LLM providers.
  - `llm/algorithm/` — sampling and LLM-driven algorithms.
  - `llm/features/` — feature handlers and models.

- `opensbt/` — core OpenSBT abstractions, algorithms, utilities and visualization code.

- `results/` — timestamped output folders created by the unified runner.

## Usage

Run the unified script from the repository root. It creates a timestamped folder below `--output-dir`, for example `results/scene_v3_ga/20260904_120000`.

```bash
# Genetic algorithm search (default)
python scripts/generate_isu_data.py \
  --algorithm nsga2 \
  --population-size 20 \
  --n-generations 30 \
  --sut dummy \
  --features-config configs/isu_challenge_features.json \
  --scene isu/blender/scenes/scene_v3.blend \
  --output-dir results/scene_v3_ga \
  --seed 1 \
  --no-wandb

# Random sampling with the same output pipeline
python scripts/generate_isu_data.py \
  --algorithm rs \
  --population-size 100 \
  --sut dummy \
  --output-dir results/scene_v3_rs \
  --no-wandb
```

### Available flags

| Flag | Default | Description |
| --- | --- | --- |
| `--algorithm` | `nsga2` | Search mode: `nsga2` for genetic search or `rs` for random sampling. |
| `--population-size` | `4` | Number of scenarios per population or random-sampling run. |
| `--n-generations` | `2` | Number of genetic-search generations. It is retained in the output metadata for RS runs. |
| `--seed` | `42` | Random seed. |
| `--sut` | `dummy` | SUT adapter, for example `dummy`, `gpt5`, `gpt4o`, or a supported Gemini adapter. |
| `--features-config` | `configs/isu_challenge_features.json` | Feature-space configuration JSON. |
| `--scene` | `isu/blender/scenes/scene_v3.blend` | Blender scene path. |
| `--output-dir` | `results/scene_v3_ga` | Parent output directory; a timestamped run directory is appended. |
| `--timeout-sec` | `220` | Maximum time for an individual Blender render. |
| `--max-time` | unset | Optional total search duration in `hh:mm:ss` format. |
| `--no-wandb` | disabled | Disable Weights & Biases logging. Logging is enabled unless this flag is provided. |

The generated result folder contains `json_inputs/`, `images/`, `seg/`, `canny/`, `instance_seg/`, `critical_images/`, `sample_manifest.json`, `seg_class_map.json`, and `instance_seg_class_map.json`. Genetic-search result files and plots are written there as well.
## Replication

A snapshot for the replication of the results in the paper is provided here: https://figshare.com/s/cb5b0eae0411e54b1bbd

## Dataset

A generated dataset of interior scenes with ISU-Test will be provided soon.

## License & Attributions

ISU-Test is released under the [MIT License](LICENSE). Copyright is held by Lev Sorokin and Rifaath Ameen (BMW). Third-party assets and dependencies, including SMPL-X Body, remain subject to their respective licenses and terms.

SMPL-X Body was used for character animation courtesy of the Max Planck Institute for Intelligent Systems. 

Authors: Lev Sorokin, Rifaath Ameen (BMW), Chen Yang, Ken E. Friedl, and Andrea Stocco.

## Citation

If you use **ISU-Test** in your research, please cite the accompanying ASE 2026 paper:

```bibtex
@inproceedings{sorokin2026isutest,
  author    = {Lev Sorokin and Rifaath Ameen and Chen Yang and Ken E. Friedl and Andrea Stocco},
  title     = {Search-based Testing of Vision Language Models for In-Car Scene Understanding},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026), Industry Track},
  year      = {2026},
  doi       = {10.1145/3832783.3834506}
}
```

A preprint is available on arXiv: https://arxiv.org/abs/2607.02300

## Contact

Lev Sorokin and Rifaath Ameen (BMW) \
lev.sorokin@bmw.de  
rifaath.ameen@bmw.de
