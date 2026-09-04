<div align="center">
  <h1>ISU-Test: Search-based Testing of Vision Language Models for In-Car Scene Understanding</h1>
  <p>
    <a href="https://doi.org/10.1145/3832783.3834506"><img src="https://img.shields.io/badge/DOI-10.1145%2F3832783.3834506-blue" alt="DOI" /></a>
    <a href="https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow" alt="Hugging Face dataset" /></a>
  </p>
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
    <td><strong>Depth map</strong><br /><img src="docs/readme_assets/scene_v3_rs/depth.png" alt="Depth map channel" width="360" /></td>
    <td><strong>Instance segmentation</strong><br /><img src="docs/readme_assets/scene_v3_rs/instance_segmentation.png" alt="Instance segmentation channel" width="360" /></td>
  </tr>
</table>

For each successful sample, the pipeline can produce these image channels:

| Channel | Output | Description |
| --- | --- | --- |
| RGB | `images/<sample>_sim.png` | Rendered Blender scene used as the primary input. |
| Depth | `depth/exr/<sample>_depth.exr` | Normalized depth pass written by Blender as an OpenEXR file. |
| Depth PNG | `depth/png/<sample>_depth.png` | 16-bit grayscale PNG conversion of the depth pass. |
| Depth visualization | `depth/vis/<sample>_depth_vis.png` | Colorized PNG visualization of the depth pass. |
| Semantic segmentation | `seg/<sample>_seg.png` | Pixel colors identify semantic classes such as human, phone, suitcase, baby seat, safety belt, beverage, car interior, exterior, blanket, and seat. |
| Canny | `canny/<sample>_canny.png` | Geometry-edge image produced by the Blender Freestyle pass and rotated 180 degrees during post-processing. |
| Instance segmentation | `instance_seg/<sample>_instance_seg.png` | Pixel colors identify individual scene instances, including occupants, phones, belts, beverages, suitcase, baby seat, baby, seats, and car regions. |

The channel color definitions are written to `seg_class_map.json` and `instance_seg_class_map.json`. `sample_manifest.json` records the relative path and scenario parameters for every successful sample, as well as failed samples when applicable.

The `labels/` folder contains one ground-truth JSON annotation file per sample. Each label file records the scenario feature values used to generate the corresponding image and segmentation channels.

## Getting started

### 1. Install Python dependencies

From the repository root, create and activate a virtual environment, then install the dependencies listed in [requirements.txt](requirements.txt):

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On systems where PowerShell script execution is restricted, activate the environment from Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

### 2. Install Blender 5.1.2

Download and install Blender 5.1.2 from the [official Blender downloads](https://www.blender.org/download/). The renderer is invoked in background mode using the Blender executable, so the installed version should match the version used for the scene and assets.

### 3. Configure the Blender executable

Set `BLENDER_APP_PATH` in [isu/config.py](isu/config.py) to the full path of the installed Blender executable. For example, on Windows:

```python
BLENDER_APP_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
```

On macOS, use the executable inside the Blender application bundle:

```python
BLENDER_APP_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
```

The scene and driver paths are configured in the same file through `BLENDER_FILE_PATH` and `BLENDER_SCRIPT_PATH`.

### 4. Add the Blender scene

Obtain the `.blend` file representing the desired scene and copy it into [isu/blender/scenes](isu/blender/scenes). The default scene is `scene_v3.blend`. Pass another scene with the `--scene` flag when running the unified generator.

### 5. Download the required SMPL-X data and assets

SMPL-X data and assets are not redistributed with this repository. Download them from the official project pages and review their license terms before use:

- **SMPL-X/AGORA NPZ data:** download the required `.npz` files from [AGORA](https://agora.is.tue.mpg.de/) and place them under `isu/blender/smpl/data/`.
- **Other SMPL-X data:** download the required files from [SMPL-X](https://smpl-x.is.tue.mpg.de/), especially the data used by the assets under `isu/blender/assets/smplx_gt/`. The SMPL-X homepage contains the relevant license information.
- **Additional SMPL resources:** the [SMPL Made Simple](https://smpl-made-simple.is.tue.mpg.de/) tutorial collection provides links to related sources and background material.

For additional shapes or clothing textures, see the [SMPLitex texture collection](https://github.com/dancasas/dancasas.github.io/tree/master/projects/SMPLitex/SMPLitex-dataset/textures). Review the terms of each downloaded asset before including it in a dataset or redistribution.

Create the human texture folders under [isu/blender/assets/human_texture](isu/blender/assets/human_texture) and add the required files:

```text
isu/blender/assets/human_texture/
├── female/
│   ├── 0white_female.png
│   └── 1black_female.png
└── male/
  ├── 0white_male.png
  └── 1black_male.png
```

The filenames must match the texture names expected by the Blender driver. The SMPL-X, AGORA, and third-party texture sources have separate licensing terms; see [License & Attributions](#license--attributions) before using them.

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

The generated result folder contains `json_inputs/`, `labels/`, `images/`, `depth/exr/`, `seg/`, `canny/`, `instance_seg/`, `critical_images/`, `sample_manifest.json`, `seg_class_map.json`, and `instance_seg_class_map.json`. Genetic-search result files and plots are written there as well.
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
