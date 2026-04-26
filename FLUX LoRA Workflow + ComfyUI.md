# FLUX LoRA Workflow + ComfyUI

This document captures the full workflow used to:

- prepare a personal image dataset for FLUX LoRA training
- fine-tune a FLUX.1-dev LoRA with `diffusers`
- configure ComfyUI to test the trained LoRA

This was done in:

- `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning`

## Dataset Preparation

### Source Images

The original dataset lives under:

- `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/FLUX.1 Dreambooth LoRA Fine Tuning Dataset`

The initial audit found several issues:

- mixed image and non-image files
- inconsistent extensions
- missing captions
- many images depending on EXIF rotation
- some low-value or ambiguous training samples

### Cleanup Script

Dataset cleanup was automated with:

- [prepare_lora_dataset.py](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/prepare_lora_dataset.py:1)

That script:

- normalized filenames
- converted prepared images to `.jpg`
- applied EXIF rotation and rewrote images without orientation dependency
- created a conservative `train/` and `review/` split
- wrote a manifest and summary

### Prepared Dataset

Prepared dataset root:

- `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset`

Prepared outputs:

- train set: `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset/train`
- review set: `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset/review`
- manifest: [manifest.csv](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset/manifest.csv:1)
- summary: [SUMMARY.md](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset/SUMMARY.md:1)

Current counts:

- `13` train-ready images
- `19` review images

## Captions

Caption sidecars were generated with:

- [generate_caption_sidecars.py](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/generate_caption_sidecars.py:1)

The trigger token was updated to:

- `Tyrel Barstow`

The class name is:

- `person`

Example caption pattern:

- `portrait photo of Tyrel Barstow person`

To regenerate captions:

```bash
python3 scripts/generate_caption_sidecars.py --overwrite
```

## Diffusers Dataset Build

The training dataset consumed by the FLUX DreamBooth script was rebuilt into a caption-aware imagefolder format with:

- [build_flux_imagefolder_dataset.py](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/build_flux_imagefolder_dataset.py:1)

Output:

- `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset/diffusers-train`

Important file:

- [metadata.jsonl](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/FLUX.1-Dreambooth-Dataset/prepared-lora-dataset/diffusers-train/metadata.jsonl:1)

## Training Environment

The training environment was created with:

- [setup_flux_training_env.sh](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/setup_flux_training_env.sh:1)

That setup created:

- `.venv-flux`
- `vendor/diffusers`

To recreate it:

```bash
cd /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning
bash scripts/setup_flux_training_env.sh
source .venv-flux/bin/activate
```

## Hugging Face Model Download

The base model used for training is:

- `black-forest-labs/FLUX.1-dev`

The full diffusers snapshot was downloaded with:

- [download_flux_diffusers_snapshot.sh](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/download_flux_diffusers_snapshot.sh:1)

Download target:

- `/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/models/FLUX.1-dev-diffusers`

Command:

```bash
cd /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning
source .venv-flux/bin/activate
bash scripts/download_flux_diffusers_snapshot.sh
```

Important note:

- the training shell needed `HF_TOKEN`
- in this environment the token was present in interactive shells, but not in non-interactive shells because `.bashrc` returned early before the export line

## Fine-Tuning

Training was launched with:

- [train_flux_lora.sh](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/train_flux_lora.sh:1)

Default run characteristics:

- base model: `models/FLUX.1-dev-diffusers`
- dataset: prepared `train/` set
- caption source: `metadata.jsonl`
- trigger phrase: `Tyrel Barstow person`
- output dir: `output/tyrel-barstow-flux-lora`
- rank: `16`
- LoRA alpha: `16`
- max steps: `500`

Launch command:

```bash
cd /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning
source .venv-flux/bin/activate
MODEL_DIR=~/FLUX.1-Dreambooth-LoRA-Fine-Tuning/models/FLUX.1-dev-diffusers \
  bash scripts/train_flux_lora.sh
```

## Training Outputs

The training run completed successfully.

Final LoRA:

- [pytorch_lora_weights.safetensors](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/output/tyrel-barstow-flux-lora/pytorch_lora_weights.safetensors:1)

Saved checkpoints:

- [checkpoint-300](</home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/output/tyrel-barstow-flux-lora/checkpoint-300>)
- [checkpoint-400](</home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/output/tyrel-barstow-flux-lora/checkpoint-400>)
- [checkpoint-500](</home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/output/tyrel-barstow-flux-lora/checkpoint-500>)

Approximate wall-clock training time:

- `6 hours 53 minutes`

## Testing In Python

The repo also includes:

- [test_flux_lora.py](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/test_flux_lora.py:1)

Example:

```bash
cd /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning
source .venv-flux/bin/activate
python scripts/test_flux_lora.py
```

## ComfyUI Setup

### Why Additional Setup Was Needed

The FLUX base model was downloaded in Hugging Face diffusers format.

ComfyUI on this machine did not directly accept the Hugging Face shard index files in:

- `UNETLoader`
- `DualCLIPLoader`

The dropdowns only exposed loadable `.safetensors` files, not `.index.json`.

### Merging FLUX Shards For ComfyUI

To make the FLUX transformer and T5 encoder usable in ComfyUI, the shards were merged with:

- [merge_flux_shards_for_comfyui.py](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/merge_flux_shards_for_comfyui.py:1)

This created:

- `/home/tyrel/ComfyUI/models/unet/FLUX.1-dev-transformer-merged.safetensors`
- `/home/tyrel/ComfyUI/models/clip/FLUX.1-dev-t5xxl-merged.safetensors`

Command:

```bash
cd /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning
source .venv-flux/bin/activate
python scripts/merge_flux_shards_for_comfyui.py
```

### LoRA Entries Added To ComfyUI

Separate LoRA entries were linked into `ComfyUI/models/loras`:

- `tyrel-barstow-flux-lora-step300.safetensors`
- `tyrel-barstow-flux-lora-step400.safetensors`
- `tyrel-barstow-flux-lora-step500.safetensors`
- `tyrel-barstow-flux-lora-final.safetensors`

These map to the actual checkpoint files in:

- `output/tyrel-barstow-flux-lora/checkpoint-300`
- `output/tyrel-barstow-flux-lora/checkpoint-400`
- `output/tyrel-barstow-flux-lora/checkpoint-500`
- `output/tyrel-barstow-flux-lora/`

## ComfyUI Workflows

ComfyUI notes live in:

- [comfyui/README.md](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/comfyui/README.md:1)

Single-LoRA workflow:

- [tyrel_barstow_flux_lora_workflow_api.json](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/comfyui/tyrel_barstow_flux_lora_workflow_api.json:1)

Checkpoint comparison workflow:

- [tyrel_barstow_flux_lora_compare_300_400_500_api.json](/home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/comfyui/tyrel_barstow_flux_lora_compare_300_400_500_api.json:1)

The comparison workflow runs the same prompt, seed, sampler, and latent through:

- checkpoint `300`
- checkpoint `400`
- checkpoint `500`

That makes it easy to see whether the earlier checkpoint is better than the final one.

## Recommended Demo Flow

1. Show the original dataset folder.
2. Show the prepared `train/` and `review/` split.
3. Show the caption sidecars using `Tyrel Barstow person`.
4. Show the training script and the output checkpoints.
5. Explain why checkpoint comparison matters.
6. Open the ComfyUI comparison workflow.
7. Generate one prompt across `300`, `400`, and `500`.
8. Compare likeness, flexibility, and overfitting.

## Rebuild Commands

Rebuild prepared dataset:

```bash
python3 scripts/prepare_lora_dataset.py
```

Regenerate captions:

```bash
python3 scripts/generate_caption_sidecars.py --overwrite
```

Rebuild diffusers dataset:

```bash
python3 scripts/build_flux_imagefolder_dataset.py
```

Re-merge FLUX weights for ComfyUI:

```bash
source /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/.venv-flux/bin/activate
python /home/tyrel/FLUX.1-Dreambooth-LoRA-Fine-Tuning/scripts/merge_flux_shards_for_comfyui.py
```
