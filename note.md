# Wan2.1 Training Launch Notes

## Summary

The RoboTwin2.0 Wan training launchers are not strict copies of the original bash scripts.
They intentionally unify the launch path and use DeepSpeed through
`examples/wanvideo/model_training/full/accelerate_config_14B.yaml`.

The user confirmed this is acceptable:

- `Wan2.1-T2V-1.3B` can also use DeepSpeed.
- The default `width=640` in the Python script is intentional.
- `Wan2.1-VACE-1.3B` has its own Python launcher.
- Common launcher logic is shared by both training scripts.
- Model paths are defined in each task launcher, not in the shared base launcher.
- Dataset paths are defined in each task launcher, not in the shared base launcher.
- VACE-only training arguments are appended in the VACE launcher, not in the shared base launcher.

## Original Bash Scripts

### `Wan2.1-T2V-1.3B.sh`

The 1.3B bash script launches training with plain `accelerate launch` options:

- `--num_processes 8`
- `--num_machines 1`
- `--mixed_precision bf16`
- `--dynamo_backend no`

It does not use `accelerate_config_14B.yaml`.

### `Wan2.1-T2V-14B.sh`

The 14B bash script launches training with:

- `--config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml`

So the 14B script uses the DeepSpeed accelerate config.

### `Wan2.1-VACE-1.3B.sh`

The VACE 1.3B bash script has VACE-specific training arguments:

- `--data_file_keys "video,vace_video,vace_reference_image"`
- `--num_frames 49`
- `--learning_rate 5e-5`
- `--remove_prefix_in_ckpt "pipe.vace."`
- `--trainable_models "vace"`
- `--extra_inputs "vace_video,vace_reference_image"`
- `--use_gradient_checkpointing_offload`

## `accelerate_config_14B.yaml`

The config enables DeepSpeed:

- `distributed_type: DEEPSPEED`
- `zero_stage: 2`
- `offload_optimizer_device: cpu`
- `offload_param_device: cpu`
- `mixed_precision: bf16`
- `num_processes: 8`

This means the config affects real training behavior.
It is not only a convenience wrapper.

## Python Script Behavior

Common launcher code lives in:

- `scripts/wan_robotwin_training_launcher.py`

The common launcher only owns shared launch behavior.
It does not define default model paths and does not append VACE-only arguments.
It also does not define default dataset paths.

The T2V launcher is:

- `scripts/train_wan2_1_t2v_robotwin2_0.py`

It defaults to:

- `model_name = "Wan2.1-T2V-1.3B"`
- `accelerate_config_path = "examples/wanvideo/model_training/full/accelerate_config_14B.yaml"`

The generated accelerate command always includes:

- `--config_file <accelerate_config_path>`
- `--deepspeed_multinode_launcher standard`

Therefore, by default, the T2V Python script runs `Wan2.1-T2V-1.3B` with the 14B DeepSpeed config.

The T2V Python script supports these model names:

- `Wan2.1-T2V-1.3B`
- `Wan2.1-T2V-14B`

The T2V launcher defines the model paths for each T2V model name directly.
It also defines its own `dataset_base_path` and `dataset_metadata_path`.

The VACE launcher is:

- `scripts/train_wan2_1_vace_robotwin2_0.py`

It defaults to:

- `model_name = "Wan2.1-VACE-1.3B"`
- `num_frames = 49`
- `learning_rate = 5e-5`
- `remove_prefix_in_checkpoint = "pipe.vace."`
- `trainable_models = "vace"`
- `data_file_keys = "video,vace_video,vace_reference_image"`
- `extra_inputs = "vace_video,vace_reference_image"`
- `use_gradient_checkpointing_offload = True`
- `inference_interval_steps = 0`

The VACE launcher defines its own model paths and appends:

- `--data_file_keys`
- `--extra_inputs`
- `--use_gradient_checkpointing_offload`

It also defines its own `dataset_base_path` and `dataset_metadata_path`.

## Final Decision

This is accepted as intentional behavior.

The scripts should be read as custom RoboTwin2.0 training launchers, not as exact reproductions of the original example bash scripts.

Important intentional differences:

- 1.3B also uses DeepSpeed.
- VACE 1.3B also uses DeepSpeed in this launcher.
- The default width is `640`, not the bash script value `832`.
- The script adds logging, code backup, periodic inference, and custom output paths.

## Practical Note

If future training should exactly match the original 1.3B bash script, then the Python launcher would need a non-DeepSpeed mode.

But for the current RoboTwin2.0 setup, unified DeepSpeed launch is considered fine.

## Wan2.1 VACE Training Inputs

Conclusion:

- `vace_reference_image` is the first frame condition.
- The training is not only conditioned on the first frame.
- It also uses the full `vace_video` as the VACE control video.

Details:

- `Wan2.1-VACE-1.3B.sh` reads three data columns:
  `video,vace_video,vace_reference_image`.
- The same script passes `vace_video,vace_reference_image` as `extra_inputs`.
- In `examples/wanvideo/model_training/train.py`,
  `vace_reference_image` is converted to `data["vace_reference_image"][0]`,
  so it is used as one image.
- The example dataset was checked:
  `reference_image.png` and the first frame of `video1.mp4` are pixel-identical.
  The mean absolute difference is `0`.
- `vace_video` is the full control video.
  It enters the VACE unit and is encoded into `vace_context`.

Example raw file shapes:

- target `video1.mp4`: `1920x1080`, `176` frames
- control `video1_softedge.mp4`: `1920x1088`, `176` frames

The raw target video and raw control video are not exactly the same shape.
But the training loader normalizes them to the script settings:

- `height = 480`
- `width = 832`
- `num_frames = 49`

`video`, `vace_video`, and `vace_reference_image` are all included in
`data_file_keys`.
So they all pass through the same `default_video_operator`, then through
`ImageCropAndResize`.

Final training shapes:

- target video: `49 x 480 x 832`
- control `vace_video`: `49 x 480 x 832`
- `vace_reference_image`: `1 x 480 x 832`

Important note:

- The code resizes/crops each file separately.
- It does not explicitly assert that the raw source shapes are the same.
- The important requirement is that after the loader, frame count and spatial
  size are aligned.

## RoboTwin2.0 `seen` / `unseen` Instructions

RoboTwin2.0 stores language instructions for each trajectory under the
`instructions/` directory.
The matching head-camera video is stored under the `video/` directory.

Each instruction JSON has this shape:

```json
{
  "seen": ["..."],
  "unseen": ["..."]
}
```

In this repository, `seen` and `unseen` should be understood as two prompt pools
for the same episode.
They are language variations, not different videos and not different tasks.

- `seen`: regular instruction wording.
- `unseen`: alternative instruction wording for language generalization.

RoboTwin2.0 treats diverse language instructions as one part of domain
randomization.
So `unseen` is mainly useful for testing or improving robustness to new wording.

Example from local `adjust_bottle/.../instructions/episode0.json`:

```text
seen:
- Pick the bottle with ridges near base head-up using the right arm
- Lift the medium green bottle with capped neck from the table without mentioning the arm.

unseen:
- Pick up the green plastic bottle with ridged bottom using the left arm in an upright position
- Pick up the green plastic bottle with ridged bottom using the correct arm
```

For `scripts/build_robotwin_videogen_dataset.py`:

- The script uses all prompts from both `seen` and `unseen`.
- It does not randomly sample prompts.
- It does not expose `--prompt-source`.
- It does not expose `--samples-per-episode`, `--seed`, `--tasks`,
  `--configs`, or `--limit`.
- One video appears multiple times in the output CSV, once per prompt.
- The default output path is:
  `RoboTwin2.0/diffsynth_videogen/metadata_all_prompts.csv`.

This is the intended setup for video generation training:
the same target video is repeated with multiple language descriptions, so the
model sees all available prompt variants for that video.

Current local dataset check:

- sample instruction file:
  `beat_block_hammer/franka_clean_50/franka_clean_50/instructions/episode1.json`
- `seen = 100`
- `unseen = 100`
- total prompts per video = `200`
- `metadata.csv` rows = `51,300`
- `metadata_all_prompts.csv` rows = `10,260,000`
- row ratio = `10,260,000 / 51,300 = 200`
- every video in `metadata_all_prompts.csv` appears exactly `200` times.

The script uses multiprocessing for reading instruction JSON files.
Worker processes only return rows.
The main process is the only process that opens and writes the CSV, so workers
do not overwrite each other.

The script is fail-fast for bad per-episode data:

- missing `instructions/` directory: skip that episode directory during scanning
- missing `episodeN.json`: raise `FileNotFoundError`
- empty `seen + unseen` prompt list: raise `ValueError`
- broken JSON: raise the original JSON error

## WanVideo Training Code Map

Summary:

- `examples/wanvideo/model_training/train.py`: training entry.
- `diffsynth/diffusion/parsers.py`: command-line args and input config.
- `diffsynth/diffusion/runner.py`: main training loop. It can run inference every fixed number of steps.

Details:

### `examples/wanvideo/model_training/train.py`

Role:

- WanVideo training entry file.
- Builds the parser with `wan_parser()`.
- Creates `accelerate.Accelerator`.
- Builds `UnifiedDataset`.
- Builds `WanTrainingModule`.
- Builds `ModelLogger`.
- Selects the launcher by `args.task`.
- Calls `launch_training_task` or `launch_data_process_task`.

Important points:

- `wan_parser()` reuses common args from `parsers.py`:
  `add_general_config(parser)` and `add_video_size_config(parser)`.
- It adds Wan-specific args, such as:
  `--tokenizer_path`, `--audio_processor_path`,
  `--max_timestep_boundary`, `--min_timestep_boundary`,
  `--initialize_model_on_cpu`, `--framewise_decoding`.
- It also adds:
  `--inference_interval_steps`.
- `build_inference_dataset_from_training_prompts()` uses the first few training
  prompts to build a small inference set.
- This inference set is passed into `launcher(..., inference_dataset=inference_dataset)`.

### `diffsynth/diffusion/parsers.py`

Role:

- Stores shared command-line config.
- Defines common training, dataset, model, output, LoRA, and gradient args.
- `train.py` imports these helper functions through `from diffsynth.diffusion import *`.

Main config groups:

- `add_dataset_base_config`: dataset path, metadata path, repeat, workers,
  data file keys.
- `add_video_size_config`: height, width, max pixels, number of frames.
- `add_model_config`: model paths, extra inputs, FP8 models, offload models.
- `add_training_config`: learning rate, epochs, trainable models, task.
- `add_output_config`: output path, log dir, checkpoint save steps.
- `add_lora_config`: LoRA base model, target modules, rank, checkpoints.
- `add_gradient_config`: gradient checkpointing and accumulation.
- `add_general_config`: combines the common config groups.

### `diffsynth/diffusion/runner.py`

Role:

- Runs the real training loop.
- Handles optimizer, scheduler, dataloader, accelerator prepare, checkpoint
  saving, and optional periodic inference.

Training flow:

1. `launch_training_task(...)` reads training args.
2. It creates `AdamW`, `ConstantLR`, and `DataLoader`.
3. It moves model to the accelerator device.
4. It wraps model, optimizer, dataloader, and scheduler with
   `accelerator.prepare(...)`.
5. It loops over epochs and batches.
6. For each batch:
   - run model forward
   - backprop with `accelerator.backward(loss)`
   - optimizer step
   - scheduler step
   - zero grad
   - call `model_logger.on_step_end(...)`

Periodic inference:

- Controlled by `inference_interval_steps`.
- In `train.py`, this value comes from `--inference_interval_steps`.
- In `runner.py`, after each training step:
  if `training_step % inference_interval_steps == 0`, it runs inference.
- Inference output is saved under:
  `output_path/inference/step-XXXXXX/`.
- Only the main process runs inference and writes videos.
- `accelerator.wait_for_everyone()` is called before and after inference.

Inference helper functions:

- `inference(...)`: switches the model to eval mode, copies the pipeline for
  inference, runs `pipe(**inference_inputs)`, then restores training mode.
- `inference_and_save(...)`: saves generated videos as mp4 files.
- `inference_and_save_on_main_process(...)`: unwraps the model and runs
  inference only on the main process.
