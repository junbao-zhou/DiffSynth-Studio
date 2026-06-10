#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, computed_field

from wan_robotwin_training_launcher import BaseLaunchConfig, run_training


DEFAULT_DATASET_BASE_PATH = Path("/workspace/group_share/adc-perception-xbrain/zhoujb4/dataset")

DEFAULT_DATASET_METADATA_PATH = (
    DEFAULT_DATASET_BASE_PATH
    # / "robotwin2.0_action_traj_videos/diffsynth_videogen/vace_metadata_20.csv"
    / "robotwin2.0_action_traj_videos/diffsynth_videogen/vace_metadata_all_prompts.csv"
)
VACE_MODEL_NAMES = (
    "Wan2.1-VACE-1.3B",
    "Wan2.1-VACE-14B",
)

DEFAULT_INFERENCE_DATASET_METADATA_PATH = (
    DEFAULT_DATASET_BASE_PATH
    / "WorldArena/WorldArena_Robotwin2.0_action_traj_videos/worldarena_robotwin2.0_vace_metadata_train_val_test.csv"
)

class LaunchConfig(BaseLaunchConfig):
    model_name: str = "Wan2.1-VACE-1.3B"
    dataset_base_path: Path = DEFAULT_DATASET_BASE_PATH
    dataset_metadata_path: Path = DEFAULT_DATASET_METADATA_PATH
    num_frames: int = Field(default=121, ge=1)
    data_file_keys: str = "video,vace_video,vace_reference_image"
    learning_rate: float = Field(default=5e-5, gt=0)
    remove_prefix_in_checkpoint: str = "pipe.vace."
    trainable_models: str = "vace"
    extra_inputs: str = "vace_video,vace_reference_image"
    use_gradient_checkpointing_offload: bool = True

    inference_dataset_base_path: Path | None = DEFAULT_DATASET_BASE_PATH
    inference_dataset_metadata_path: Path | None = DEFAULT_INFERENCE_DATASET_METADATA_PATH
    inference_num_samples: int = Field(default=4, ge=1)
    inference_data_file_keys: str = "vace_video,vace_reference_image"

    # inference_interval_steps: int = Field(default=2, ge=0)
    # save_steps: int = Field(default=2, ge=1)
    # max_checkpoints: int = Field(default=2, ge=1)

    @classmethod
    def allowed_model_names(cls) -> tuple[str, ...]:
        return VACE_MODEL_NAMES

    @computed_field
    @property
    def model_paths(self) -> str:
        if self.model_name == "Wan2.1-VACE-1.3B":
            model_paths = [
                f"{self.model_directory}/diffusion_pytorch_model.safetensors",
                f"{self.model_directory}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{self.model_directory}/Wan2.1_VAE.pth",
            ]
            return json.dumps(model_paths, separators=(",", ":"))

        if self.model_name == "Wan2.1-VACE-14B":
            diffusion_paths = [
                f"{self.model_directory}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ]
            model_paths = [
                diffusion_paths,
                f"{self.model_directory}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{self.model_directory}/Wan2.1_VAE.pth",
            ]
            return json.dumps(model_paths, separators=(",", ":"))

        raise ValueError(f"Unsupported VACE model_name: {self.model_name}")

    def build_accelerate_command(self) -> list[str]:
        command = super().build_accelerate_command()
        command.extend(
            [
                "--data_file_keys",
                self.data_file_keys,
                "--extra_inputs",
                self.extra_inputs,
            ]
        )
        if self.use_gradient_checkpointing_offload:
            command.append("--use_gradient_checkpointing_offload")
        return command


def main() -> None:
    run_training(LaunchConfig)


if __name__ == "__main__":
    main()
