#!/usr/bin/env python3

import json
from pathlib import Path

from pydantic import Field, computed_field

from wan_robotwin_training_launcher import BaseLaunchConfig, run_training


DEFAULT_DATASET_BASE_PATH = Path("/workspace/group_share/adc-perception-xbrain/zhoujb4/dataset/RoboTwin2.0")
DEFAULT_DATASET_METADATA_PATH = DEFAULT_DATASET_BASE_PATH / "diffsynth_videogen/metadata.csv"
T2V_MODEL_NAMES = (
    "Wan2.1-T2V-1.3B",
    "Wan2.1-T2V-14B",
)


class LaunchConfig(BaseLaunchConfig):
    model_name: str = "Wan2.1-T2V-1.3B"
    dataset_base_path: Path = DEFAULT_DATASET_BASE_PATH
    dataset_metadata_path: Path = DEFAULT_DATASET_METADATA_PATH

    inference_dataset_base_path: Path | None = DEFAULT_DATASET_BASE_PATH
    inference_dataset_metadata_path: Path | None = DEFAULT_DATASET_METADATA_PATH

    # inference_interval_steps: int = Field(default=2, ge=0)
    # save_steps: int = Field(default=2, ge=1)
    # max_checkpoints: int = Field(default=2, ge=1)

    @classmethod
    def allowed_model_names(cls) -> tuple[str, ...]:
        return T2V_MODEL_NAMES

    @computed_field
    @property
    def model_paths(self) -> str:
        if self.model_name == "Wan2.1-T2V-1.3B":
            model_paths = [
                f"{self.model_directory}/diffusion_pytorch_model.safetensors",
                f"{self.model_directory}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{self.model_directory}/Wan2.1_VAE.pth",
            ]
            return json.dumps(model_paths, separators=(",", ":"))

        if self.model_name == "Wan2.1-T2V-14B":
            diffusion_paths = [
                f"{self.model_directory}/diffusion_pytorch_model-00001-of-00006.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00002-of-00006.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00003-of-00006.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00004-of-00006.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00005-of-00006.safetensors",
                f"{self.model_directory}/diffusion_pytorch_model-00006-of-00006.safetensors",
            ]
            model_paths = [
                diffusion_paths,
                f"{self.model_directory}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{self.model_directory}/Wan2.1_VAE.pth",
            ]
            return json.dumps(model_paths, separators=(",", ":"))

        raise ValueError(f"Unsupported T2V model_name: {self.model_name}")


def main() -> None:
    run_training(LaunchConfig)


if __name__ == "__main__":
    main()
