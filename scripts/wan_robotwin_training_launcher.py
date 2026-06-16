#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
from typing import Optional

import copy_codes
from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliSettingsSource, CliSuppress, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCELERATE_CONFIG_PATH = Path("examples/wanvideo/model_training/full/accelerate_config_14B.yaml")
DEFAULT_TRAIN_SCRIPT_PATH = Path("examples/wanvideo/model_training/train.py")
DEFAULT_OUTPUT_ROOT = Path("./outputs/train/robotwin2.0")
WAN_INFERENCE_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def get_environment_variable(
    name: str,
    default: str = "",
) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def parse_non_negative_integer_from_environment(
    name: str,
    default: str,
) -> int:
    raw_value = get_environment_variable(name, default)
    if not raw_value.isdigit():
        raise SystemExit(f"{name} must be a non-negative integer, got: {raw_value}")
    return int(raw_value)


def parse_positive_integer_from_environment(
    name: str,
    default: str,
) -> int:
    raw_value = get_environment_variable(name, default)
    if not raw_value.isdigit():
        raise SystemExit(f"{name} must be a positive integer, got: {raw_value}")
    value = int(raw_value)
    if value < 1:
        raise SystemExit(f"{name} must be >= 1, got: {value}")
    return value


def machine_rank_from_environment() -> int:
    return parse_non_negative_integer_from_environment("NODE_RANK", "0")


def master_address_from_environment() -> str:
    return get_environment_variable("MASTER_ADDR", "127.0.0.1")


def master_port_from_environment() -> int:
    return parse_positive_integer_from_environment("MASTER_PORT", "29500")


class BaseLaunchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",
    )

    save_id: str = "test"
    model_name: str
    dataset_base_path: Path
    dataset_metadata_path: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    output_path: Optional[Path] = None
    log_dir: Optional[Path] = None
    height: int = Field(default=480, ge=1)
    width: int = Field(default=640, ge=1)
    num_frames: int = Field(default=81, ge=1)
    dataset_repeat: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0)
    num_epochs: int = Field(default=2, ge=1)
    remove_prefix_in_checkpoint: str = "pipe.dit."
    trainable_models: str = "dit"
    inference_interval_steps: int = Field(default=200, ge=0)
    inference_dataset_base_path: Optional[Path] = None
    inference_dataset_metadata_path: Optional[Path] = None
    inference_data_file_keys: str = ""
    inference_num_samples: int = Field(default=3, ge=1)
    inference_height: Optional[int] = Field(default=None, ge=1)
    inference_width: Optional[int] = Field(default=None, ge=1)
    inference_num_frames: Optional[int] = Field(default=None, ge=1)
    inference_max_pixels: Optional[int] = Field(default=None, ge=1)
    inference_negative_prompt: str = WAN_INFERENCE_NEGATIVE_PROMPT
    inference_seed: int = Field(default=0, ge=0)
    inference_num_inference_steps: int = Field(default=50, ge=1)
    inference_tiled: bool = True
    save_steps: int = Field(default=200, ge=1)
    max_checkpoints: int = Field(default=5, ge=1)
    accelerate_config_path: Path = DEFAULT_ACCELERATE_CONFIG_PATH
    train_script_path: Path = DEFAULT_TRAIN_SCRIPT_PATH
    num_machines: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=8, ge=1)
    machine_rank: CliSuppress[int] = Field(default_factory=machine_rank_from_environment)
    master_address: CliSuppress[str] = Field(default_factory=master_address_from_environment)
    master_port: CliSuppress[int] = Field(default_factory=master_port_from_environment)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_kebab_case=False,
                cli_avoid_json=True,
                cli_hide_none_type=True,
            ),
            init_settings,
        )

    @classmethod
    def allowed_model_names(cls) -> tuple[str, ...]:
        return ()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        allowed_model_names = cls.allowed_model_names()
        if allowed_model_names and model_name not in allowed_model_names:
            raise ValueError(
                f"model_name must be one of {allowed_model_names}, got: {model_name}"
            )
        return model_name

    @field_validator("machine_rank")
    @classmethod
    def validate_machine_rank(cls, machine_rank: int) -> int:
        if machine_rank < 0:
            raise ValueError(f"machine_rank must be >= 0, got: {machine_rank}")
        return machine_rank

    @field_validator("master_port")
    @classmethod
    def validate_master_port(cls, master_port: int) -> int:
        if master_port < 1:
            raise ValueError(f"master_port must be >= 1, got: {master_port}")
        return master_port

    @model_validator(mode="after")
    def fill_derived_paths(self):
        if self.output_path is None:
            self.output_path = self.output_root / f"{self.model_name}_full" / self.save_id
        if self.log_dir is None:
            self.log_dir = self.output_path / "logs"
        if self.inference_height is None:
            self.inference_height = self.height
        if self.inference_width is None:
            self.inference_width = self.width
        if self.inference_num_frames is None:
            self.inference_num_frames = self.num_frames
        if self.inference_max_pixels is None:
            self.inference_max_pixels = self.height * self.width
        return self

    @computed_field
    @property
    def model_directory(self) -> str:
        return f"Wan-AI/{self.model_name}"

    @computed_field
    @property
    def tokenizer_path(self) -> str:
        return f"{self.model_directory}/google/umt5-xxl"

    @computed_field
    @property
    def num_processes(self) -> int:
        return self.num_machines * self.gpus_per_node

    def validate_distributed_config(self) -> None:
        if self.machine_rank >= self.num_machines:
            raise SystemExit(
                "MACHINE_RANK must be < NUM_MACHINES, "
                f"got MACHINE_RANK={self.machine_rank}, NUM_MACHINES={self.num_machines}"
            )

    def build_accelerate_command(self) -> list[str]:
        command = [
            "accelerate",
            "launch",
            "--config_file",
            str(self.accelerate_config_path),
            "--num_machines",
            str(self.num_machines),
            "--num_processes",
            str(self.num_processes),
            "--machine_rank",
            str(self.machine_rank),
            "--main_process_ip",
            self.master_address,
            "--main_process_port",
            str(self.master_port),
            "--deepspeed_multinode_launcher",
            "standard",
            str(self.train_script_path),
            "--dataset_base_path",
            str(self.dataset_base_path),
            "--dataset_metadata_path",
            str(self.dataset_metadata_path),
            "--height",
            str(self.height),
            "--width",
            str(self.width),
            "--num_frames",
            str(self.num_frames),
            "--dataset_repeat",
            str(self.dataset_repeat),
            "--model_paths",
            self.model_paths,
            "--tokenizer_path",
            self.tokenizer_path,
            "--learning_rate",
            str(self.learning_rate),
            "--num_epochs",
            str(self.num_epochs),
            "--remove_prefix_in_ckpt",
            self.remove_prefix_in_checkpoint,
            "--output_path",
            str(self.output_path),
            "--log_dir",
            str(self.log_dir),
            "--trainable_models",
            self.trainable_models,
            "--inference_interval_steps",
            str(self.inference_interval_steps),
            "--save_steps",
            str(self.save_steps),
            "--max_checkpoints",
            str(self.max_checkpoints),
        ]
        if self.inference_dataset_base_path is not None:
            command.extend(
                [
                    "--inference_dataset_base_path",
                    str(self.inference_dataset_base_path),
                ]
            )
        if self.inference_dataset_metadata_path is not None:
            command.extend(
                [
                    "--inference_dataset_metadata_path",
                    str(self.inference_dataset_metadata_path),
                ]
            )
        if self.inference_data_file_keys != "":
            command.extend(
                [
                    "--inference_data_file_keys",
                    self.inference_data_file_keys,
                ]
            )
        command.extend(
            [
                "--inference_num_samples",
                str(self.inference_num_samples),
                "--inference_height",
                str(self.inference_height),
                "--inference_width",
                str(self.inference_width),
                "--inference_num_frames",
                str(self.inference_num_frames),
                "--inference_max_pixels",
                str(self.inference_max_pixels),
                "--inference_negative_prompt",
                self.inference_negative_prompt,
                "--inference_seed",
                str(self.inference_seed),
                "--inference_num_inference_steps",
                str(self.inference_num_inference_steps),
            ]
        )
        if not self.inference_tiled:
            command.append("--no_inference_tiled")
        return command


def print_launch_config(
    launch_config: BaseLaunchConfig,
) -> None:
    for key, value in launch_config.model_dump().items():
        print(f"{key} : {value}", flush=True)


def configure_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"


def backup_code_for_rank_zero(
    launch_config: BaseLaunchConfig,
) -> None:
    if launch_config.machine_rank != 0:
        return

    code_backup_dir = Path(launch_config.log_dir) / "codes"
    copy_codes.copy_local(
        copy_codes.collect_files(
            PROJECT_ROOT,
            copy_codes.build_matcher(copy_codes.load_gitignore(PROJECT_ROOT)),
            code_only=True,
            include_git=False,
        ),
        PROJECT_ROOT,
        code_backup_dir,
        overwrite=True,
    )
    print(f"Code backup saved to {code_backup_dir}", flush=True)


def run_training(
    launch_config_class: type[BaseLaunchConfig],
) -> None:
    launch_config = launch_config_class()
    launch_config.validate_distributed_config()
    print_launch_config(launch_config)
    backup_code_for_rank_zero(launch_config)
    configure_offline_environment()
    command = launch_config.build_accelerate_command()
    print(
        f"""
[launch] accelerate command:
{ " ".join(command) }
""",
        flush=True,
    )
    subprocess.run(command, check=True)
