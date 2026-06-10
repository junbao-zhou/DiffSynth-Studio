import os, re, torch
from pathlib import Path
from accelerate import Accelerator


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x:x,
        max_checkpoints: int = 5,
    ):
        if max_checkpoints < 1:
            raise ValueError(f"max_checkpoints must be >= 1, got: {max_checkpoints}")
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.max_checkpoints = max_checkpoints
        self.num_steps = 0

    def _list_checkpoints(self) -> list[Path]:
        output_path = Path(self.output_path)
        if not output_path.exists():
            return []
        checkpoint_pattern = re.compile(r"^(step|epoch)-\d+\.safetensors$")
        return [
            path
            for path in output_path.iterdir()
            if path.is_file() and checkpoint_pattern.match(path.name)
        ]

    def _remove_old_checkpoints(self) -> None:
        checkpoints = sorted(
            self._list_checkpoints(),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        checkpoints_to_remove = checkpoints[:-self.max_checkpoints]
        for checkpoint_path in checkpoints_to_remove:
            checkpoint_path.unlink()


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            self._remove_old_checkpoints()


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            self._remove_old_checkpoints()
