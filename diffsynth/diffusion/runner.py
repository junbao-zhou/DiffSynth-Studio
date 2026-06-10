import copy
import os, torch
from pathlib import Path

from tqdm import tqdm
from accelerate import Accelerator
from diffsynth.utils.data import save_video
from diffsynth.utils.logger import logger
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _copy_pipe_for_inference(pipe):
    inference_pipe = copy.copy(pipe)
    inference_pipe.scheduler = copy.deepcopy(pipe.scheduler)
    return inference_pipe


def summarize_data_value(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if value.__class__.__name__ == "Image" and hasattr(value, "size"):
        width, height = value.size
        return {
            "type": "Image",
            "shape": (height, width),
        }
    if isinstance(value, list):
        summary = {
            "type": "list",
            "length": len(value),
        }
        if value and value[0].__class__.__name__ == "Image" and hasattr(value[0], "size"):
            width, height = value[0].size
            summary["type"] = "video"
            summary["shape"] = (len(value), height, width)
        return summary
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "type": type(value).__name__,
    }


def summarize_data_mapping(data):
    return {
        name: summarize_data_value(value)
        for name, value in data.items()
    }


def build_inference_pipe_kwargs(args):
    if args is None:
        return {}

    inference_pipe_kwargs = {
        "height": args.inference_height if args.inference_height is not None else args.height,
        "width": args.inference_width if args.inference_width is not None else args.width,
        "num_frames": args.inference_num_frames if args.inference_num_frames is not None else args.num_frames,
        "seed": args.inference_seed,
        "num_inference_steps": args.inference_num_inference_steps,
        "tiled": args.inference_tiled,
        "framewise_decoding": args.framewise_decoding,
    }
    if args.inference_negative_prompt is not None:
        inference_pipe_kwargs["negative_prompt"] = args.inference_negative_prompt
    return inference_pipe_kwargs


def inference(
    model: DiffusionTrainingModule,
    inference_dataset,
    args=None,
) -> list[tuple[int, str, list]]:
    was_training = model.training
    inference_pipe = _copy_pipe_for_inference(model.pipe)
    inference_pipe_kwargs = build_inference_pipe_kwargs(args)
    videos = []
    model.eval()
    for inference_id in range(len(inference_dataset)):
        inference_data = inference_dataset[inference_id]
        if not isinstance(inference_data, dict):
            raise TypeError(
                "Each inference dataset item must be a dict passed to pipe.__call__."
            )
        inference_inputs = dict(inference_data)
        inference_inputs.pop("video", None)
        inference_inputs.update(inference_pipe_kwargs)
        logger.info(
            f"Start inference {inference_id = } / {len(inference_dataset)} | "
            f"input_keys={list(inference_inputs.keys())} | "
            f"input_summary={summarize_data_mapping(inference_inputs)}"
        )
        videos.append(
            (
                inference_id,
                inference_data.get("prompt", ""),
                inference_pipe(**inference_inputs),
            )
        )
    model.train(was_training)
    return videos


def inference_and_save(
    model: DiffusionTrainingModule,
    inference_dataset,
    save_dir: Path | str,
    fps: float = 15,
    quality: int = 9,
    args=None,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Start inference and save videos to {save_dir}")
    videos = inference(
        model=model,
        inference_dataset=inference_dataset,
        args=args,
    )
    for inference_id, prompt, video in videos:
        save_path = save_dir / f"inference-{inference_id:06d}-{prompt[:50].replace(' ', '_')}.mp4"
        save_video(video, str(save_path), fps=fps, quality=quality)
        logger.info(f"Saved inference {inference_id = } video to {save_path}")
    logger.info(f"Finished inference and save videos to {save_dir}")


def run_on_main_process_after_everyone(
    accelerator: Accelerator,
    function,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        function()
    accelerator.wait_for_everyone()


def inference_and_save_on_main_process(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    inference_dataset,
    save_dir: Path | str,
    fps: float = 15,
    quality: int = 9,
    args=None,
) -> None:
    def _inference_and_save() -> None:
        inference_and_save(
            model=accelerator.unwrap_model(model),
            inference_dataset=inference_dataset,
            save_dir=save_dir,
            fps=fps,
            quality=quality,
            args=args,
        )

    run_on_main_process_after_everyone(
        accelerator=accelerator,
        function=_inference_and_save,
    )


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    inference_interval_steps: int = 0,
    inference_dataset: torch.utils.data.Dataset = None,
    inference_fps: float = 15,
    inference_quality: int = 9,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        inference_interval_steps = args.inference_interval_steps

    if inference_interval_steps > 0 and inference_dataset is None:
        logger.info(
            "Periodic inference is disabled because inference_dataset is not provided."
        )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    logger.info(
        f"Start training: {num_epochs = }, {len(dataloader) = }, "
        f"{learning_rate = }, {weight_decay = }"
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        logger.info(f"Start epoch {epoch_id = } / {num_epochs}")
        for data_id, data in enumerate(tqdm(dataloader)):
            logger.info(
                f"Start data {data_id = } / {len(dataloader)} | "
                f"{list(data.keys()) = } | "
                f"{data.get('prompt', '')[:100] = } | "
                f"data_summary={summarize_data_mapping(data)}"
            )
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
            training_step = model_logger.num_steps
            if (
                inference_interval_steps > 0
                and inference_dataset is not None
                and training_step % inference_interval_steps == 0
            ):
                inference_save_dir = (
                    Path(model_logger.output_path)
                    / "inference"
                    / f"step-{training_step:06d}"
                )
                inference_and_save_on_main_process(
                    accelerator=accelerator,
                    model=model,
                    inference_dataset=inference_dataset,
                    save_dir=inference_save_dir,
                    fps=inference_fps,
                    quality=inference_quality,
                    args=args,
                )
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        logger.info(f"Finished epoch {epoch_id = } / {num_epochs}")
    model_logger.on_training_end(accelerator, model, save_steps)
    logger.info("Finished training")


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    num_batches = len(dataloader)
    logger.info(f"Start data process: num_batches={num_batches}")
    
    for data_id, data in enumerate(tqdm(dataloader)):
        logger.info(f"Start data process item {data_id + 1}/{num_batches}")
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
