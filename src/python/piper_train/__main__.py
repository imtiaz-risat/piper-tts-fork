import argparse
import json
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


class HFCheckpointUploader(ModelCheckpoint):
    def __init__(self, hf_repo_id: str, hf_token: str, session_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.session_id = session_id
        self._api = None

    def _ensure_api(self):
        """Ensure HuggingFace API is initialized."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()

    def _upload_last(self):
        """Upload last checkpoint to HuggingFace."""
        last_ckpt_path = getattr(self, "last_model_path", None)
        if not last_ckpt_path:
            dirpath = getattr(self, "dirpath", None)
            if dirpath and Path(dirpath).exists():
                try:
                    ckpts = sorted(
                        Path(dirpath).glob("*.ckpt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if ckpts:
                        last_ckpt_path = str(ckpts[0])
                except Exception:
                    last_ckpt_path = None
        if last_ckpt_path and Path(last_ckpt_path).exists():
            try:
                self._ensure_api()
                self._api.create_repo(
                    repo_id=self.hf_repo_id,
                    token=self.hf_token,
                    exist_ok=True,
                )
                filename = Path(last_ckpt_path).name
                path_in_repo = (
                    f"{self.session_id}/{filename}" if self.session_id else filename
                )
                self._api.upload_file(
                    path_or_fileobj=last_ckpt_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.hf_repo_id,
                    token=self.hf_token,
                )
                _LOGGER.info("Uploaded checkpoint to HuggingFace: %s", last_ckpt_path)
            except Exception as e:
                _LOGGER.error("HuggingFace upload failed: %s", e)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Upload last checkpoint on save checkpoint."""
        self._upload_last()

    def on_train_epoch_end(self, trainer, pl_module):
        """Upload last checkpoint on train epoch end."""
        super().on_train_epoch_end(trainer, pl_module)
        # self._upload_last()

    def on_validation_end(self, trainer, pl_module):
        """Upload last checkpoint on validation end."""
        super().on_validation_end(trainer, pl_module)
        # self._upload_last()


class KeepLastKCheckpoints(pl.Callback):
    def __init__(self, k: int):
        self.k = k

    def _prune(self, dirpath):
        if not dirpath:
            return
        p = Path(dirpath)
        if not p.exists():
            return
        ckpts = sorted(
            p.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True
        )
        for old in ckpts[self.k:]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                self._prune(getattr(cb, "dirpath", None))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not trainer.is_global_zero:
            return
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                self._prune(getattr(cb, "dirpath", None))


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--push-to-hf", action="store_true", help="Enable HF upload")
    parser.add_argument(
        "--hf-repo-id", help="HuggingFace repo ID, e.g. username/my-model"
    )
    parser.add_argument("--hf-token", help="HuggingFace token with write access")
    parser.add_argument("--session-id", help="Subfolder in HF repo to store checkpoints")
    parser.add_argument("--keep-last-k", type=int, help="Keep only last K checkpoints")
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        callbacks = []
        if getattr(args, "push_to_hf", False):
            if not args.hf_repo_id or not args.hf_token:
                _LOGGER.error(
                    "--push-to-hf requires --hf-repo-id and --hf-token"
                )
            else:
                callbacks.append(
                    HFCheckpointUploader(
                        hf_repo_id=args.hf_repo_id,
                        hf_token=args.hf_token,
                        session_id=getattr(args, "session_id", None),
                        every_n_epochs=args.checkpoint_epochs,
                        save_top_k=-1,
                    )
                )
        else:
            callbacks.append(
                ModelCheckpoint(
                    every_n_epochs=args.checkpoint_epochs,
                    save_top_k=-1,
                )
            )
        if getattr(args, "keep_last_k", None):
            callbacks.append(KeepLastKCheckpoints(args.keep_last_k))
        trainer.callbacks = callbacks
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    trainer.fit(model)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
