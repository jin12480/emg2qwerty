# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import hydra
import omegaconf
import torch
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


class _EpochSummaryCallback(pl.Callback):
    """One line per train epoch (no step-level tqdm spam). Val still runs but bar is off."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        loss = m.get("train/loss")
        cer = m.get("train/CER")
        loss_s = f"{loss:.4f}" if loss is not None and hasattr(loss, "item") else (str(loss) if loss is not None else "n/a")
        cer_s = f"{cer:.4f}" if cer is not None and hasattr(cer, "item") else (str(cer) if cer is not None else "n/a")
        print(f"[epoch {epoch}] train loss={loss_s}  train/CER={cer_s}", flush=True)


class _ValEpochSummaryCallback(pl.Callback):
    """One line per validation epoch so logs have full val curve (for plotting)."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        loss = m.get("val/loss")
        cer = m.get("val/CER")
        loss_s = f"{loss:.4f}" if loss is not None and hasattr(loss, "item") else (str(loss) if loss is not None else "n/a")
        cer_s = f"{cer:.4f}" if cer is not None and hasattr(cer, "item") else (str(cer) if cer is not None else "n/a")
        print(f"[epoch {epoch}] val loss={loss_s}  val/CER={cer_s}", flush=True)


class _BestValTrainSnapshotCallback(pl.Callback):
    """Capture train CER/IER/DER/SER for the epoch that achieves best val/CER.

    This matches the user's desired "train@best-val-epoch" reporting (not post-hoc eval on train set).
    """

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_train_bd: dict[str, float | None] = {}

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb is None:
            return

        best_path = getattr(ckpt_cb, "best_model_path", "") or ""
        last_path = getattr(ckpt_cb, "last_model_path", "") or ""
        # When best improves, PL typically saves a new checkpoint; last_path often equals best_model_path then.
        # This heuristic is robust enough for our single-checkpoint-callback setup.
        if not best_path or best_path != last_path:
            return

        m = trainer.callback_metrics
        self.best_epoch = trainer.current_epoch
        self.best_train_bd = {
            "CER": _metric_get(m, "train/CER"),
            "IER": _metric_get(m, "train/IER"),
            "DER": _metric_get(m, "train/DER"),
            "SER": _metric_get(m, "train/SER"),
        }


def _metric_get(d: dict, key: str):
    v = d.get(key)
    if v is None:
        return None
    return float(v) if hasattr(v, "item") else float(v)


def _breakdown_from_metrics(d: dict | None, prefix: str) -> dict[str, float | None]:
    """Extract CER, IER, DER, SER from trainer output dict (e.g. val_metrics[0])."""
    if not d or not isinstance(d, dict):
        return {"CER": None, "IER": None, "DER": None, "SER": None}
    return {
        "CER": _metric_get(d, f"{prefix}/CER"),
        "IER": _metric_get(d, f"{prefix}/IER"),
        "DER": _metric_get(d, f"{prefix}/DER"),
        "SER": _metric_get(d, f"{prefix}/SER"),
    }


def _print_aligned_cer_table(
    train_bd: dict[str, float | None],
    val_bd: dict[str, float | None],
    test_bd: dict[str, float | None],
) -> None:
    """Single table: train / val / test aligned with best val epoch + best ckpt."""
    rows = [
        ("train", train_bd),
        ("val", val_bd),
        ("test", test_bd),
    ]
    print(
        f"{'split':<6}  {'CER':>10}  {'IER':>10}  {'DER':>10}  {'SER':>10}",
        flush=True,
    )
    print("  " + "-" * 52, flush=True)
    for name, b in rows:
        def f(x):
            return f"{x:10.4f}" if x is not None else f"{'n/a':>10}"

        print(
            f"{name:<6}  {f(b.get('CER'))}  {f(b.get('IER'))}  {f(b.get('DER'))}  {f(b.get('SER'))}",
            flush=True,
        )


def _print_breakdown_block(title: str, b: dict[str, float | None]) -> None:
    cer, ier, der, ser = b.get("CER"), b.get("IER"), b.get("DER"), b.get("SER")
    if all(x is None for x in (cer, ier, der, ser)):
        print(f"  {title}: n/a", flush=True)
        return
    parts = []
    if cer is not None:
        parts.append(f"CER={cer:.6g}")
    if ier is not None:
        parts.append(f"IER={ier:.6g}")
    if der is not None:
        parts.append(f"DER={der:.6g}")
    if ser is not None:
        parts.append(f"SER={ser:.6g}")
    print(f"  {title}: " + "  ".join(parts), flush=True)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        try:
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.listconfig.ListConfig])
        except AttributeError:
            pass
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]
    callbacks.append(_EpochSummaryCallback())
    callbacks.append(_ValEpochSummaryCallback())
    best_train_cb = _BestValTrainSnapshotCallback()
    callbacks.append(best_train_cb)

    # Disable step-level progress bar; epoch summary printed by callback instead
    trainer_kwargs = OmegaConf.to_container(config.trainer, resolve=True)  # type: ignore[arg-type]
    if not isinstance(trainer_kwargs, dict):
        trainer_kwargs = dict(config.trainer)
    trainer_kwargs["enable_progress_bar"] = False

    # Initialize trainer
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)
    else:
        pass

    if config.train:
        # Load best checkpoint for val/test reporting
        try:
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.listconfig.ListConfig])
        except AttributeError:
            pass
        module = module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    }
    pprint.pprint(results, sort_dicts=False)

    # Final summary: CER + IER/DER/SER breakdown (same as CharacterErrorRates)
    val_bd = _breakdown_from_metrics(
        val_metrics[0] if val_metrics else None, "val"
    )
    test_bd = _breakdown_from_metrics(
        test_metrics[0] if test_metrics else None, "test"
    )
    train_bd = best_train_cb.best_train_bd if config.train else {}
    print("\n========== Character error breakdown (best ckpt; train is best-val-epoch train) ==========", flush=True)
    if best_train_cb.best_epoch is not None:
        print(f"  train = metrics logged during epoch {best_train_cb.best_epoch} when val/CER achieved its best", flush=True)
    else:
        print("  train = n/a (best-val-epoch snapshot not available)", flush=True)

    if train_bd and val_bd.get("CER") is not None and test_bd.get("CER") is not None:
        _print_aligned_cer_table(train_bd, val_bd, test_bd)
    else:
        if not train_bd:
            print("  train: n/a", flush=True)
        _print_breakdown_block("val", val_bd)
        _print_breakdown_block("test", test_bd)
    print("================================================================================\n", flush=True)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
