import argparse
import logging
import os
import typing
from functools import partial
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb.wandb_run import Run

from opf.dataset import CaseDataModule, PowerflowData
from opf.models import ModelRegistry
from opf.models.base import OPFModel
from opf.modules import OPFDual

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def add_common_args(parser):
    parser.add_argument("--compile", action="store_true", dest="compile")

    # logging arguments
    group = parser.add_argument_group("Logging")
    group.add_argument("--log_dir", type=str, default="./logs")
    group.add_argument("--notes", type=str, default="")
    group.add_argument("--no_log", action="store_false", dest="log")

    # data arguments
    group = parser.add_argument_group("Data")
    group.add_argument("--data_dir", type=str, default="./data")
    group.add_argument("--case_name", type=str, default="case118_ieee")
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--num_workers", type=int, default=0)
    group.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Relative size of training dataset.",
    )
    group.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Relative size of validation dataset.",
    )
    group.add_argument(
        "--test_split", type=float, default=0.1, help="Relative size of test dataset."
    )

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument(
        "--hotstart", type=str, default=None, help="ID of run to hotstart with."
    )
    group.add_argument("--fast_dev_run", action="store_true")
    group.add_argument("--no_gpu", action="store_false", dest="gpu")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--patience", type=int, default=50)
    group.add_argument("--gradient_clip_val", type=float, default=0)

    # lightning module arguments
    group = parser.add_argument_group("Lightning Module")
    OPFDual.add_model_specific_args(group)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # program arguments
    parser.add_argument("operation", choices=["train", "study"])
    subparsers = parser.add_subparsers(title="model", dest="model_name", required=True)

    for model_name, model_cls in ModelRegistry.items():
        subparser = subparsers.add_parser(
            model_name, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        group = subparser.add_argument_group("Model")
        model_cls.add_model_specific_args(group)
        add_common_args(subparser)

    params = parser.parse_args()
    params_dict = vars(params)

    torch.set_float32_matmul_precision("high")
    if params.operation == "study":
        study(params_dict)
    elif params.operation == "train":
        train(make_trainer(params_dict), params_dict)
    else:
        raise ValueError(f"Unknown operation: {params.operation}")


def make_trainer(params, callbacks=[], wandb_kwargs={}):
    logger = None
    if params["log"]:
        logger = WandbLogger(
            entity="damowerko-academic",
            project="opf",
            save_dir=params["log_dir"],
            config=params,
            log_model=True,
            notes=params["notes"],
            **wandb_kwargs,
        )

        logger.log_hyperparams(params)
        typing.cast(Run, logger.experiment).log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                (path.endswith(".py") or path.endswith(".jl"))
                and "logs" not in path
                and ("src" in path or "scripts" in path)
            ),
        )

        # logger specific callbacks
        # callbacks += [
        #     ModelCheckpoint(
        #         monitor="val/invariant",
        #         dirpath=Path(params["log_dir"]) / "checkpoints",
        #         filename="best",
        #         auto_insert_metric_name=False,
        #         mode="min",
        #         save_top_k=1,
        #     ),
        # ]
    callbacks += [EarlyStopping(monitor="val/invariant", patience=params["patience"])]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        accelerator="cuda" if params["gpu"] else "cpu",
        devices=1,
        max_epochs=params["max_epochs"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
        gradient_clip_val=params["gradient_clip_val"],
        log_every_n_steps=1,
    )
    return trainer


def warmup_batch(module, dm, params):
    data: PowerflowData = next(iter(dm.train_dataloader()))
    graph, _ = data
    # move lightning_module to same device as data
    device = "cuda" if params["gpu"] else "cpu"
    module = module.to(device=device)
    graph.to(device=device)
    module(graph)


def _train(trainer: Trainer, params):
    logger.info(f"Initializing data module.")
    dm = CaseDataModule(
        dual_graph=ModelRegistry.is_dual(params["model_name"]),
        pin_memory=params["gpu"],
        data_splits=(params["train_split"], params["val_split"], params["test_split"]),
        **params,
    )

    logger.info("Calling datamodule setup.")
    dm.setup()
    assert dm.powerflow_parameters is not None
    n_nodes = (
        dm.powerflow_parameters.n_bus,
        dm.powerflow_parameters.n_branch,
        dm.powerflow_parameters.n_gen,
    )

    logger.info(f"Initializing model.")
    extra_model_kwargs = {}
    if params["model_name"] == "mlp":
        extra_model_kwargs["n_nodes"] = n_nodes
    model = ModelRegistry.get_class(params["model_name"])(
        metadata=dm.metadata(),
        out_channels=4,
        **extra_model_kwargs,
        **params,
    )

    logger.info(f"Running single batch to initialize lazy layers.")
    warmup_batch(model, dm, params)
    logger.info(
        f"Compiling model." if params["compile"] else "Skipping model compilation."
    )

    if params["compile"]:
        model = typing.cast(OPFModel, torch.compile(model.cuda()))
        logger.info("Running single batch to compile model.")
        warmup_batch(model, dm, params)
        logger.info("Model compilation complete.")

    logger.info("Initializing the lightning module.")
    lightning_module = OPFDual(
        model, n_nodes, n_train=len(dm.train_dataset), dual_graph=ModelRegistry.is_dual(params["model_name"]), **params  # type: ignore
    )

    if params["compile"]:
        logger.info(
            "Running a single batch to compile the model and/or initialize lazy weights."
        )

    logger.info("Starting training.")
    trainer.fit(lightning_module, dm)
    # TODO: Add julia support in docker
    # logger.info("Starting testing.")
    # trainer.test(lightning_module, dm)


def train(trainer: Trainer, params):
    _train(trainer, params)
    # trainer.test(model, dm)
    for logger in trainer.loggers:
        logger.finalize("finished")


def study(params: dict):
    study_name = "opf-hybrid-118"
    storage = os.environ["OPTUNA_STORAGE"]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource=200, reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True,
        directions=["minimize"],
    )
    study.optimize(
        partial(objective, default_params=params),
        n_trials=1,
    )


def objective(trial: optuna.trial.Trial, default_params: dict):
    n_channels = 64
    params = dict(
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        lr_dual_shared=trial.suggest_float("lr_dual_shared", 1e-5, 10.0, log=True),
        lr_dual_pointwise=trial.suggest_float(
            "lr_dual_pointwise", 1e-5, 10.0, log=True
        ),
        wd=trial.suggest_float("wd", 1e-16, 1.0, log=True),
        wd_dual_shared=trial.suggest_float("wd_dual_shared", 1e-16, 1.0, log=True),
        wd_dual_pointwise=trial.suggest_float(
            "wd_dual_pointwise", 1e-16, 1.0, log=True
        ),
        dropout=0.0,
        multiplier_type="hybrid",
        cost_weight=trial.suggest_float("cost_weight", 1e-2, 1e2, log=True),
        supervised_weight=0.0,
        augmented_weight=0.0,
        powerflow_weight=0.0,
        case_name="case118_ieee",
        # Architecture parameters
        n_layers=20,
        batch_size=25,
        n_heads=2,
        n_channels=n_channels,
        max_epochs=200,
        patience=100,
        # MLP parameters
        mlp_hidden_channels=2 * n_channels,
        mlp_read_layers=2,
        mlp_per_gnn_layers=2,
    )
    params = {**default_params, **params}
    trainer = make_trainer(
        params,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val/invariant"
            )
        ],
        wandb_kwargs=dict(group=trial.study.study_name),
    )

    train(trainer, params)

    # finish up
    if isinstance(trainer.logger, WandbLogger):
        trial.set_user_attr("wandb_id", trainer.logger.experiment.id)
    for logger in trainer.loggers:
        logger.finalize("finished")

    print(trainer.callback_metrics)
    return trainer.callback_metrics["val/invariant"].item()


if __name__ == "__main__":
    main()
