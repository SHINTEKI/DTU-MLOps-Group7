from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.make_dataset import DataModule
from src.models.model import ResNeSt


@hydra.main(
    version_base="1.1", config_path="../../conf", config_name="config.yaml"
)
def train(config):
    paths = config.paths
    Path(paths.log_path + config.experiment.name).mkdir(
        parents=True, exist_ok=True
    )
    Path(paths.model_path + config.experiment.name).mkdir(
        parents=True, exist_ok=True
    )

    wandb_logger = WandbLogger(
        name=config.experiment.name,
        save_dir=paths.log_path + config.experiment.name,
        project="mlops-project",
        log_model=config.wandb.log_model,
    )

    datamodule = DataModule(config)
    model = ResNeSt(config.experiment)

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths.model_path + config.experiment.name,
        filename="{epoch:02d}-{val_accuracy:.4f}",
        monitor=config.experiment.monitor,
        mode=config.experiment.monitor_mode,
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor=config.experiment.monitor,
        patience=config.experiment.es_patience,
        mode=config.experiment.monitor_mode,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=config.experiment.max_epochs,
        num_sanity_val_steps=0,
        devices=1,
        accelerator="cpu",  # 💡你没有 GPU
        precision=32,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
