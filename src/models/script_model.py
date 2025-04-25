import glob
import os

import torch

from src.models.model import ResNeSt


def convert_ckpt_to_script_model(config):
    model_path = config.paths.model_path

    ckpt_list = glob.glob(os.path.join(model_path, "**", "*.ckpt"), recursive=True)

    if len(ckpt_list) == 0:
        raise FileNotFoundError(f"No .ckpt file found under {model_path}")
    elif len(ckpt_list) > 1:
        raise RuntimeError(
            f"Multiple .ckpt files found under {model_path}, please keep only one."
        )

    ckpt_path = ckpt_list[0]
    print(f"✅ Using checkpoint: {ckpt_path}")

    model = ResNeSt.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device("cpu"),
        hparams=config,
    )
    script_model = model.to_torchscript(method="script")

    deploy_path = os.path.join(model_path, "deployable_model.pt")
    torch.jit.save(script_model, deploy_path)
    print(f"✅ TorchScript model saved to: {deploy_path}")
