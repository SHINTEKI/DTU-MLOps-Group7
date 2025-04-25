# src/models/predict_model.py

import glob
import json
import logging
import os
import sys
from pathlib import Path

import click
import cv2
import hydra
import numpy as np
import torch
from hydra import compose

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.make_dataset import CROPSIZE, IMGNET_MEAN, IMGNET_STD
from src.models.script_model import convert_ckpt_to_script_model

IMAGE_EXT = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

log = logging.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def predict(input_filepath) -> None:
    hydra.initialize(
        config_path="../../conf", job_name="predict", version_base=None
    )
    config = compose(config_name="predict.yaml")
    paths = config.paths

    # Find or generate the model
    model_path = os.path.join(paths.model_path, "deployable_model.pt")
    if not os.path.exists(model_path):
        convert_ckpt_to_script_model(config)

    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()

    # Handle both file and folder input
    if os.path.isfile(input_filepath):
        files = [input_filepath]
    else:
        files = glob.glob(os.path.join(input_filepath, "*.*"))

    if not files:
        print(f"No images found at {input_filepath}")
        return

    for file in files:
        filename, ext = os.path.splitext(file)
        if ext.lower() not in IMAGE_EXT:
            log.warning(f"{file} is not a valid image file! Skipping.")
            continue

        img = cv2.imread(file)
        img = cv2.resize(img, (CROPSIZE, CROPSIZE))
        img = (img - 255 * np.array(IMGNET_MEAN)) / (
            255 * np.array(IMGNET_STD)
        )
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        img = torch.from_numpy(img).float().to(device)

        with torch.no_grad():
            output = model(img)
            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)

        outcome = mapping_to_outcome(top_class.item())
        print(f"[✓] Prediction for {file}: {outcome}")


def mapping_to_outcome(top_class):
    index_file = Path("app/index_to_name.json")
    if not index_file.exists():
        raise FileNotFoundError(
            "index_to_name.json file not found in app/ directory!"
        )

    with open(index_file) as f:
        data = json.load(f)

    return data[str(top_class)][1]


if __name__ == "__main__":
    predict()
