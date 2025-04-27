#!/bin/bash

# 拉取数据和模型
dvc pull data.dvc
dvc pull models.dvc

# 运行训练脚本
python -m src.models.train_model