import numpy as np
import pandas as pd

import torch
from torch.utils import data

from utils import AverageMeter, seed_everything, get_filename
from PolicyGradient import REINFORCE, PPO
from dataset import VideoDataset, LstmDataset

from tqdm.auto import tqdm

import glob
import os
import argparse
import json
import logging


parser = argparse.ArgumentParser(description="Video summarization through Deep RL")
parser.add_argument(
    "--run_name", default="", type=str, required=True, help="name to identify exp",
)
parser.add_argument(
    "--dataset",
    default="",
    type=str,
    required=True,
    help="dataset to use [tvsum or summe]",
)
parser.add_argument(
    "--cnn_feat",
    default="resnet50",
    type=str,
    help="CNN feature extractor to use [resnet50 or resnet101]",
)
parser.add_argument(
    "--decoder",
    default="lstm",
    type=str,
    help="Decoder model to use [lstm or transformer]",
)
parser.add_argument(
    "--algorithm",
    default="reinforce",
    type=str,
    help="RL algorithm to train policy network [reinforce or ppo]",
)
parser.add_argument(
    "--use_augs", action="store_true", help="Use augmented features",
)
parser.add_argument(
    "--train_cnn", action="store_true", help="Train the CNN backbone [resnet50]",
)
parser.add_argument(
    "--epochs", default=60, type=int, help="number of epochs",
)
parser.add_argument(
    "--num_episodes", default=5, type=int, help="number of episodes",
)
parser.add_argument(
    "--seed", default=1, type=int, required=True, help="seed",
)
args = parser.parse_args()

seed_everything(seed=args.seed)

device = torch.device("cuda")


def load_dataloader(args, train_paths, val_paths):
    if args.train_cnn:
        train_dataset = VideoDataset(train_paths, args.dataset)
        val_dataset = VideoDataset(val_paths, args.dataset)
    else:
        train_dataset = LstmDataset(
            train_paths, args.cnn_feat, args.dataset, args.use_augs
        )
        val_dataset = LstmDataset(val_paths, args.cnn_feat, args.dataset, args.use_augs)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_dataloader, val_dataloader


def load_baselines(train_paths):
    baselines = {}
    for path in train_paths:
        id = get_filename(path)
        baselines[id] = 0.0
    return baselines


logging.basicConfig(
    filename=f"logs/{args.run_name}.log", level=logging.INFO, format="%(message)s",
)

fold_scores = []
for fold in range(5):
    print("Fold::", fold)
    f = open(f"folds/fold_{fold}.json")
    dataset = json.load(f)
    f.close()
    train_paths = dataset["train"]
    val_paths = dataset["val"]
    train_dataloader, val_dataloader = load_dataloader(args, train_paths, val_paths)
    baselines = load_baselines(train_paths)
    if args.algorithm == "reinforce":
        agent = REINFORCE(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            baselines=baselines,
            args=args,
            fold=fold,
            device=device,
        )
    else:
        agent = PPO(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            baselines=baselines,
            args=args,
            fold=fold,
            device=device,
        )
    best_eval_score = agent.learn(args.epochs, args.num_episodes)
    fold_scores.append(best_eval_score)
    print("--------------------------------------")

print("Avg OOF score:", np.mean(fold_scores))
logging.info(f"Avg OOF score: {np.mean(fold_scores)}")
