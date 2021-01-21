import numpy as np
import pandas as pd

import torch
from torch.utils import data

from utils import AverageMeter, seed_everything, get_filename
from PolicyGradient import REINFORCE, PPO
from dataset import VideoDataset, LstmDataset

from tqdm.auto import tqdm

import argparse
import json
import glob

parser = argparse.ArgumentParser(
    description="Video summarization through Deep RL - test"
)
parser.add_argument(
    "--run_name", default="", type=str, required=True, help="name to identify exp"
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
    "--train_cnn", action="store_true", help="Train the CNN backbone [resnet50]",
)
parser.add_argument(
    "--algorithm",
    default="reinforce",
    type=str,
    help="RL algorithm to train policy network [reinforce or ppo]",
)
args = parser.parse_args()
device = torch.device("cuda")


def load_dataloader(args, train_paths, val_paths):
    train_dataset = LstmDataset(train_paths, args.cnn_feat, args.dataset)
    val_dataset = LstmDataset(val_paths, args.cnn_feat, args.dataset)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_dataloader, val_dataloader


fold_scores = []
for fold in range(5):
    print("Fold::", fold)
    if args.dataset == "tvsum":
        f = open(f"folds/fold_{fold}.json")
        dataset = json.load(f)
        f.close()
        train_paths = dataset["train"]
        val_paths = dataset["val"]
        train_dataloader, val_dataloader = load_dataloader(args, train_paths, val_paths)
    else:
        paths = glob.glob("SumMe/features/*.pt")
        train_dataloader, val_dataloader = load_dataloader(args, paths, paths)

    if args.algorithm == "reinforce":
        agent = REINFORCE(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            args=args,
            fold=fold,
            device=device,
        )
    else:
        agent = PPO(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            args=args,
            fold=fold,
            device=device,
        )
    agent.load_policy(f"models/{args.run_name}/model_{fold}.pth")
    eval_score = agent.evaluate_policy(log=True)
    fold_scores.append(eval_score)

print("Avg OOF score:", np.mean(fold_scores))

