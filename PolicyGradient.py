import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import torchvision

import transformers

from utils import AverageMeter, seed_everything, get_filename
from reward import compute_reward
from tools import generate_summary, evaluate_summary

import logging
import os
import glob
from copy import deepcopy

from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, args, hid_dim=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.args = args
        if args.cnn_feat == "resnet50":
            in_dim = 2048
        else:
            in_dim = 1024

        if args.train_cnn:
            self.resnet = torchvision.models.mobilenet_v2(pretrained=True)
            modules = list(self.resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            in_dim = 1280

        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        feature = x.clone()
        if self.args.train_cnn:
            x = self.resnet(x)
            x = x.mean((2, 3))
            feature = x.detach().clone()
            x = x.unsqueeze(0)

        h, _ = self.rnn(x)
        out = torch.sigmoid(self.head(h))
        return out, feature


class Transformer(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.positional_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_token = 0

        self.layers = nn.ModuleList(
            [transformers.BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.rnn = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, x):
        feature = x.clone()
        seq_len = x.size(1) + 1
        batch_size = x.size(0)

        start_embed = self.positional_embedding(
            torch.LongTensor([self.start_token]).to(self.device)
        )
        start_embed = start_embed.unsqueeze(0).repeat(batch_size, 1, 1)

        sequence = torch.arange(seq_len).expand(1, -1).to(self.device)
        pos_embed = self.positional_embedding(sequence)
        x = torch.cat((start_embed, x), dim=1)
        x = x + pos_embed
        x = self.LayerNorm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
            x = x[0]

        x, _ = self.rnn(x)
        x = x[:, 1:, :]
        x = torch.sigmoid(self.head(x))
        return x, feature


class REINFORCE:
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        args,
        fold,
        baselines=None,
        gamma=0.99,
        beta=0.01,
        lr=1e-5,
        device="cpu",
    ):
        if args.decoder == "lstm":
            self.policy = PolicyNet(args=args)
        else:
            config = transformers.BertConfig()
            config.hidden_size = 2048
            config.num_attention_heads = 8
            config.num_hidden_layers = 4
            config.max_position_embeddings = 1500
            self.policy = Transformer(config, device)

        self.policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.baselines = baselines

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = device
        self.args = args
        self.fold = fold
        self.beta = beta

    def save_policy(self):
        if not os.path.exists(f"models/{self.args.run_name}"):
            os.mkdir(f"models/{self.args.run_name}")
        self.policy.eval()
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"models/{self.args.run_name}/model_{self.fold}.pth",
        )
        self.policy.train()

    def load_policy(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print("Checkpoint loaded")

    def evaluate_policy(self, dataloader=None, log=False):
        self.policy.eval()

        if not dataloader:
            dataloader = self.val_dataloader

        f_scores = AverageMeter()
        pbar = tqdm(dataloader, desc=f"Evaluation")

        score_dict = {}
        for batch_data in pbar:
            feature = batch_data["feature"].to(self.device)
            if self.args.train_cnn:
                feature = feature.squeeze(0)
            user_summary = batch_data["user_summary"].squeeze().numpy()
            change_points = batch_data["change_points"].squeeze().numpy()
            nfps = batch_data["nfps"].squeeze().numpy().tolist()
            picks = batch_data["picks"].squeeze().numpy()
            n_frames = batch_data["n_frames"].squeeze().item()
            id = batch_data["id"][0]

            with torch.no_grad():
                probs, _ = self.policy(feature)
                probs = probs.squeeze()
                probs = probs.cpu().numpy()

            summary = generate_summary(probs, change_points, n_frames, nfps, picks)
            metric, _, _ = evaluate_summary(summary, user_summary)

            score_dict[id] = metric
            f_scores.update(metric)

            pbar.set_postfix(f_score=f_scores.avg)

        if log:
            if not os.path.exists(f"evaluation_logs/{self.args.run_name}"):
                os.mkdir(f"evaluation_logs/{self.args.run_name}")
            df = pd.DataFrame(score_dict.items(), columns=["id", "f_score"])
            df.to_csv(
                f"evaluation_logs/{self.args.run_name}/fold_{self.fold}.csv",
                index=False,
            )

        return f_scores.avg

    def learn(self, epochs, num_episodes):
        logging.info(f"Fold: {self.fold}")

        best_eval_score = -float("inf")
        self.policy.train()

        for epoch in range(epochs):
            losses = AverageMeter()

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for idx, batch_data in enumerate(pbar):
                feature = batch_data["feature"].to(self.device)
                if self.args.train_cnn:
                    feature = feature.squeeze(0)
                id = batch_data["id"][0]
                probs, feature = self.policy(feature)

                loss = self.beta * (probs.mean() - 0.5) ** 2
                distr = Bernoulli(probs)
                rewards = AverageMeter()
                selected_frames = AverageMeter()

                for _ in range(num_episodes):
                    actions = distr.sample()
                    selected_frames.update((actions == 1.0).sum().item())
                    log_probs = distr.log_prob(actions)
                    reward = compute_reward(feature, actions, use_gpu=True)
                    loss = loss - log_probs.mean() * (reward - self.baselines[id])
                    rewards.update(reward.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.optimizer.step()
                self.baselines[id] = 0.9 * self.baselines[id] + 0.1 * rewards.avg

                losses.update(loss.item())
                avg_frames_selc = selected_frames.avg / feature.size(1)
                pbar.set_postfix(
                    loss=losses.avg, reward=rewards.avg, avg_frames_selc=avg_frames_selc
                )
                logging.info(
                    "Epoch: {}, Step: {}, Loss: {}, Reward: {}".format(
                        epoch + 1, idx + 1, loss.item(), rewards.avg
                    )
                )

            if epoch % 5 == 0:
                eval_metric = self.evaluate_policy()
                if eval_metric > best_eval_score:
                    best_eval_score = eval_metric

                self.policy.train()

            self.save_policy()

        logging.info("--------------------------------------")
        logging.info(f"Best eval score: {best_eval_score}")
        print("Best eval score:", best_eval_score)
        logging.info("--------------------------------------")

        return best_eval_score


class PPO:
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        args,
        fold,
        baselines=None,
        gamma=0.99,
        beta=0.01,
        eps_clip=0.2,
        lr=1e-5,
        device="cpu",
    ):
        if args.decoder == "lstm":
            self.policy = PolicyNet(args=args)
        else:
            config = transformers.BertConfig()
            config.hidden_size = 2048
            config.num_attention_heads = 8
            config.num_hidden_layers = 4
            config.max_position_embeddings = 1500
            self.policy = Transformer(config, device)

        self.policy.to(device)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.baselines = baselines

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = device
        self.args = args
        self.fold = fold
        self.beta = beta
        self.eps_clip = eps_clip

    def update_old(self):
        self.policy.eval()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy.train()

    def save_policy(self):
        if not os.path.exists(f"models/{self.args.run_name}"):
            os.mkdir(f"models/{self.args.run_name}")
        self.policy.eval()
        torch.save(
            {
                "model": self.policy.state_dict(),
                "model_old": self.policy_old.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"models/{self.args.run_name}/model_{self.fold}.pth",
        )
        self.policy.train()

    def load_policy(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model"])
        self.policy_old.load_state_dict(ckpt["model_old"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print("Checkpoint loaded")

    def evaluate_policy(self, dataloader=None, log=False):

        if not dataloader:
            dataloader = self.val_dataloader

        f_scores = AverageMeter()
        pbar = tqdm(dataloader, desc=f"Evaluation")

        score_dict = {}
        for batch_data in pbar:
            feature = batch_data["feature"].to(self.device)
            if self.args.train_cnn:
                feature = feature.squeeze(0)
            user_summary = batch_data["user_summary"].squeeze().numpy()
            change_points = batch_data["change_points"].squeeze().numpy()
            nfps = batch_data["nfps"].squeeze().numpy().tolist()
            picks = batch_data["picks"].squeeze().numpy()
            n_frames = batch_data["n_frames"].squeeze().item()
            id = batch_data["id"][0]

            with torch.no_grad():
                probs, _ = self.policy_old(feature)
                probs = probs.squeeze()
                probs = probs.cpu().numpy()

            summary = generate_summary(probs, change_points, n_frames, nfps, picks)
            metric, _, _ = evaluate_summary(summary, user_summary)

            score_dict[id] = metric
            f_scores.update(metric)

            pbar.set_postfix(f_score=f_scores.avg)

        if log:
            if not os.path.exists(f"evaluation_logs/{self.args.run_name}"):
                os.mkdir(f"evaluation_logs/{self.args.run_name}")
            df = pd.DataFrame(score_dict.items(), columns=["id", "f_score"])
            df.to_csv(
                f"evaluation_logs/{self.args.run_name}/fold_{self.fold}.csv",
                index=False,
            )

        return f_scores.avg

    def learn(self, epochs, num_episodes):
        logging.info(f"Fold: {self.fold}")

        best_eval_score = -float("inf")
        self.policy.train()

        for epoch in range(epochs):
            losses = AverageMeter()

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for idx, batch_data in enumerate(pbar):
                feature = batch_data["feature"].to(self.device)
                if self.args.train_cnn:
                    feature = feature.squeeze(0)
                id = batch_data["id"][0]
                with torch.no_grad():
                    old_probs, _ = self.policy_old(feature)

                old_distr = Bernoulli(old_probs)
                rewards = AverageMeter()
                selected_frames = AverageMeter()

                for _ in range(num_episodes):
                    probs, feature = self.policy(feature)
                    distr = Bernoulli(probs)
                    actions = distr.sample()
                    old_actions = old_distr.sample()
                    selected_frames.update((old_actions == 1.0).sum().item())
                    log_probs = distr.log_prob(old_actions)
                    old_log_probs = old_distr.log_prob(old_actions)
                    entropy = distr.entropy()

                    reward = compute_reward(feature, actions, use_gpu=True)
                    rewards.update(reward.item())

                    ratios = torch.exp(log_probs - old_log_probs)
                    advantages = reward - self.baselines[id]

                    self.optimizer.zero_grad()
                    surr1 = ratios.mean() * advantages
                    surr2 = (
                        torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                        * advantages
                    )

                    loss = (-torch.min(surr1, surr2) - 0.01 * entropy).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                    self.optimizer.step()
                    self.baselines[id] = 0.9 * self.baselines[id] + 0.1 * rewards.avg

                self.update_old()
                losses.update(loss.item())
                avg_frames_selc = selected_frames.avg / feature.size(1)
                pbar.set_postfix(
                    loss=losses.avg, reward=rewards.avg, avg_frames_selc=avg_frames_selc
                )
                logging.info(
                    "Epoch: {}, Step: {}, Loss: {}, Reward: {}".format(
                        epoch + 1, idx + 1, loss.item(), rewards.avg
                    )
                )

            if epoch % 5 == 0:
                eval_metric = self.evaluate_policy()
                if eval_metric > best_eval_score:
                    best_eval_score = eval_metric

                self.policy.train()

            self.save_policy()

        logging.info("--------------------------------------")
        logging.info(f"Best eval score: {best_eval_score}")
        print("Best eval score:", best_eval_score)
        logging.info("--------------------------------------")

        return best_eval_score
