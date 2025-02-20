import csv
import os
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from Dataset import DatasetHELM
from score_function import ScoreFunction, length_customisation, natural_amino_acid_score, ScoreTime
from Student_RNN import RNN
from utils.data_structs import unique, seq_to_helm
import pandas as pd
import pytorch_lightning as pl
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("../HELM",
                                          ignore_mismatched_sizes=True)


def piecewise_denormalize_tensor(tensor_normalized, old_range1=(-10, -6), old_range2=(-6, -4),
                                 new_range1=(-1, 0), new_range2=(0, 1)):
    # 计算每个区间的宽度
    old_width1 = old_range1[1] - old_range1[0]
    old_width2 = old_range2[1] - old_range2[0]
    new_width1 = new_range1[1] - new_range1[0]
    new_width2 = new_range2[1] - new_range2[0]

    # 第一个区间的反归一化
    tensor_denormalized = torch.where(
        (tensor_normalized >= new_range1[0]) & (tensor_normalized <= new_range1[1]),
        (old_width1 / new_width1) * (tensor_normalized - new_range1[0]) + old_range1[0],
        tensor_normalized
    )

    # 第二个区间的反归一化
    tensor_denormalized = torch.where(
        (tensor_normalized >= new_range2[0]) & (tensor_normalized <= new_range2[1]),
        (old_width2 / new_width2) * (tensor_normalized - new_range2[0]) + old_range2[0],
        tensor_denormalized
    )
    tensor_denormalized = float(tensor_denormalized)
    return tensor_denormalized


class Customize(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("HELM-Triple",
                                                       ignore_mismatched_sizes=True)
        self.restore_prior_from = 'model/Prior-RNN.ckpt'
        self.restore_agent_from = 'model/Prior-RNN.ckpt'
        self.agent_save = 'model/Len6-batch2.ckpt'
        self.sigma = 60
        self.save_dir = 'Result/Len6-batch2/'
        self.dataset = DatasetHELM(path='dataset/for train/pretrain.csv')
        vocab = self.dataset.amino_acid_vocab

        self.Prior = RNN(vocab)
        self.Agent = RNN(vocab)
        self.Prior.rnn.load_state_dict(torch.load(self.restore_prior_from))
        self.Agent.rnn.load_state_dict(torch.load(self.restore_agent_from))
        for param in self.Prior.rnn.parameters():
            param.requires_grad = False

        self.scoring_function_binary = ScoreFunction('Binary')
        self.scoring_function_triple = ScoreFunction('Triple')
        self.scoring_function_regression = ScoreFunction('Regression')
        self.scoring_time = ScoreTime()
        for param in self.scoring_function_binary.parameters():
            param.requires_grad = False
        for param in self.scoring_function_regression.parameters():
            param.requires_grad = False
        for param in self.scoring_function_triple.parameters():
            param.requires_grad = False
        for param in self.scoring_time.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        helm_save = []
        seqs, agent_likelihood, entropy = self.Agent.sample(batch_size=256)
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]

        prior_likelihood = self.Prior.likelihood(torch.tensor(seqs))
        helms = seq_to_helm(seqs)
        helm_indices = [self.tokenizer.encode(str(sequence), add_special_tokens=False) for sequence in helms]
        helm_indices = [indices + [2] for indices in helm_indices]
        helm_ = self.dataset.collate_fn(helm_indices)
        helm_ids, helm_mask = helm_['helm_ids'].to(device=device), helm_['helm_mask'].to(device=device)
        helm_save.extend(helms)

        # customs = []
        # opioid_seq = [5, 12, 7, 8, 8, 22, 331]
        # for indices in helm_indices:
        #     custom_indices = indices[:-2]
        #     custom_indices.extend(opioid_seq)
        #     customs.append(custom_indices)
        # helm_indices = [indices + [2] for indices in customs]
        # helm_ = self.dataset.collate_fn(helm_indices)
        # custom_helm_ids, custom_helm_mask = helm_['helm_ids'].to(device=device), helm_['helm_mask'].to(device=device)
        # custom_helm = seq_to_helm(customs)
        # helm_save.extend(custom_helm)

        # permeability score function
        score_reg = self.scoring_function_regression(helm_ids, helm_mask).to(device)
        score_bin = self.scoring_function_binary(helm_ids, helm_mask).to(device)
        score_tri = self.scoring_function_triple(helm_ids, helm_mask).to(device)
        score_permeability = score_reg + score_bin + score_tri

        score_length = length_customisation(helm_ids, length=6)

        score_natural_neg = natural_amino_acid_score(helm_indices, score='Neg')
        score_natural_pos = natural_amino_acid_score(helm_indices, score='Pos')
        score_natural = score_natural_pos + score_natural_neg
        score_time = self.scoring_time(helm_ids, helm_mask).to(device)

        # write the generation of agent to csv file and parameters of agent model
        all_rows = []
        for i, (sequence, score1, score2, score3) in enumerate(zip(helm_save, score_reg, score_bin, score_tri)):
            indice = str(sequence)[2:-2]
            indice = self.tokenizer.encode(indice)
            token = self.tokenizer.decode(indice)
            token = self.tokenizer.tokenize(token)
            token = token[1:-1]
            bind, amino_acid = token[-1:], token[:-1]
            acid = '.'.join(amino_acid)
            if not bind:
                continue
            helm = f'PEPTIDE{i}{{{acid}}}$PEPTIDE{i},PEPTIDE{i},{bind[0]}$$$'
            sequence_length = len(indice) - 3
            permeability = piecewise_denormalize_tensor(tensor_normalized=score1)
            all_rows.append({
                'HELM': str(helm),
                'Score1': float(permeability),
                'Score2': score2.item(),
                'Score3': score3.item(),
                'sequence_length': sequence_length
            })
        df = pd.DataFrame(all_rows)
        if batch_idx % 50 == 0 and batch_idx != 0:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            df.to_csv(self.save_dir + f'sample_{batch_idx}.csv', index=False, header=True)
        if batch_idx % 100 == 0 and batch_idx != 0:
            torch.save(self.Agent.rnn.state_dict(), self.agent_save)

        # reinforce process
        augmented_likelihood_permeability = prior_likelihood + self.sigma * torch.tensor(score_permeability)
        augmented_likelihood_length = prior_likelihood + self.sigma * torch.tensor(score_length)
        augmented_likelihood_time = prior_likelihood + self.sigma * torch.tensor(score_time)
        augmented_likelihood_natural = prior_likelihood + self.sigma * torch.tensor(score_natural)

        loss_permeability = torch.pow((augmented_likelihood_permeability - agent_likelihood), 2)
        loss_length = torch.pow((augmented_likelihood_length - agent_likelihood), 2)
        loss_time = torch.pow((augmented_likelihood_time - agent_likelihood), 2)
        loss_natural = torch.pow((augmented_likelihood_natural - agent_likelihood), 2)

        loss = loss_permeability + loss_length + loss_natural + loss_time * 0.2
        loss = loss.mean()
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        self.log("train_loss", loss.clone().detach(),
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        dataset = DatasetHELM(path='dataset/for train/pretrain.csv')
        return DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.Agent.rnn.parameters(), lr=1e-5)























