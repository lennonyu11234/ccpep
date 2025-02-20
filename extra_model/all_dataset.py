import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from helm_dataset import DatasetHELM


class DatasetAll(Dataset):
    def __init__(self, path,
                 csv_file_polar,
                 csv_file_non,
                 csv_file_sasa,
                 csv_file_hpep,
                 csv_file_hdopc,
                 csv_file_hsol,
                 csv_file_rmsd,
                 csv_file_rg,
                 csv_file_rgx,
                 csv_file_rgy,
                 csv_file_rgz,
                 label_col_index=0
                 ):
        self.helm_ori = DatasetHELM(path)
        self.data_frame_polar = pd.read_csv(csv_file_polar)
        self.tensor_data_polar = torch.tensor(self.data_frame_polar.to_numpy(), dtype=torch.float32)
        self.labels = self.tensor_data_polar[:, label_col_index]
        self.features_polar = self.tensor_data_polar[:, label_col_index + 1:]

        self.data_frame_non = pd.read_csv(csv_file_non)
        self.tensor_data_non = torch.tensor(self.data_frame_non.to_numpy(), dtype=torch.float32)
        self.features_non = self.tensor_data_non[:, label_col_index + 1:]

        self.data_frame_sasa = pd.read_csv(csv_file_sasa)
        self.tensor_data_sasa = torch.tensor(self.data_frame_sasa.to_numpy(), dtype=torch.float32)
        self.features_sasa = self.tensor_data_sasa[:, label_col_index + 1:]

        self.data_frame_hpep = pd.read_csv(csv_file_hpep)
        self.tensor_data_hpep = torch.tensor(self.data_frame_hpep.to_numpy(), dtype=torch.float32)
        self.features_hpep = self.tensor_data_hpep[:, label_col_index + 1:]

        self.data_frame_hdopc = pd.read_csv(csv_file_hdopc)
        self.tensor_data_hdopc = torch.tensor(self.data_frame_hdopc.to_numpy(), dtype=torch.float32)
        self.features_hdopc = self.tensor_data_hdopc[:, label_col_index + 1:]

        self.data_frame_hsol = pd.read_csv(csv_file_hsol)
        self.tensor_data_hsol = torch.tensor(self.data_frame_hsol.to_numpy(), dtype=torch.float32)
        self.features_hsol = self.tensor_data_hsol[:, label_col_index + 1:]

        self.data_frame_rmsd = pd.read_csv(csv_file_rmsd)
        self.tensor_data_rmsd = torch.tensor(self.data_frame_rmsd.to_numpy(), dtype=torch.float32)
        self.features_rmsd = self.tensor_data_rmsd[:, label_col_index + 1:]

        self.data_frame_rg = pd.read_csv(csv_file_rg)
        self.tensor_data_rg = torch.tensor(self.data_frame_rg.to_numpy(), dtype=torch.float32)
        self.features_rg = self.tensor_data_rg[:, label_col_index + 1:]

        self.data_frame_rgx = pd.read_csv(csv_file_rgx)
        self.tensor_data_rgx = torch.tensor(self.data_frame_rgx.to_numpy(), dtype=torch.float32)
        self.features_rgx = self.tensor_data_rgx[:, label_col_index + 1:]

        self.data_frame_rgy = pd.read_csv(csv_file_rgy)
        self.tensor_data_rgy = torch.tensor(self.data_frame_rgy.to_numpy(), dtype=torch.float32)
        self.features_rgy = self.tensor_data_rgy[:, label_col_index + 1:]

        self.data_frame_rgz = pd.read_csv(csv_file_rgz)
        self.tensor_data_rgz = torch.tensor(self.data_frame_rgz.to_numpy(), dtype=torch.float32)
        self.features_rgz = self.tensor_data_rgz[:, label_col_index + 1:]

    def __len__(self):
        return len(self.helm_ori)

    def __getitem__(self, idx):
        polar = self.features_polar[idx].unsqueeze(dim=1)
        non = self.features_non[idx].unsqueeze(dim=1)
        sasa = self.features_sasa[idx].unsqueeze(dim=1)
        hpep = self.features_hpep[idx].unsqueeze(dim=1)
        hdopc = self.features_hdopc[idx].unsqueeze(dim=1)
        hsol = self.features_hsol[idx].unsqueeze(dim=1)
        rmsd = self.features_rmsd[idx].unsqueeze(dim=1)
        rg = self.features_rg[idx].unsqueeze(dim=1)
        rgx = self.features_rgx[idx].unsqueeze(dim=1)
        rgy = self.features_rgy[idx].unsqueeze(dim=1)
        rgz = self.features_rgz[idx].unsqueeze(dim=1)
        time_seq = torch.cat((polar, non, sasa, hpep, hdopc, hsol, rmsd, rg, rgx, rgy, rgz), dim=1)
        time_seq = time_seq.transpose(1, 0)
        return {'seq': self.helm_ori[idx],
                'time_seq': time_seq.tolist(),
                'label': int(self.labels[idx])
                }

    def padding(self, sequence, max_len=18):
        padded_sequence = sequence + [0] * (max_len - len(sequence))
        return padded_sequence

    def collate_fn(self, batch):
        sequences, helm_attention_mask, time_seq, label = [], [], [], []
        for i in batch:
            sequences.append(i['seq'])
            helm_attention_mask.append([1 for _ in range(len(i['seq']))] +
                                       [0 for _ in range(18 - len(i['seq']))])
            time_seq.append(i['time_seq'])
            label.append(i['label'])

        padded_sequences = [self.padding(seq) for seq in sequences]

        return {
            'helm_ids': torch.tensor(padded_sequences),
            'helm_mask': torch.tensor(helm_attention_mask),
            'label': torch.tensor(label),
            'time_seq': torch.tensor(time_seq),
        }


