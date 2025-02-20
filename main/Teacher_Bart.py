import csv
import json
import os.path
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer, BartConfig
import pytorch_lightning as pl
from Dataset import DatasetHELM


class GeneratorHELM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained('../HELM',
                                                                 ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained("../HELM",
                                                       ignore_mismatched_sizes=True)
        self.bart.resize_token_embeddings(len(self.tokenizer))

    def training_step(self, batch, batch_idx):
        cyc_ids, cyc_mask = batch['helm_ids'], batch['helm_mask']
        output = self.bart(
            input_ids=cyc_ids,
            attention_mask=cyc_mask,
            labels=cyc_ids
        )
        loss = output.loss
        self.log("train_loss", loss.clone().detach(),
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        cyc_ids, cyc_mask = batch['helm_ids'], batch['helm_mask']
        preds = self.bart.generate(input_ids=cyc_ids,
                                   attention_mask=cyc_mask,
                                   max_length=18,
                                   min_length=4,
                                   num_return_sequences=10,
                                   num_beams=10)
        pred = [self.tokenizer.decode(g,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True) for g in preds]

        csv_file_path = 'Result/Teacher-Bart-generation.csv'
        with open(csv_file_path, 'a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for prediction in pred:
                helm_idx = self.tokenizer.encode(prediction)
                valid = False
                for index in helm_idx:
                    if 316 <= index <= 345:
                        valid = True

                if valid:
                    monomer, binding = self.add_dots_to_tokens(prediction)
                    a = 'PEPTIDE{{{}}}$PEPTIDE,PEPTIDE,{}$$$'.format(monomer, binding)
                    writer.writerow([a])

    def on_train_epoch_end(self):
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            save_dir = 'model/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            config = BartConfig('HELM-Triple')
            model_weight_path = os.path.join(save_dir, 'Teacher-Bart.ckpt')
            torch.save(self.bart.state_dict(), model_weight_path)
            config_path = os.path.join(save_dir, 'Teacher-Bart.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.__dict__, f, ensure_ascii=False)

    def train_dataloader(self):
        dataset = DatasetHELM(path='dataset/for train/pretrain.csv')
        return DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=dataset.collate_fn)

    def predict_dataloader(self):
        dataset = DatasetHELM(path='dataset/for train/pretrain.csv')
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.bart.parameters(), lr=1e-5)

    def add_dots_to_tokens(self, text):
        helm_idx = self.tokenizer.encode(text)
        monomer_idx, binding_idx = [], []
        found_bind = False
        for i, index in enumerate(helm_idx):
            if 316 <= index <= 345:
                if not found_bind:
                    binding_idx.append(i)
                    found_bind = True
            else:
                if not found_bind:
                    monomer_idx.append(i)
            if found_bind:
                break

        monomer_idx = list(filter(lambda x: x != 0, monomer_idx))
        helm = self.tokenizer.tokenize(text)

        monomers = [helm[j - 1] for j in monomer_idx]
        binding = [helm[j - 1] for j in binding_idx]
        return '.'.join(monomers), str(binding)

