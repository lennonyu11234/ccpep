import torch
from all_dataset import DatasetAll
from helm_dataset import DatasetHELM, DatasetLabeled
from torch.utils.data import DataLoader, random_split
from contrastive_module import Contrastive
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = DatasetAll('dataset/peptide_list.csv',
                     'dataset/polar.csv',
                     'dataset/non.csv',
                     'dataset/sasa.csv',
                     'dataset/hpep.csv',
                     'dataset/hdopc.csv',
                     'dataset/hsol.csv',
                     'dataset/rmsd.csv',
                     'dataset/rg.csv',
                     'dataset/rgx.csv',
                     'dataset/rgy.csv',
                     'dataset/rgz.csv')
torch.manual_seed(25)
total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=20, collate_fn=dataset.collate_fn,
                          drop_last=True)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=10, collate_fn=dataset.collate_fn,
                         drop_last=True)


class TrainContrastive(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Contrastive(num_hiddens=1024,
                                 temp=0.1,
                                 queue_size=200,
                                 momentum=0.995)

    def training_step(self, batch, batch_ids):
        time_seq = batch['time_seq']
        helm_ids, helm_mask = batch['helm_ids'], batch['helm_mask']
        label = batch['label'].float().unsqueeze(dim=1)
        loss, time_acc, helm_acc = self.model(helm_ids,
                                              helm_mask,
                                              time_seq,
                                              label)
        self.log('Train_loss', loss.clone().detach(),
                 on_epoch=True, prog_bar=True,
                 on_step=False)
        self.log('Train_acc_time', time_acc.clone().detach(),
                 on_epoch=True, prog_bar=True,
                 on_step=False)
        self.log('Train_acc_helm', helm_acc.clone().detach(),
                 on_epoch=True, prog_bar=True,
                 on_step=False)
        return loss

    def validation_step(self, batch, batch_ids):
        helm_ids, helm_mask = batch['helm_ids'], batch['helm_mask']
        label = batch['label'].float().unsqueeze(dim=1)
        output = self.model.predict(helm_ids, helm_mask)
        pred = (output > 0.5).int()
        correct = (pred == label).float()
        accuracy = correct.sum() / correct.numel()
        precision = precision_score(label.cpu().numpy(), pred.cpu().numpy())
        recall = recall_score(label.cpu().numpy(), pred.cpu().numpy())
        f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy())
        mcc = matthews_corrcoef(label.cpu().numpy(), pred.cpu().numpy())

        self.log('Val_acc', accuracy,
                 on_epoch=True, prog_bar=True,
                 on_step=False)
        self.log('Val_precision', precision,
                 on_epoch=True, prog_bar=False,
                 on_step=False)
        self.log('Val_recall', recall,
                 on_epoch=True, prog_bar=False,
                 on_step=False)
        self.log('Val_F1', f1,
                 on_epoch=True, prog_bar=False,
                 on_step=False)
        self.log('Val_MCC', mcc,
                 on_epoch=True, prog_bar=False,
                 on_step=False)

    def predict_step(self, batch, batch_ids):
        helm_ids, helm_mask = batch['helm_ids'], batch['helm_mask']
        label = batch['label'].float().unsqueeze(dim=1)
        output = self.model.predict(helm_ids, helm_mask)
        pred = (output > 0.5).int()
        # correct = (pred == label).float()
        accuracy = accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
        precision = precision_score(label.cpu().numpy(), pred.cpu().numpy())
        recall = recall_score(label.cpu().numpy(), pred.cpu().numpy())
        f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy())
        mcc = matthews_corrcoef(label.cpu().numpy(), pred.cpu().numpy())

        print(f'ACC:{accuracy:.4f}')
        print(f'precision:{precision:.4f}')
        print(f'recall:{recall:.4f}')
        print(f'f1:{f1:.4f}')
        print(f'mcc:{mcc:.4f}')

    def train_dataloader(self):
        # return DataLoader(dataset, batch_size=40, shuffle=True, drop_last=True,
        #                   collate_fn=dataset.collate_fn)
        return train_loader

    def val_dataloader(self):
        return test_loader

    def predict_dataloader(self):
        dataset_test = DatasetLabeled(path='dataset/CycPeptMPDB_Peptide_Length_6.csv')
        return DataLoader(dataset_test, batch_size=3000, shuffle=False, collate_fn=dataset_test.collate_fn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def on_train_epoch_end(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            model_path = 'model/contrastive_5.pth'
            torch.save(self.model.state_dict(), model_path)
            print(f'Model parameters have been saved to {model_path}')






