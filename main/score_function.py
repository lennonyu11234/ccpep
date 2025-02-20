import torch
from torch import nn
from transformers import AutoTokenizer, BartForSequenceClassification
from extra_model.helm_module import helm_encoder
from extra_model.time_module import ClassifyHeadHELM
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("../HELM",
                                          ignore_mismatched_sizes=True)


class ScoreFunction(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        if func == 'Triple':
            self.tokenizer = AutoTokenizer.from_pretrained("../HELM",
                                                           ignore_mismatched_sizes=True)
            self.Score = BartForSequenceClassification.from_pretrained('../HELM',
                                                                       ignore_mismatched_sizes=True)
            self.Score.resize_token_embeddings(len(self.tokenizer))
            self.Score.load_state_dict(torch.load('model/Score_Function/Triple/Triple-Bart-HELM-fold1.ckpt'))

        if func == 'Binary':
            self.tokenizer = AutoTokenizer.from_pretrained("HELM-Triple",
                                                           ignore_mismatched_sizes=True)
            self.Score = BartForSequenceClassification.from_pretrained('HELM-Binary',
                                                                       ignore_mismatched_sizes=True)
            self.Score.resize_token_embeddings(len(self.tokenizer))
            self.Score.load_state_dict(torch.load('model/Score_Function/Binary/Binary-Bart-HELM-fold1.ckpt'))

        if func == 'Regression':
            self.tokenizer = AutoTokenizer.from_pretrained("HELM-Regression",
                                                           ignore_mismatched_sizes=True)
            self.Score = BartForSequenceClassification.from_pretrained('HELM-Regression',
                                                                       ignore_mismatched_sizes=True)
            self.Score.resize_token_embeddings(len(self.tokenizer))
            self.Score.load_state_dict(torch.load('model/Score_Function/Regression/Regression-Bart-HELM-fold1.ckpt'))

    def forward(self, cyc_ids, cyc_mask):
        if self.func == 'Triple':
            output = self.Score(
                input_ids=cyc_ids,
                attention_mask=cyc_mask
            )
            output = torch.argmax(output.logits, dim=1)
            value0, value1, value2 = -0.5, 0.2, 0.3
            final_output = torch.tensor(output)
            final_output = torch.where(final_output == 0, value0, final_output)
            final_output = torch.where(final_output == 1, value1, final_output)
            final_output = torch.where(final_output == 2, value2, final_output)
            return final_output

        if self.func == 'Binary':
            output = self.Score(
                input_ids=cyc_ids,
                attention_mask=cyc_mask
            )
            output = torch.argmax(output.logits, dim=1)
            value0, value1, value2 = -0.5, 0.2, 0.3
            final_output = torch.tensor(output)
            final_output = torch.where(final_output == 0, value0, final_output)
            final_output = torch.where(final_output == 1, value1, final_output)
            final_output = torch.where(final_output == 2, value2, final_output)
            return final_output

        if self.func == 'Regression':
            output = self.Score(
                input_ids=cyc_ids,
                attention_mask=cyc_mask
            )
            predictions = output.logits.squeeze(dim=1)
            final_output = self.piecewise_normalize_tensor(predictions)
            return final_output

    def piecewise_normalize_tensor(self, tensor, old_range1=(-10, -6), old_range2=(-6, -4),
                                   new_range1=(-1, 0), new_range2=(0, 1)):
        old_min1, old_max1 = old_range1
        old_min2, old_max2 = old_range2
        new_min1, new_max1 = new_range1
        new_min2, new_max2 = new_range2

        # 第一个区间的归一化
        tensor_normalized = torch.where((tensor >= old_min1) & (tensor <= old_max1),
                                        ((new_max1 - new_min1) / (old_max1 - old_min1)) * (
                                                    tensor - old_min1) + new_min1,
                                        tensor)

        # 第二个区间的归一化
        tensor_normalized = torch.where((tensor >= old_min2) & (tensor <= old_max2),
                                        ((new_max2 - new_min2) / (old_max2 - old_min2)) * (
                                                    tensor - old_min2) + new_min2,
                                        tensor_normalized)

        return tensor_normalized


class ScoreTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.helm_enc = helm_encoder
        self.classify_h = ClassifyHeadHELM()
        self.helm_enc.load_state_dict(torch.load('../extra_model/model/helm_enc_final.pth'))
        self.classify_h.load_state_dict(torch.load('../extra_model/model/helm_project_final.pth'))

    def forward(self, helm_ids, helm_mask):
        feat_h_ori = self.helm_enc(
                input_ids=helm_ids,
                attention_mask=helm_mask).last_hidden_state

        helm_input = feat_h_ori.view(feat_h_ori.size(0), -1)
        pred = self.classify_h(helm_input)
        pred = (pred > 0.5).int()
        value0, value1 = -0.5, 0.2
        final_output = torch.tensor(pred)
        final_output = torch.where(final_output == 0, value0, final_output)
        final_output = torch.where(final_output == 1, value1, final_output)

        return final_output


def length_customisation(helm, length):
    length = length + 2
    score_length_list = []
    for helm_ids in helm:
        helm_ids = [x for x in helm_ids if x != 0]
        pep_length = len(helm_ids)
        if pep_length != length:
            minus = abs(pep_length - length)
            score = -(minus / length)
            score_length_list.append(score)
        else:
            score_length_list.append(0.2)
    return torch.tensor(score_length_list).to(device)


def natural_amino_acid_score(helm, score=None):
    if score == 'Neg':
        score4_list = []
        for helm_ids in helm:
            total_count = 0
            natural_count = 0
            natural_set = {4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13,
                           14, 15, 16, 17, 18,
                           19, 20, 21, 22,
                           # 268,
                           # 274, 278, 279, 280, 282,
                           # 284, 289, 290, 292, 294,
                           # 295
                           }
            for i in helm_ids:
                total_count += 1
                if i in natural_set:
                    natural_count += 1

            synthetic_ratio = (total_count - natural_count) / total_count
            score4 = -synthetic_ratio
            score4_list.append(score4)
    elif score == 'Pos':
        score4_list = []
        for helm_ids in helm:
            total_count = 0
            natural_count = 0
            natural_set = {4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13,
                           14, 15, 16, 17, 18,
                           19, 20, 21, 22,
                           # 268,
                           # 274, 278, 279, 280, 282,
                           # 284, 289, 290, 292, 294,
                           # 295
                           }
            for i in helm_ids:
                total_count += 1
                if i in natural_set:
                    natural_count += 1

            synthetic_ratio = natural_count / total_count
            score4 = synthetic_ratio
            score4_list.append(score4)
    return torch.tensor(score4_list).to(device)























