import torch
import torch.nn.functional as F
from torch import nn
from helm_module import helm_encoder
from time_module import TimeEncoder, ClassifyHeadTime, ClassifyHeadHELM


class Contrastive(nn.Module):
    def __init__(self,
                 num_hiddens,
                 temp,
                 queue_size,
                 momentum):
        super().__init__()
        # ============== initialize model =====================

        self.helm_enc = helm_encoder
        self.time_enc = TimeEncoder(num_hidden=1024,
                                    num_head=16,
                                    num_blocks=12,
                                    dropout=0.01)
        self.helm_enc_m = helm_encoder
        self.time_enc_m = TimeEncoder(num_hidden=1024,
                                      num_head=16,
                                      num_blocks=12,
                                      dropout=0.01)

        # self.time_enc.load_state_dict(torch.load('model/time_pre.pth'))
        # self.time_enc_m.load_state_dict(torch.load('model/time_pre.pth'))
        #
        # self.helm_enc.load_state_dict(torch.load('model/helm_enc_pre.pth'))
        # self.helm_enc_m.load_state_dict(torch.load('model/helm_enc_pre.pth'))

        self.project_H = nn.Linear(num_hiddens, num_hiddens)
        self.project_T = nn.Linear(num_hiddens, num_hiddens)
        self.project_H_m = nn.Linear(num_hiddens, num_hiddens)
        self.project_T_m = nn.Linear(num_hiddens, num_hiddens)

        self.model_pairs = [
            [self.project_T, self.project_T_m],
            [self.project_H, self.project_H_m],
            [self.helm_enc, self.helm_enc_m],
            [self.time_enc, self.time_enc_m]
        ]
        self.copy_params()

        self.classify_t = ClassifyHeadTime()
        self.classify_h = ClassifyHeadHELM()
        # self.classify_t.load_state_dict(torch.load('model/project_t_pre.pth'))
        # self.classify_h.load_state_dict(torch.load('model/helm_project_pre.pth'))

        # ============== initialize hyperparameters =============
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.queue_size = queue_size
        self.momentum = momentum

        # ============== initialize queue =======================
        self.register_buffer("time_queue", torch.randn(num_hiddens, self.queue_size))
        self.register_buffer("helm_queue", torch.randn(num_hiddens, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.time_queue = nn.functional.normalize(self.time_queue, dim=0)
        self.helm_queue = nn.functional.normalize(self.helm_queue, dim=0)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, helm_ids, helm_mask, time_seq, label, alpha=0.4):
        # ============== Contrastive Loss ========================
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # ============== Get feature =============================
        feat_t_ori = self.time_enc(time_seq)
        feat_h_ori = self.helm_enc(input_ids=helm_ids,
                                   attention_mask=helm_mask).last_hidden_state

        # feat_t = self.project_T(torch.mean(feat_t_ori, dim=1))
        # feat_h = self.project_H(torch.mean(feat_h_ori, dim=1))
        feat_t = F.normalize(self.project_T(feat_t_ori[:, 0, :]), dim=-1)
        feat_h = F.normalize(self.project_H(feat_h_ori[:, 0, :]), dim=-1)

        # ============== Get Momentum feature ====================
        with torch.no_grad():
            feat_t_m = self.time_enc_m(time_seq)
            feat_h_m = self.helm_enc_m(input_ids=helm_ids,
                                       attention_mask=helm_mask).last_hidden_state
            feat_t_m = F.normalize(self.project_T_m(feat_t_m[:, 0, :]), dim=-1)
            feat_h_m = F.normalize(self.project_H_m(feat_h_m[:, 0, :]), dim=-1)

            # ============== Get Similarity Targets ==============
            t_feat_all = torch.cat([feat_t_m.t(), self.time_queue.clone().detach()], dim=1)
            h_feat_all = torch.cat([feat_h_m.t(), self.helm_queue.clone().detach()], dim=1)

            sim_t2h_m = feat_t_m @ h_feat_all / self.temp
            sim_h2t_m = feat_h_m @ t_feat_all / self.temp

            sim_targets = torch.zeros(sim_t2h_m.size()).to(feat_t.device)
            sim_targets.fill_diagonal_(1)

            sim_t2h_targets = alpha * F.softmax(sim_t2h_m, dim=1) + (1 - alpha) * sim_targets
            sim_h2t_targets = alpha * F.softmax(sim_h2t_m, dim=1) + (1 - alpha) * sim_targets

        # ============== Calculate Loss ===========================
        sim_t2h = feat_t @ h_feat_all / self.temp
        sim_h2t = feat_h @ t_feat_all / self.temp

        loss_t2h = -torch.sum(F.log_softmax(sim_t2h, dim=1) * sim_t2h_targets, dim=1).mean()
        loss_h2t = -torch.sum(F.log_softmax(sim_h2t, dim=1) * sim_h2t_targets, dim=1).mean()

        loss_contrastive = (loss_t2h + loss_h2t) / 2
        self._dequeue_and_enqueue(feat_t_m, feat_h_m)

        # ============== Double BCE Loss ===========================
        time_input = feat_t_ori.view(feat_t_ori.size(0), -1)
        helm_input = feat_h_ori.view(feat_h_ori.size(0), -1)

        time_output = self.classify_t(time_input)
        helm_output = self.classify_h(helm_input)

        loss_time = self.criterion(time_output, label)
        loss_helm = self.criterion(helm_output, label)

        loss = loss_contrastive + loss_time + loss_helm

        time_predictions = (time_output > 0.5).int()
        time_correct = (time_predictions == label).float()
        time_accuracy = time_correct.sum() / time_correct.numel()

        helm_predictions = (helm_output > 0.5).int()
        helm_correct = (helm_predictions == label).float()
        helm_accuracy = helm_correct.sum() / helm_correct.numel()

        return loss, time_accuracy, helm_accuracy

    def predict(self, helm_ids, helm_mask):
        feat_h = self.helm_enc(input_ids=helm_ids,
                               attention_mask=helm_mask).last_hidden_state

        input_h = feat_h.view(feat_h.size(0), -1)
        output_h = self.classify_h(input_h)
        return output_h

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat_t, feat_h):
        batch_size = feat_t.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        # Check if the last batch will exceed the queue size
        self.time_queue[:, ptr:ptr + batch_size] = feat_t.T
        self.helm_queue[:, ptr:ptr + batch_size] = feat_h.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

















