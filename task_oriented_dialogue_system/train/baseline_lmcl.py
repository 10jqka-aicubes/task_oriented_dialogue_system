import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from tqdm import tqdm


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


class TextMatchBert(nn.Module):
    def __init__(self, config, device):
        super(TextMatchBert, self).__init__()
        self.device = device
        # 可训练历史编码器
        self.dialog_encoder = BertForUtteranceEncoding.from_pretrained(config.dialog_encoder_file)
        # 通用回复编码器
        self.response_encoder = BertForUtteranceEncoding.from_pretrained(config.dialog_encoder_file)
        # 固定参数
        for p in self.response_encoder.bert.parameters():
            p.requires_grad = False
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.label_map = None
        self.m = 0.1

    def forward(self, phase, input_ids, input_len, labels, label_len, label_map_id=None):
        if phase.lower() in {"train", "dev"}:
            token_type_ids, input_attention_mask = self._make_aux_input_tensors(input_ids, input_len)
            input_hidden = self.dialog_encoder(input_ids, input_attention_mask, token_type_ids)
            input_rep = input_hidden.last_hidden_state[:, 0:1, :].squeeze(axis=1)
            # loss version 3.0 larger margin cosine loss
            input_rep_repeat = input_rep.unsqueeze(-1).repeat(1, 1, self.label_map.size(0))
            label_rep_repeat = self.label_map.unsqueeze(0).repeat(input_rep.size(0), 1, 1).transpose(2, 1)
            cos_table = self.cos(input_rep_repeat, label_rep_repeat)
            all_sum = torch.sum(torch.exp(cos_table), axis=1) - self.m
            label_rep_e_t = torch.zeros(len(label_map_id), dtype=torch.float32).to(self.device)
            for x, y in enumerate(label_map_id):
                label_rep_e_t[x] = cos_table[x][y]
            label_rep_e_t_m = label_rep_e_t - self.m
            label_rep_e_t_m = torch.exp(label_rep_e_t_m)
            label_rep_e_t = torch.exp(label_rep_e_t)
            prob = label_rep_e_t_m / (label_rep_e_t_m + all_sum - label_rep_e_t) + 1e-8
            loss = -torch.log(prob)
            # loss version 1.0
            # loss = torch.exp(-self.cos(input_rep, label_rep))
            # loss version 2.0
            # loss = 1.0 - self.cos(input_rep, label_rep)
            loss_mean = torch.mean(loss)
            loss_sum = torch.sum(loss)
            bs = input_ids.size(0)
            return loss_mean, (loss_sum, bs)
        elif phase.lower() in {"test"}:
            cos_scores = []
            token_type_ids, input_attention_mask = self._make_aux_input_tensors(input_ids, input_len)
            input_hidden = self.dialog_encoder(input_ids, input_attention_mask, token_type_ids)
            input_rep = input_hidden.last_hidden_state[:, 0:1, :].squeeze(axis=1)
            for sample_id in range(input_rep.size()[0]):
                input_rep_item = input_rep[sample_id : sample_id + 1, :].repeat(self.label_map.size()[0], 1)
                cos_out = self.cos(input_rep_item, self.label_map)
                cos_out = cos_out.tolist()
                cos_scores.append(cos_out)
            return cos_scores

    def get_answer_rp(self, labelid2tokenid):
        label_rps = []
        print("calculate label representation")
        for label_token in tqdm(labelid2tokenid, ncols=100):
            label_token_ids = torch.tensor(label_token).long().to(self.device)
            label_attention_mask = torch.ones(label_token_ids.size()).long().to(self.device)
            label_type = torch.zeros(label_token_ids.size()).long().to(self.device)
            label_token_ids = label_token_ids.view(1, -1)
            label_attention_mask = label_attention_mask.view(1, -1)
            label_type = label_type.view(1, -1)
            label_rp = self.response_encoder(label_token_ids, label_attention_mask, label_type)
            label_rp = label_rp.last_hidden_state[:, 0:1, :]
            label_rps.append(label_rp)
        self.label_map = torch.cat(label_rps, 1).squeeze()

    def _make_aux_input_tensors(self, ids, sent_len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for bs_id, turn_len in enumerate(sent_len):
            type_id = 0
            last_index = 0
            curr_index = 0
            for sent_id, turn in enumerate(turn_len):
                curr_index = last_index + turn
                if type_id == 0:
                    type_id = 1
                else:
                    token_type_ids[bs_id][last_index:curr_index] = torch.tensor([1] * turn, dtype=torch.long).to(
                        self.device
                    )
                    type_id = 0
                last_index = curr_index
        attention_mask = ids > 0
        attention_mask = attention_mask.long().to(self.device)
        return token_type_ids, attention_mask

    def _make_aux_label_tensors(self, ids, sent_len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        attention_mask = ids > 0
        attention_mask = attention_mask.long().to(self.device)
        return token_type_ids, attention_mask


if __name__ == "__main__":
    pass
