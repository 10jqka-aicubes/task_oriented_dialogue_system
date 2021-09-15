import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from tqdm import tqdm
import numpy as np


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
        self.dialog_encoder = BertForUtteranceEncoding.from_pretrained(config.dialog_encoder_dir)
        # 通用回复编码器
        self.response_encoder = BertForUtteranceEncoding.from_pretrained(config.dialog_encoder_dir)
        # 固定参数
        for p in self.response_encoder.bert.parameters():
            p.requires_grad = False
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.label_map = None
        self.m = 0.1

    def forward(self, input_ids, input_type):
        top_1_ids = []
        token_type_ids = input_type
        attention_mask = input_ids > 0
        attention_mask = attention_mask.long().to(self.device)
        input_hidden = self.dialog_encoder(input_ids, attention_mask, token_type_ids)
        input_rep = input_hidden.last_hidden_state[:, 0:1, :].squeeze(axis=1)
        for sample_id in range(input_rep.size()[0]):
            input_rep_item = input_rep[sample_id : sample_id + 1, :].repeat(self.label_map.size()[0], 1)
            cos_out = self.cos(input_rep_item, self.label_map)
            cos_out = cos_out.tolist()
            cos_out = np.array(cos_out)
            top_one_id = cos_out.argsort()[::-1][0]
            top_1_ids.append(top_one_id)
        return top_1_ids

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
