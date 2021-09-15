import os
import json
from collections import OrderedDict
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class InputExample(object):
    def __init__(self, history, user):
        self.history = history
        self.user = user


class InputFeatures(object):
    def __init__(self, input_ids, input_type):
        self.input_ids = input_ids
        self.input_type = input_type


class DataProcessor(object):
    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            data_set = json.load(f)
        return data_set


class Processor(DataProcessor):
    def __init__(self, args):
        super(Processor, self).__init__()
        self.label_map = OrderedDict()
        answer_list_file = args.test_anwser_candidates
        if os.path.exists(answer_list_file):
            with open(answer_list_file, "r") as f:
                line_list = json.loads(f.read())
                line_list = sorted(list(set(line_list)))
            for item_id, item in enumerate(line_list):
                self.label_map[item] = item_id
        else:
            raise ValueError("no answer list")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(data_dir))

    def _create_examples(self, data_set):
        examples = []
        for dialog in data_set:
            history = dialog["history"]
            user = dialog["user"]
            examples.append(InputExample(history=history, user=user))
        return examples

    def get_label_map(self):
        return self.label_map


class IDDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_length=512, max_single_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.all_features = []
        self.max_seq_length = max_seq_length
        self.max_single_length = max_single_length
        for example in tqdm(examples, ncols=100):
            input_ids, input_type = [tokenizer.convert_tokens_to_ids("[CLS]")], [0]
            history = example.history
            for turn in history:
                history_user = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn["user"]) + ["[SEP]"])
                history_sys = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn["sys"]) + ["[SEP]"])
                # add user
                input_ids += history_user
                input_type += len(history_user) * [1]
                # add sys
                input_ids += history_sys
                input_type += len(history_sys) * [0]
            user = example.user
            user_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(user) + ["[SEP]"])
            input_ids += user_ids
            input_type += len(user_ids) * [1]
            assert len(input_ids) == len(input_type)
            input_ids, input_type = self._truncate_dialog_turn(input_ids, input_type)
            self.all_features.append(InputFeatures(input_ids=deepcopy(input_ids), input_type=deepcopy(input_type)))

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, index):
        input_features = self.all_features[index]
        input_ids = torch.tensor(input_features.input_ids, dtype=torch.long)
        input_type = torch.tensor(input_features.input_type, dtype=torch.long)
        return input_ids, input_type

    def _truncate_dialog_turn(self, input_ids, input_type):
        if len(input_ids) <= self.max_seq_length:
            return input_ids, input_type
        else:
            cls_input, cls_type = input_ids[0:1], input_type[0:1]
            input_ids = cls_input + input_ids[-self.max_seq_length + 1 :]
            input_type = cls_type + input_type[-self.max_seq_length + 1 :]
            return input_ids, input_type

    def _truncate_single_sentence(self, token_ids):
        if len(token_ids) <= self.max_single_length:
            return token_ids, len(token_ids)
        return token_ids[0 : self.max_single_length - 1] + token_ids[-1::], self.max_single_length

    def get_label_map_ids(self, tokenizer, label_map):
        label_map2token = []
        for label_token, label_id in label_map.items():
            label_tokens = ["[CLS]"] + [x for x in tokenizer.tokenize(label_token)] + ["[SEP]"]
            label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            if len(label_ids) > self.max_seq_length:
                label_ids = label_ids[0:1] + label_ids[-self.max_seq_length + 1 :]
            label_map2token.append(label_ids)
        return label_map2token


def collate_fn(batch, pad_ids=0):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        result = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            result[i, : seq[i].size(0)] = seq[i]
        return result

    input_ids_list = []
    input_type_list = []
    for x in batch:
        input_ids_list.append(x[0])
        input_type_list.append(x[1])
    input_ids = padding(input_ids_list, torch.LongTensor([pad_ids]))
    input_type = padding(input_type_list, torch.LongTensor([0]))
    return input_ids, input_type
