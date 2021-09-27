import sys
import os
import json
from collections import OrderedDict

cur_path, _ = os.path.split(os.path.abspath(__file__))
import pdb


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, dial_id=None, label_map_id=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.dial_id = dial_id
        self.label_map_id = label_map_id


class InputFeatures(object):
    def __init__(self, input_ids, input_len, label_id, label_len, dial_id, turn_id, label_map_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_ids = label_id
        self.label_len = label_len
        self.dial_id = dial_id
        self.turn_id = turn_id
        self.label_map_id = label_map_id


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

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

    def get_label_map(self):
        return self.label_map


if __name__ == "__main__":
    pass
