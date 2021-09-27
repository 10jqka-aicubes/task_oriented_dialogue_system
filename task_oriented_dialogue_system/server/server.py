import os
import json
import datetime
import argparse
import numpy as np
import random
import torch

from tornado import gen, web
from tornado.ioloop import IOLoop
from transformers import BertTokenizer
from bert_encoder_match.dataprocess import Processor
from bert_encoder_match.main import IDDataset


class KefuBertHandler(web.RequestHandler):
    def initialize(self):
        self.return_dict = {"status_code": 0, "status_msg": "ok"}

    @gen.coroutine
    def post(self):
        yield self._do()

    @gen.coroutine
    def get(self):
        yield self._do()

    @gen.coroutine
    def _do(self):
        print("-" * 80)
        print(datetime.datetime.now())
        print(self.request.arguments)

        memory_drop_flag = False
        try:
            context = self.get_argument("context", "")
            context = json.loads(context)

            private_memory = self.get_argument("private_memory", "")
            if not private_memory:
                private_memory = {}
            else:
                private_memory = json.loads(private_memory)

        except ValueError:
            self.return_dict["status_msg"] = "context error"
            self.return_dict["status_code"] = -1
        else:
            input_ids, input_len = [tokenizer.convert_tokens_to_ids("[CLS]")], [1]
            for i in range(0, len(context), 2):
                tokens_a = [x for x in tokenizer.tokenize(context[i])] + ["[SEP]"]
                user_ids, user_len = eval_dataset._truncate_single_sentence(tokenizer.convert_tokens_to_ids(tokens_a))
                input_ids, input_len = eval_dataset._truncate_dialog_turn(input_ids, input_len, user_ids, user_len)
                if i + 1 < len(context):
                    sys_tokens = [x for x in tokenizer.tokenize(context[i + 1])] + ["[SEP]"]
                    sys_ids, sys_len = eval_dataset._truncate_single_sentence(
                        tokenizer.convert_tokens_to_ids(sys_tokens)
                    )
                    input_ids, input_len = eval_dataset._truncate_dialog_turn(input_ids, input_len, sys_ids, sys_len)

            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            input_len = torch.tensor([input_len], dtype=torch.long).to(device)

            with torch.no_grad():
                cos_lists = model("test", input_ids, input_len, labels=None, label_len=None)

            i = np.argmax(cos_lists[0])
            r = label_map_items[i][0]

            try:
                private_memory = json.dumps(private_memory)
            except Exception:
                private_memory = json.dumps({})
                memory_drop_flag = True
            self.return_dict["answer"] = r
            self.return_dict["private_memory"] = private_memory
            self.return_dict["private_memory_drop"] = "私有记忆丢失" if memory_drop_flag else ""
        self.write(self.return_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", type=str, required=True, help="model file name")
    parser.add_argument(
        "--test_anwser_candidates", default=None, type=str, required=True, help="test_anwser_candidates"
    )
    parser.add_argument(
        "--bert_dir",
        default="/.pretrained_bert",
        type=str,
        required=False,
        help="The directory of the pretrained BERT model",
    )
    parser.add_argument(
        "--dialog_encoder_file",
        default="/.pretrained_bert",
        type=str,
        required=False,
        help="The directory of the pretrained BERT model",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total dialog batch size for training.")
    parser.add_argument("--dev_batch_size", default=1, type=int, help="Total dialog batch size for validation.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Total dialog batch size for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=999, help="random seed for initialization")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--patience", default=3.0, type=float, help="The number of epochs to allow no further improvement."
    )
    parser.add_argument("--eval_init", action="store_true", help="evaluate initial model parameters")
    parser.add_argument("--eval_match_top_n", type=int, default=1, help="random seed for initialization")
    args = parser.parse_args()

    # setup device
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    pad_ids = tokenizer.convert_tokens_to_ids("[PAD]")

    # build the model
    TextMatchBert = getattr(__import__(args.model, fromlist=('baseline_lmcl')), "TextMatchBert")
    model = TextMatchBert(args, device)

    # reload_model
    is_reload = True
    try:
        output_model_file = os.path.join(args.output_dir, "step_model")
        ptr_model = torch.load(output_model_file, map_location=torch.device("cpu"))
        model.load_state_dict(ptr_model, strict=False)
        print("#######reload_from\t{}#########".format(output_model_file))
    except Exception:
        is_reload = False
    model.to(device)
    model.eval()

    # get data
    assert (is_reload and not args.eval_init) or (not is_reload and args.eval_init)
    processor = Processor(args)
    label_map = processor.get_label_map()
    # label_map = {k: v for k, v in label_map.items() if v < 100}
    label_map_items = [x for x in label_map.items()]
    eval_examples = []
    eval_dataset = IDDataset(eval_examples, tokenizer)
    labelid2tokenid = eval_dataset.get_label_map_ids(tokenizer, label_map)
    with torch.no_grad():
        model.get_answer_rp(labelid2tokenid)

    # model.label_map = torch.tensor(np.load('label_emb.npy'), dtype=torch.float32).to(device)

    application = web.Application([(r"/kefu", KefuBertHandler, {})])
    application.listen(12595)
    print("initial_ok")

    IOLoop.current().start()
