import os
import argparse
from copy import deepcopy
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm, trange
from .dataprocess import Processor, InputFeatures


class IDDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_length=512, max_single_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.all_features = []
        self.max_single_length = max_single_length
        self.max_seq_length = max_seq_length
        last_dialog_id = None
        input_ids, input_len = [], []
        last_sys_tokens = []
        print("transfer data to sample")
        for example in tqdm(examples, ncols=100):
            tmp_dialog_id = example.dial_id
            turn_id = example.guid.split("$")[-1]
            if tmp_dialog_id != last_dialog_id:
                last_dialog_id = tmp_dialog_id
                input_ids, input_len = [tokenizer.convert_tokens_to_ids("[CLS]")], [1]
                last_sys_tokens = []
            if last_sys_tokens:
                last_sys_ids, last_sys_len = self._truncate_single_sentence(
                    tokenizer.convert_tokens_to_ids(last_sys_tokens)
                )
                input_ids, input_len = self._truncate_dialog_turn(input_ids, input_len, last_sys_ids, last_sys_len)
            tokens_a = [x for x in tokenizer.tokenize(example.text_a)] + ["[SEP]"]
            last_sys_tokens = [x for x in tokenizer.tokenize(example.text_b)] + ["[SEP]"]
            user_ids, user_len = self._truncate_single_sentence(tokenizer.convert_tokens_to_ids(tokens_a))
            input_ids, input_len = self._truncate_dialog_turn(input_ids, input_len, user_ids, user_len)
            label_tokens = ["[CLS]"] + [x for x in tokenizer.tokenize(example.label)] + ["[SEP]"]
            label_ids, label_len = self._truncate_single_sentence(tokenizer.convert_tokens_to_ids(label_tokens))
            label_map_id = example.label_map_id
            if int(turn_id) >= 1:
                self.all_features.append(
                    InputFeatures(
                        input_ids=deepcopy(input_ids),
                        input_len=deepcopy(input_len),
                        label_id=label_ids,
                        label_len=label_len,
                        dial_id=tmp_dialog_id,
                        turn_id=turn_id,
                        label_map_id=label_map_id,
                    )
                )

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, index):
        input_features = self.all_features[index]
        input_ids = torch.tensor(input_features.input_ids, dtype=torch.long)
        input_len = torch.tensor(input_features.input_len, dtype=torch.long)
        label_ids = torch.tensor(input_features.label_ids, dtype=torch.long)
        label_len = torch.tensor(input_features.label_len, dtype=torch.long)
        dialog_id = input_features.dial_id
        turn_id = input_features.turn_id
        label_map_id = input_features.label_map_id
        return input_ids, input_len, label_ids, label_len, dialog_id, turn_id, label_map_id

    def _truncate_single_sentence(self, token_ids):
        if len(token_ids) <= self.max_single_length:
            return token_ids, len(token_ids)
        return token_ids[0 : self.max_single_length - 1] + token_ids[-1::], self.max_single_length

    def _truncate_dialog_turn(self, input_ids, input_len, add_ids, add_len):
        input_ids.extend(add_ids)
        input_len.append(add_len)
        delete_turn = 1
        while sum(input_len) > self.max_seq_length:
            if input_len[delete_turn] <= 1:
                delete_turn += 1
            else:
                input_ids = (
                    input_ids[0 : sum(input_len[0:delete_turn])]
                    + input_ids[-(sum(input_len[delete_turn + 1 : :]) + 1) : :]
                )
                input_len[delete_turn] = 1
        return input_ids, input_len

    def get_label_map_ids(self, tokenizer, label_map):
        label_map2token = []
        for label_token, label_id in label_map.items():
            label_tokens = ["[CLS]"] + [x for x in tokenizer.tokenize(label_token)] + ["[SEP]"]
            label_ids, _ = self._truncate_single_sentence(tokenizer.convert_tokens_to_ids(label_tokens))
            label_map2token.append(label_ids)
        return label_map2token


def collate_fn(batch, pad_ids=0):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        result = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            result[i, : seq[i].size(0)] = seq[i]
        return result

    input_ids_list, input_len_list, label_ids_list, label_len_list, dialog_id_list, turn_id_list, label_map_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for x in batch:
        input_ids_list.append(x[0])
        input_len_list.append(x[1])
        label_ids_list.append(x[2])
        label_len_list.append(x[3])
        dialog_id_list.append(x[4])
        turn_id_list.append(x[5])
        label_map_list.append(x[6])

    input_ids = padding(input_ids_list, torch.LongTensor([pad_ids]))
    input_len = padding(input_len_list, torch.LongTensor([pad_ids]))
    label_ids = padding(label_ids_list, torch.LongTensor([pad_ids]))
    label_len = label_len_list
    return input_ids, input_len, label_ids, label_len, dialog_id_list, turn_id_list, label_map_list


def eval_cos(cos_lists, label_map_id, top_n=1):
    cor = []
    wro = []
    assert len(cos_lists) == len(label_map_id)
    sample_results = []
    for sample_id, sample_cos in enumerate(cos_lists):
        arr = np.array(sample_cos)
        top_k_idx = arr.argsort()[::-1][0:top_n]
        sample_results.append(top_k_idx)
        if label_map_id[sample_id] in top_k_idx:
            cor.append(sample_id)
        else:
            wro.append(sample_id)
    return cor, wro, sample_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", type=str, required=True, help="model file name")
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir")
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

    # output file
    os.makedirs(args.output_dir, exist_ok=True)

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

    # resize train batch
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps)
        )
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

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
    TextMatchBert = getattr(__import__(args.model), "TextMatchBert")
    model = TextMatchBert(args, device)

    # reload_model
    is_reload = True
    try:
        output_model_file = os.path.join(args.output_dir, "step_model")
        ptr_model = torch.load(output_model_file)
        model.load_state_dict(ptr_model, strict=False)
        print("#######reload_from\t{}#########".format(output_model_file))
    except Exception:
        is_reload = False
    model.to(device)

    # get data
    processor = Processor(args)

    # train
    if args.do_train:
        # 获得所有回复标签
        label_map = processor.get_label_map()
        # train data
        train_examples = processor.get_train_examples(args.data_dir)
        # sample
        train_dataset = IDDataset(train_examples, tokenizer)

        num_train_samples = len(train_dataset)
        num_train_steps = int(num_train_samples / args.train_batch_size / args.gradient_accumulation_steps) * int(
            args.num_train_epochs
        )
        print("***** Running training *****")
        print("  Num examples = %d" % num_train_samples)
        print("  Batch size = %d" % args.train_batch_size)
        print("  Num steps = %d" % num_train_steps)
        labelid2tokenid = train_dataset.get_label_map_ids(tokenizer, label_map)
        with torch.no_grad():
            model.get_answer_rp(labelid2tokenid)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            drop_last=True,
            num_workers=1,
            collate_fn=lambda x: collate_fn(x, pad_ids),
        )
        # dev data
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_dataset = IDDataset(dev_examples, tokenizer)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(
            dev_dataset,
            sampler=dev_sampler,
            batch_size=args.dev_batch_size,
            drop_last=False,
            num_workers=1,
            collate_fn=lambda x: collate_fn(x, pad_ids),
        )
        print("***** Running validation *****")
        print("  Num examples = %d", len(dev_dataset))
        print("  Batch size = %d", args.dev_batch_size)
        print("Loaded data!")

        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                    "lr": args.learning_rate,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": args.learning_rate,
                },
            ]
            return optimizer_grouped_parameters

        # set optimizer
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        optimizer = BertAdam(
            optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=num_train_steps
        )
        # oom_time = 0
        last_update = 0
        best_dev_mean_loss = None
        # if reload set best_dev_mean_loss
        if is_reload:
            model.eval()
            dev_total_loss = 0
            dev_total_bs = 0
            print("Validation...")
            for step, batch in enumerate(tqdm(dev_dataloader)):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, input_len, label_ids, label_len, dialog_id, turn_id, label_map_id = batch
                with torch.no_grad():
                    _, loss_tuple = model("dev", input_ids, input_len, label_ids, label_len, label_map_id)
                dev_total_loss += loss_tuple[0]
                dev_total_bs += loss_tuple[1]
            dev_mean_loss = dev_total_loss / dev_total_bs
            best_dev_mean_loss = dev_mean_loss
            print("initial valid loss: %.6f best loss: %.6f" % (dev_mean_loss, best_dev_mean_loss))
            acc_loss_file = open(os.path.join(args.output_dir, "acc_loss.txt"), "a", encoding="utf-8")
            acc_loss_file.write("initial valid loss: %.6f best loss: %.6f\n" % (dev_mean_loss, best_dev_mean_loss))
            acc_loss_file.close()

        # train
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            print("Training...")
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, input_len, label_ids, label_len, dialog_id, turn_id, label_map_id = batch
                loss, _ = model("train", input_ids, input_len, label_ids, label_len, label_map_id)
                if args.gradient_accumulation_steps > 1:
                    print("loss: %.6f" % loss.item())
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # perform evaluation on validation dataset
            model.eval()
            dev_total_loss = 0
            dev_total_bs = 0
            print("Validation...")
            for step, batch in enumerate(tqdm(dev_dataloader)):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, input_len, label_ids, label_len, dialog_id, turn_id, label_map_id = batch
                with torch.no_grad():
                    _, loss_tuple = model("dev", input_ids, input_len, label_ids, label_len, label_map_id)
                dev_total_loss += loss_tuple[0]
                dev_total_bs += loss_tuple[1]
            dev_mean_loss = dev_total_loss / dev_total_bs
            if best_dev_mean_loss is None or dev_mean_loss < best_dev_mean_loss:
                # save model
                output_model_file = os.path.join(args.output_dir, "step_model")
                torch.save(model.state_dict(), output_model_file)
                best_dev_mean_loss = dev_mean_loss
                last_update = epoch
            print("epoch: %d valid loss: %.6f best loss: %.6f" % (epoch, dev_mean_loss, best_dev_mean_loss))
            acc_loss_file = open(os.path.join(args.output_dir, "acc_loss.txt"), "a", encoding="utf-8")
            acc_loss_file.write(
                "epoch: %d valid loss: %.6f best loss: %.6f\n" % (epoch, dev_mean_loss, best_dev_mean_loss)
            )
            acc_loss_file.close()
            if last_update + args.patience <= epoch:
                break

    if args.do_eval:
        model.eval()
        if (is_reload and not args.eval_init) or (not is_reload and args.eval_init):
            label_map = processor.get_label_map()
            label_map_items = [x for x in label_map.items()]
            eval_examples = processor.get_test_examples(args.data_dir)
            eval_dataset = IDDataset(eval_examples, tokenizer)
            labelid2tokenid = eval_dataset.get_label_map_ids(tokenizer, label_map)
            with torch.no_grad():
                model.get_answer_rp(labelid2tokenid)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
                drop_last=False,
                num_workers=1,
                collate_fn=lambda x: collate_fn(x, pad_ids),
            )
            print("***** Running evaluation *****")
            print("  Num examples = %d", len(eval_dataset))
            print("  Batch size = %d", args.eval_batch_size)

            sample_num = 0
            cor_num = 0
            out_str = ""
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, input_len, label_ids, label_len, dialog_id, turn_id, label_map_id = batch
                with torch.no_grad():
                    cos_lists = model("test", input_ids, input_len, label_ids, label_len)
                cor, wro, sample_results = eval_cos(cos_lists, label_map_id, args.eval_match_top_n)
                # 错误样本分析
                for wro_item in wro:
                    input_sentence = tokenizer.decode(input_ids[wro_item]).replace("[PAD]", "")
                    predict_label = label_map_items[sample_results[wro_item][0]][0]
                    actual_label = label_map_items[label_map_id[wro_item]][0]
                    dialog_id_sample = dialog_id[wro_item]
                    turn_id_sample = turn_id[wro_item]
                    out_item_str = "{}\t{}\t{}\t{}\t{}\n".format(
                        input_sentence, predict_label, actual_label, dialog_id_sample, turn_id_sample
                    )
                    out_str += out_item_str

                sample_num += len(cor) + len(wro)
                cor_num += len(cor)
            wr_sample_file = open(os.path.join(args.output_dir, "wr_sample.txt"), "a", encoding="utf-8")
            wr_sample_file.write(out_str)
            wr_sample_file.close()
            print("Top %d acc %.4f" % (args.eval_match_top_n, float(cor_num) / sample_num))
        else:
            raise ValueError("No model or set two models to evaluation")


if __name__ == "__main__":
    main()
