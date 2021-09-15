import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import SequentialSampler, DataLoader


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--test_data", default=None, type=str, required=True, help="The input data file")
    parser.add_argument("--test_anwser_candidates", default=None, type=str, required=True, help="The input anwser file")
    parser.add_argument(
        "--bert_dir", default="", type=str, required=False, help="The directory of the pretrained BERT model"
    )
    parser.add_argument(
        "--dialog_encoder_dir", default="", type=str, required=False, help="The directory of the pretrained BERT model"
    )
    parser.add_argument("--saved_model_dir", default="", type=str, required=False, help="saved model file")
    parser.add_argument(
        "--predict_file_path",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--eval_batch_size", default=2, type=int, help="dialog batch size for evaluation.")
    args = parser.parse_args()
    return args


def eval_predict(args):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if not os.path.exists(args.saved_model_dir):
        raise ValueError("no saved model", args.saved_model_dir)
    sys.path.append(args.saved_model_dir)

    from model import TextMatchBert

    predict_model = TextMatchBert(args, device)
    # load model
    # is_reload = True
    try:
        output_model_file = os.path.join(args.saved_model_dir, "step_model")
        print(output_model_file)
        ptr_model = torch.load(output_model_file)
        predict_model.load_state_dict(ptr_model, strict=False)
        print("#######reload_from\t{}#########".format(output_model_file))
    except Exception:
        import traceback

        print(traceback.format_exc())
        # is_reload = False
    predict_model.to(device)

    # get data
    from dataprocess import Processor, IDDataset, collate_fn

    processor = Processor(args)
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    pad_ids = tokenizer.convert_tokens_to_ids("[PAD]")

    label_map = processor.get_label_map()
    label_map_items = [x for x in label_map.items()]
    eval_examples = processor.get_test_examples(args.test_data)
    eval_dataset = IDDataset(eval_examples, tokenizer)
    labelid2tokenid = eval_dataset.get_label_map_ids(tokenizer, label_map)
    with torch.no_grad():
        predict_model.get_answer_rp(labelid2tokenid)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        drop_last=False,
        num_workers=1,
        collate_fn=lambda x: collate_fn(x, pad_ids),
    )
    predict_contents = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        input_ids, input_type = [x.to(device) for x in batch]
        with torch.no_grad():
            top_1_ids = predict_model(input_ids, input_type)
        predict_content = [label_map_items[x][0] for x in top_1_ids]
        predict_contents.extend(predict_content)
    with open(args.predict_file_path, "w") as f:
        json.dump(predict_contents, f, ensure_ascii=False, indent=4)
    print("finished!!!")


if __name__ == "__main__":
    args = define_parser()
    eval_predict(args)
