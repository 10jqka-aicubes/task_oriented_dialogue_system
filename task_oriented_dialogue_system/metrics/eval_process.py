import os
import sys
import argparse
from tqdm import tqdm, trange
from transformers import BertTokenizer

cur_path, _ = os.path.split(os.path.abspath(__file__))
from metrics import Bleu, Rouge
import json
import pdb

result_dict = dict()


def main():
    # define parameter
    args = define_parser()

    # fetch data
    candidate_path = args.predict_file_path
    reference_path = args.reference_file_path
    candidates, references = fetch_data(candidate_path, reference_path)

    # tokenizer data
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    assert len(candidates) == len(references)
    candidates_tokenizers, references_tokenizers = [], []
    for candidate, reference in zip(candidates, references):
        candidates_tokenizers.append(tokenizer.tokenize(candidate.strip()))
        references_tokenizers.append(tokenizer.tokenize(reference.strip()))

    # calculate bleu
    eval_bleu([candidates_tokenizers, references_tokenizers])

    # calculate rouge
    eval_rouge([candidates_tokenizers, references_tokenizers])

    # write result
    with open(args.result_file, "w") as f:
        json.dump(result_dict, f, indent=4)
    print("done!!!")


def fetch_data(cand, ref):
    """Store each reference and candidate sentences as a list"""
    with open(cand, "r") as f:
        candidate = json.load(f)
    with open(ref, "r") as f:
        references = json.load(f)
    return candidate, references


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_file_path", default=None, type=str, required=True, help="The input data dir")
    parser.add_argument("--predict_file_path", default=None, type=str, required=True, help="prediction of the model")
    parser.add_argument(
        "--bert_dir", default="", type=str, required=False, help="The directory of the pretrained BERT model"
    )
    parser.add_argument("--result_file", default="", type=str, required=False, help="metrics result file")
    args = parser.parse_args()
    return args


def eval_bleu(output):
    bleu = Bleu()
    bleu.reset()
    candidates_tok, references_tok = output
    for candidate_tok, references_tok in zip(candidates_tok, references_tok):
        bleu.update([candidate_tok, [references_tok]])
    bleu_score = bleu.compute()
    print("BLEU: {}".format(bleu_score))
    result_dict["BLEU"] = bleu_score


def eval_rouge(output):
    rouge = Rouge()
    rouge.reset()
    candidates_tok, references_tok = output
    for candidate_tok, references_tok in zip(candidates_tok, references_tok):
        rouge.update([candidate_tok, [references_tok]])
    rouge_score = rouge.compute()
    print("ROUGE: {}".format(rouge_score))
    result_dict["ROUGE"] = rouge_score


if __name__ == "__main__":
    main()
