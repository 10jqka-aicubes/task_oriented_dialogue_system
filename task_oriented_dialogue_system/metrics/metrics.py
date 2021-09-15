# we develop our metrics code base on the following code:
# https://github.com/alexa/alexa-with-dstc9-track1-dataset/blob/42d08defdf31b0b913fb68b15854d668b1d4b729/baseline/utils/metrics.py
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html
import sys
import math
from collections import Counter
from fractions import Fraction


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def update(self, output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


class Bleu(Metric):
    def __init__(self):
        super(Bleu, self).__init__()
        self._bleu = None
        self._count = None

    def reset(self):
        super(Bleu, self).reset()
        self._bleu = 0.0
        self._count = 0

    def update(self, output):
        hypothesis, references = output
        bleu = sentence_bleu(references, hypothesis)
        self._bleu += bleu
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("BLEU-1 must have at least one example before it can be computed!")
        return self._bleu / self._count


def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
    return corpus_bleu([references], [hypothesis], weights=weights)


def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)):
    assert len(list_of_references) == len(hypotheses)
    p_numerators = Counter()
    p_denominators = Counter()
    hyp_lengths, ref_lengths = 0, 0
    for references, hypothesis in zip(list_of_references, hypotheses):
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closet_ref_length(references, hyp_len)
    bp = brevity_penalty(ref_lengths, hyp_lengths)
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False) for i, _ in enumerate(weights, start=1)]
    if p_numerators[1] == 0:
        return 0
    p_n_float = [float(x) if x.numerator != 0 else sys.float_info.min for x in p_n]
    score = [w_i * math.log(float(p_i)) for w_i, p_i in zip(weights, p_n_float)]
    score = bp * math.exp(math.fsum(score))
    return score


def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)


def modified_precision(references, hypothesis, n):
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts[ngram] if ngram in max_counts else 0, reference_counts[ngram])
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()}
    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))
    return Fraction(numerator, denominator, _normalize=Fraction)


def ngrams(token_list, n):
    output = []
    len_token_list = len(token_list)
    if len_token_list >= n:
        for step_index, _ in enumerate(token_list):
            if step_index + n <= len_token_list:
                output.append(tuple(token_list[step_index : step_index + n]))
    return output


def closet_ref_length(references, hyp_len):
    ref_lens = (len(reference) for reference in references)
    closet_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    return closet_ref_len


class Rouge(Metric):
    def __init__(self):
        super(Rouge, self).__init__()
        self.scorer = ROUGE()
        self._rouge = None
        self._count = None

    def reset(self):
        super(Rouge, self).reset()
        self._rouge = 0
        self._count = 0

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output
        rouge = self.scorer.calc_score(hypothesis, reference)
        self._rouge += rouge
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("ROUGE-L must have at least one example before it can be computed!")
        return self._rouge / self._count


def my_lcs(string, sub):
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class ROUGE(object):
    def __init__(self):
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        assert len(refs) > 0
        prec = []
        rec = []

        token_c = candidate
        for reference in refs:
            # split into tokens
            token_r = reference
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score
