# Partly from: https://github.com/sunnweiwei/user-satisfaction-simulation/blob/master/baselines/spearman.py

import re
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

roberta = SentenceTransformer('stsb-roberta-large')

# x, y must be one-dimensional arrays of the same length
# Spearman algorithm
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))


def _text_minimal_processing(text: str) -> str:
    result = re.sub(r'[^\w\s]', ' ', text)
    result = re.sub(' +', ' ', result)
    return result.lower()


def get_bleu_1_4(references: list, candidates: list):
    refs  = [_text_minimal_processing(ref) for ref in references]
    cands = [_text_minimal_processing(cand) for cand in candidates]
    assert len(refs) == len(cands)

    sum_bleu_1 = 0
    sum_bleu_4 = 0
    for reference, candidate in zip(refs, cands):
        ref = [reference.split(' ')]
        cand = candidate.split(' ')

        sum_bleu_1 += sentence_bleu(ref, cand, weights=(1, 0, 0, 0))
        sum_bleu_4 += sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25))

    return round(sum_bleu_1/len(refs), 5), round(sum_bleu_4/len(refs), 5)


def get_rouge_1_2_L(references: list, candidates: list):
    refs  = [_text_minimal_processing(ref) for ref in references]
    cands = [_text_minimal_processing(cand) for cand in candidates]
    assert len(refs) == len(cands)

    rouge = Rouge()
    scores = rouge.get_scores(hyps=cands, refs=refs, avg=True)
    
    rouge_1_f1 = scores.get('rouge-1').get('f')
    rouge_2_f1 = scores.get('rouge-2').get('f')
    rouge_L_f1 = scores.get('rouge-l').get('f')
    
    return round(rouge_1_f1, 5), round(rouge_2_f1, 5), round(rouge_L_f1, 5)


def get_sts(sentences1: list, sentences2: list) -> float:
    sents1  = [_text_minimal_processing(sent) for sent in sentences1]
    sents2 = [_text_minimal_processing(sent) for sent in sentences2]
    assert len(sents1) == len(sents2)

    sum_similarities = 0
    for sent1, sent2 in zip(sents1, sents2):
        embedding1 = roberta.encode([sent1], convert_to_tensor=True)
        embedding2 = roberta.encode([sent2], convert_to_tensor=True)
        sum_similarities += float(util.cos_sim(embedding1, embedding2)[0][0])
    
    return round(sum_similarities / len(sents1), 5)


if __name__ == "__main__":
    reference_sentence = 'Yes book the tickets,also I want places to go in town and I want it to be at the center of the town.'
    candidate_sentence = 'Yes, please book it for me.'

    bleu_1, bleu_4 = get_bleu_1_4([reference_sentence], [candidate_sentence])
    print(bleu_1, bleu_4)

    rouge_1_f1, rouge_2_f1, rouge_L_f1 = get_rouge_1_2_L([reference_sentence], [candidate_sentence])
    print(rouge_1_f1, rouge_2_f1, rouge_L_f1)

    sts = get_sts([reference_sentence], [candidate_sentence])
    print(sts)