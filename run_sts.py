# via examples from https://github.com/facebookresearch/SentEval

from collections import defaultdict
from functools import partial
from subspaces.subspaces import compute_subspace_sim, compute_vector_sim, subspace_embed_sentence, dyn_embedding_sizes, dyn_embedding_len_sizes, \
    get_pca_components, vector_embed_sentence
import json
import logging
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Set PATHs
PATH_TO_SENTEVAL = '../'


import senteval

import numpy as np

"""
Note:

The user has to implement two functions:
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) outputs a numpy array of sentence embeddings
        iii) Your sentence encoder should be in "params"
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
"""


# if we pass this prepare function, then arora-style deprojection is applied to word-vectors before computing subspaces
# (it's also applied to sent vectors in the vector-only case)
main_pc = None
def prepare_deproject(params, samples, alpha_embed=0.001):
    method = params["_method"]
    log.debug("computing first principal component over all sents")
    global main_pc
    sents = [sent for sent in samples if sent != []]
    sents_emb = [vector_embed_sentence(sent, alpha_embed=alpha_embed, method=method, normalize=False) for sent in sents]
    sents_emb = np.array([s for s in sents_emb if s is not None])
    V, _ = get_pca_components(sents_emb, 1)
    main_pc = V[0]
    log.debug("Done computing first principal component.")
    return


def prepare_none(params, samples, method=None):
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    return np.array([" ".join(b).lower() for b in batch])

# TODO adapt these settings for what model(s) you want to evaluate exactly
# TODO clone senteval repo (https://github.com/facebookresearch/SentEval) and put it in the root folder of this repo

# Set params for SentEval
PATH_TO_DATA = './SentEval-master/data/senteval_data'


DO_DEPROJECTION = False

DO_WPCA = True  # do frequency weighting on words while computing subspace
DO_MAGRATIO = True  # use ellipsoids rather than just subspaces

reference = {}
if __name__ == "__main__":
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']  # 'STSBenchmark',
    all_results = {}

    if DO_DEPROJECTION:
        log.debug("Using arora-style deprojection!")
        prepare = prepare_deproject
    else:
        prepare = prepare_none

    if DO_WPCA:
        log.debug("Will do weighted PCA (and word-weighting in vectors)!")
        alpha = 0.001
    else:
        alpha = None

    if DO_MAGRATIO:
        log.debug("Will use ellipsoid features!")
        # beta = 0.5 # TODO use this
        beta = 1.0
    else:
        beta = 0.

    # for setting in ["vec", "sp"]:
    for method in ["glove", "fasttext"]:  # glove_cc
        # for method in ["spacy"]:  # ""fasttext"]:
        # if setting == "vec":
        sim = lambda a, b: compute_vector_sim(str(a), str(b), method=method, deproject_pc=main_pc, alpha_embed=alpha)
        params_senteval = {
            'task_path': PATH_TO_DATA, 'usepytorch': False, '_method': method,
            'similarity': sim
        }
        params_senteval = dotdict(params_senteval)
        # se = senteval.SentEval(params_senteval, batcher, prepare)  # todo this is default
        se = senteval.SentEval(params_senteval, batcher, prepare)

        results = se.eval(transfer_tasks)
        all_results[("vec", method, "none")] = results
        # else:

        for rank in [4, 5, 6, 0.6, 0.7, 0.8, 0.9]:  # for rank in [0.8, 0.9, 0.95, 0.99]:
            sim = lambda a, b: compute_subspace_sim(
                a, b, rank_or_energy=rank, method=method, alpha=alpha,
                beta=beta, deproject_pc=main_pc
            )
            params_senteval = {
                'task_path': PATH_TO_DATA, 'usepytorch': False, '_method': method,
                'similarity': sim
            }
            params_senteval = dotdict(params_senteval)
            se = senteval.SentEval(params_senteval, batcher, prepare)
            results = se.eval(transfer_tasks)
            all_results[("sp", method, rank)] = results

    final_table = {
        "STS12": {
            "MSRpar": [0. for _ in range(len(all_results))],
            "MSRvid": [0. for _ in range(len(all_results))],
            "SMTeuroparl": [0. for _ in range(len(all_results))],
            "surprise.OnWN": [0. for _ in range(len(all_results))],
            "surprise.SMTnews": [0. for _ in range(len(all_results))],
        },
        "STS13": {
            "FNWN": [0. for _ in range(len(all_results))],
            "OnWN": [0. for _ in range(len(all_results))],
            "headlines": [0. for _ in range(len(all_results))],
        },
        "STS14": {
            "OnWN": [0. for _ in range(len(all_results))],
            "deft-forum": [0. for _ in range(len(all_results))],
            "deft-news": [0. for _ in range(len(all_results))],
            "headlines": [0. for _ in range(len(all_results))],
            "images": [0. for _ in range(len(all_results))],
            "tweet-news": [0. for _ in range(len(all_results))],
        },
        "STS15": {
            "answers-forums": [0. for _ in range(len(all_results))],
            "answers-students": [0. for _ in range(len(all_results))],
            "belief": [0. for _ in range(len(all_results))],
            "headlines": [0. for _ in range(len(all_results))],
            "images": [0. for _ in range(len(all_results))],
        },
        "STS16": {
            "answer-answer": [0. for _ in range(len(all_results))],
            "headlines": [0. for _ in range(len(all_results))],
            "plagiarism": [0. for _ in range(len(all_results))],
            "postediting": [0. for _ in range(len(all_results))],
            "question-question": [0. for _ in range(len(all_results))]
        },
        # "STSBenchmark": defaultdict(lambda: [0. for _ in range(len(all_results))]),
        "ZALL": {
            "STS12": [0. for _ in range(len(all_results))],
            "STS13": [0. for _ in range(len(all_results))],
            "STS14": [0. for _ in range(len(all_results))],
            "STS15": [0. for _ in range(len(all_results))],
            "STS16": [0. for _ in range(len(all_results))]
        }
    }
    headers = []
    for i, ((setting, method, rank), results) in enumerate(sorted(all_results.items(), key=lambda k: k[0])):
        keep = []
        conf = "%s-%s-%s" % (setting, method, rank)

        headers.append(conf)
        log.debug(conf)
        for year in sorted(results.keys()):
            for task in sorted(results[year].keys()):
                if task == "all":
                    final_table["ZALL"][year][i] = results[year][task]["pearson"]["mean"]
                    keep.append((results[year][task]["pearson"]["mean"], year, task))
                else:
                    final_table[year][task][i] = results[year][task]["pearson"][0]
                    log.debug("%.5s \t %s \t %s" % (results[year][task]["pearson"][0], year, task))
        for tup in keep:
            log.debug("%.5s \t ALL \t %s \t %s" % tup)
        log.debug("--------------------------------")

    print("\t".join(headers) + "\tTASK")
    for year in sorted(final_table.keys()):
        for task in sorted(final_table[year].keys()):
            print("\t".join([str(v)[:5] for v in final_table[year][task]]) + ("\t%s-%s" % (year, task)))
    input()
    print(json.dumps(dyn_embedding_sizes, indent=2))
    print(dyn_embedding_len_sizes)
