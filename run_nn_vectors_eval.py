from operator import itemgetter
import os
from subspaces.subspaces import vector_embed_sentence, compute_vector_sim, stopwords
from random import shuffle, seed
import json
import logging
import string
import time
import itertools
import nmslib
import pickle
import numpy as np
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

seed(1234)

# TODO make this point to a folder that will be used to cache data
NNCACHE_FOLDER = "/path/to/nncache"

# TODO adapt this to point to a txt file that just has plain lower case sentences on each line
sents_data_dir = "/path/tp/just_sents.lower.txt"

letters = list("зсьовфдагтурйпб«эыинямжчеклю»ш")

translator = str.maketrans('', '', string.punctuation)


def evaluate_accuracy(approx_results, queries_gt):
    accuracies = []
    for query_id in approx_results:
        approx = approx_results[query_id]
        gt = queries_gt[query_id]
        found = set([res_ids for (res_ids, _) in approx])
        looking_for = set([gt_res_ids for (gt_res_ids, _, _) in gt])
        matches = found.intersection(looking_for)
        # print(approx, gt, matches)
        # input()
        acc = float(
            len(matches)
        ) / len(looking_for)
        accuracies.append(acc)
    total_accuracy = float(sum(accuracies)) / len(accuracies)
    return accuracies, total_accuracy


def dict_product(dicts):
    return [dict(zip(dicts, x)) for x in itertools.product(*dicts.values())]


def clean_sent(sent):
    sent = sent.translate(translator).strip().split()
    sent = [word for word in sent if word.lower() not in stopwords]
    return " ".join(sent)


def load_raw_sentence_data(db_size, num_queries):
    all_sents = []
    with open(sents_data_dir, "r") as inf:
        for i, line in enumerate(inf):
            if len(all_sents) >= max(db_size * 10, 2000000):  # we load more and then randomize
                break
            found = False
            for l in letters:
                if l in line:
                    found = True
                    break

            if found:
                continue
            sent = clean_sent(line)
            if 8 <= len(sent.split()) < 40:
                all_sents.append(line.strip())

    all_sents = list(enumerate(all_sents))
    shuffle(all_sents)
    log.debug("Loaded %s sents" % (db_size + num_queries))

    use_query_sents = all_sents[:num_queries]
    use_db_sents = all_sents[num_queries:num_queries + db_size]

    assert len(use_db_sents) == db_size
    assert len(use_query_sents) == num_queries
    print("Loaded sents (%s query and %s db)" % (len(use_query_sents), len(use_db_sents)))

    return use_query_sents, use_db_sents


def load_or_create_input_dataset(db_size, num_queries, method="fasttext"):
    fname = NNCACHE_FOLDER + "/input.%s.db%s.q%s.json" % (method, db_size, num_queries)
    if os.path.exists(fname):
        log.debug("Loading raw dataset from %s" % fname)
        with open(fname, "r") as inf:
            return json.load(inf)
    log.debug("Creating raw dataset and saving to %s" % fname)
    q, db = load_raw_sentence_data(db_size, num_queries)
    res = {"queries": q, "samples": db}
    with open(fname, "w") as outf:
        outf.write(json.dumps(res, indent=4))

    log.debug("done.")
    return res


def load_or_generate_ground_truth_data(db_size, num_queries, top, method, db_vec, query_vectors):
    fname = NNCACHE_FOLDER + "/%s.db%s.%sqs.top%s.vecs.gt" % (method, db_size, num_queries, top)
    if os.path.exists(fname):
        log.debug("Loading gt data from %s" % fname)
        with open(fname, "rb") as inf:
            (queries_gt, took) = pickle.load(inf)
        for key in queries_gt:  # only keep top
            queries_gt[key] = queries_gt[key][:top]

        return queries_gt, took

    queries_gt = {}
    start = time.time()
    for i__, (id_, qsp, query) in enumerate(query_vectors):
        log.debug("%s: Computing similarities for %s.." % (i__, query))
        similarities = [
            (i, compute_vector_sim(qsp, sample, _embed=False, normalized=True), sent) for i, sample, sent in db_vec
        ]
        similarities = list(reversed(sorted(similarities, key=itemgetter(1))))  # largest sims first
        queries_gt[id_] = similarities[:top]
    took = time.time() - start
    with open(fname, "wb") as outf:
        pickle.dump((queries_gt, took), outf)
    return queries_gt, took


def build_nmslib_index(db_vec, db_size, M=16, efConstruction=500, maxM=32, maxM0=64):
    fname = NNCACHE_FOLDER + "/nmsidx_{db_size}-{M}-{efConstruction}-{maxM}-{maxM0}".format(
        db_size=db_size, M=M, efConstruction=efConstruction, maxM=maxM, maxM0=maxM0
    )
    index = nmslib.init()
    if os.path.exists(fname):
        log.debug("Loading existing index!")
        index.loadIndex(fname, True)
        with open(fname + ".time", "r") as inf:
            took1 = float(inf.read().strip())
        return index, took1

    ids, data, sents = zip(*db_vec)

    log.debug("Building index..")
    start = time.time()
    index.addDataPointBatch(data=data, ids=ids)
    log.debug("Insert done, building..")
    index.createIndex({
        # 'post': 2
        "M": M,
        "efConstruction": efConstruction,  # 200 - 1000
        "maxM": maxM,
        "maxM0": maxM0,
        "delaunay_type": 2
    }, print_progress=True)
    took1 = time.time() - start
    index.saveIndex(fname)
    with open(fname + ".time", "w") as outf:
        outf.write("%s" % took1)

    log.debug("Done!")
    return index, took1


def cached_generate_vector_data(method, db_size, num_queries):
    fname = NNCACHE_FOLDER + "/%s.%s.vecs" % (method, db_size)
    if os.path.exists(fname):
        log.debug("Vector data is cached, loading..")
        with open(fname, "rb") as inf:
            query_vectors = pickle.load(inf)
        with open("%s.dbvec" % fname, "rb") as inf:
            db_vec = pickle.load(inf)

        return db_vec, query_vectors

    log.debug("Loading (or creating) and converting data to vectors from scratch..")
    loaded = load_or_create_input_dataset(db_size, num_queries)
    db = loaded["samples"]
    q = loaded["queries"]

    log.debug("Generating vector embeddings..")
    query_vectors = [
        (i, vector_embed_sentence(clean_sent(sent), method=method, normalize=True), sent)
        for (i, sent) in q
    ]
    db_vec = [
        (i, vector_embed_sentence(clean_sent(sent), method=method, normalize=True), sent)
        for (i, sent) in db
    ]
    db_vec = [(i, sp, s) for (i, sp, s) in db_vec if sp is not None]
    log.debug("Saving..")
    with open(fname, "wb") as outf:
        pickle.dump(query_vectors, outf)
    with open("%s.dbvec" % fname, "wb") as outf:
        pickle.dump(db_vec, outf)
    log.debug("Done.")

    return db_vec, query_vectors


def run_vectors_benchmark(
        db_size, num_queries, top, method="fasttext", M=16, efConstruction=500, maxM=32, maxM0=64, efSearch=1000
):
    db_vec, query_vectors = cached_generate_vector_data(method, db_size, num_queries)
    queries_gt, gt_took = load_or_generate_ground_truth_data(db_size, num_queries, top, method, db_vec, query_vectors)

    index, index_took = build_nmslib_index(db_vec, db_size, M=M, efConstruction=efConstruction, maxM=maxM, maxM0=maxM0)
    log.debug("Setting query params..")
    index.setQueryTimeParams({
        "ef": efSearch,  # 200 - 1000
    })

    start = time.time()
    log.debug("Running queries..")
    query_ids, query_vectors, queries = zip(*query_vectors)
    res_ids_dists = index.knnQueryBatch(query_spaces=query_vectors, k=top, num_threads=16)
    queries_took = time.time() - start

    approx_results = {
        query_id: list(zip(rids, rdists)) for (query_id, (rids, rdists)) in zip(query_ids, res_ids_dists)
    }
    log.debug("Done. Running evaluation..")

    accuracies, total_accuracy = evaluate_accuracy(approx_results, queries_gt)
    return gt_took, index_took, queries_took, total_accuracy, accuracies

if __name__ == '__main__':
    # db_size = 1000000
    dbsize = 1000000
    numqueries = 1000
    top = 100
    method = "fasttext"
    # debug_quality()
    # run_data_prep()
    space = {
        "method": ["fasttext"],
        "M": [64],
        "efConstruction": [2000],
        "maxM": [64],
        "maxM0": [64],
        "efSearch": [200, 500, 750, 1000, 2000]
    }
    # results = {}
    full_space = dict_product(space)
    with open("./nn_vec_res/nn_vecs_db%s_%sqs_top%sparam_results.txt" % (dbsize, numqueries, top), "w") as outf:
        for i, param_set in enumerate(full_space):
            log.debug("Running combination %s of %s" % (i + 1, len(full_space)))
            method = param_set["method"]
            M = param_set["M"]
            efConstruction = param_set["efConstruction"]
            maxM = param_set["maxM"]
            maxM0 = param_set["maxM0"]
            efSearch = param_set["efSearch"]
            gt_took1, idx_took2, qs_took, total_acc, fine_acc = run_vectors_benchmark(
                dbsize, numqueries, top, method=method, M=M, efConstruction=efConstruction, maxM=maxM, maxM0=maxM0,
                efSearch=efSearch
            )
            sdev = np.std(np.array(fine_acc))
            # results[str(param_set)] = (gt_took1, idx_took2, qs_took, total_acc, fine_acc)
            s1 = "Params: %s" % param_set
            s2 = "accuracy %s, sdev %s, gt %.6s, insert %.6s, lookup %.6s" % (
                total_acc, sdev, gt_took1, idx_took2, qs_took
            )
            log.debug(s1)
            log.debug(s2)
            mult = 10000.
            res = {
                "gt_t": int(gt_took1 * mult) / mult,
                "idx_t": int(idx_took2 * mult) / mult,
                "q_t": int(qs_took * mult) / mult,
                "top%s-acc" % top: int(total_acc * mult) / mult,
                "sd": int(sdev * mult) / mult
            }
            res.update(param_set)
            outf.write(
                json.dumps(res) + "\n"
            )

