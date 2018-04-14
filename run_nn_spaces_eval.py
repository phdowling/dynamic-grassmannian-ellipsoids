from operator import itemgetter
import os

from run_nn_vectors_eval import load_or_create_input_dataset, clean_sent, evaluate_accuracy, dict_product, NNCACHE_FOLDER
from subspaces.subspaces import subspace_embed_sentence, subspace_similarity, stopwords
from subspaces.optimize_subspace_rotations import optimize
from random import seed
import logging
import string
import time
from subspaces.nearest_subspace.nmslanns import BetterANSS
from subspaces.nearest_subspace.randomized_anss import RandomizedANSS
import pickle
import numpy as np
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

seed(1234)

letters = list("зсьовфдагтурйпб«эыинямжчеклю»ш")

translator = str.maketrans('', '', string.punctuation)


def load_or_generate_ground_truth_data(db_size, num_queries, top, method, db_spaces, query_spaces):
    fname = NNCACHE_FOLDER + "/%s.db%s.%sqs.top%s.spaces.gt" % (method, db_size, num_queries, top)
    if os.path.exists(fname):
        log.debug("Loading gt data from %s" % fname)
        with open(fname, "rb") as inf:
            (queries_gt, took) = pickle.load(inf)
        for key in queries_gt:  # only keep top
            queries_gt[key] = queries_gt[key][:top]
        log.debug("gt data was cashed (originally took %s seconds to generate)" % took)
        return queries_gt, took

    queries_gt = {}
    start = time.time()
    for i__, (id_, qsp, query) in enumerate(query_spaces):
        log.debug("%s: Computing similarities for %s (db spaces size: %s).." % (i__, query, len(db_spaces)))
        similarities = [
            (i, subspace_similarity(qsp, sample), sent) for i, sample, sent in db_spaces
        ]
        similarities = list(reversed(sorted(similarities, key=itemgetter(1))))  # largest sims first
        queries_gt[id_] = similarities[:top]

    took = time.time() - start
    log.debug("Took %s seconds. Saving gt data to %s." % (took, fname))
    with open(fname, "wb") as outf:
        pickle.dump((queries_gt, took), outf)
    return queries_gt, took


def build_anss_index(
        db_spaces, db_size, M=16, efConstruction=500, maxM=32, maxM0=64, idx_type="rand",
        random_spaces=1, optimized=False
):
    if idx_type == "rand":
        index = RandomizedANSS(random_spaces)  # TODO parameterize
    else:
        index = BetterANSS()

    # TODO adapt this location to your machine

    fname = NNCACHE_FOLDER + "/anssidx_{db_size}-{M}-{efConstruction}-{maxM}-{maxM0}-{index_type}".format(
        db_size=db_size, M=M, efConstruction=efConstruction, maxM=maxM, maxM0=maxM0, index_type=idx_type
    )

    if optimized:
        fname += "%s_optimized" % optimized

    if os.path.exists(fname):
        log.debug("Loading existing index from %s!" % fname)
        index.loadIndex(fname, True)
        with open(fname + ".time", "r") as inf:
            took1 = float(inf.read().strip())
        return index, took1
    else:
        log.debug("No index at %s, constructing." % fname)

    ids, data, sents = zip(*db_spaces)

    log.debug("Building index..")
    start = time.time()
    index.addDataPointBatch(data=data, ids=ids)
    log.debug("Insert done, building..")
    index.createIndex({
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


def cached_generate_spaces_data(method, db_size, num_queries, rank_or_energy=0.9):
    fname = NNCACHE_FOLDER + "/%s.%s.e%s.spaces" % (method, db_size, rank_or_energy)
    if os.path.exists(fname):
        log.debug("Spaces data is cached, loading..")
        with open(fname, "rb") as inf:
            query_spaces = pickle.load(inf)
        with open("%s.dbspaces" % fname, "rb") as inf:
            db_spaces = pickle.load(inf)

        return db_spaces, query_spaces

    log.debug(
        "Loading (or creating) %s sents (%s query) and converting data to subspaces from scratch.."
        % (db_size, num_queries)
    )
    loaded = load_or_create_input_dataset(db_size, num_queries)
    db = loaded["samples"]
    q = loaded["queries"]
    log.debug("Have %s db sents and %s query" % (len(db), len(q)))

    log.debug("Generating space embeddings..")
    query_spaces = [
        (i, subspace_embed_sentence(clean_sent(sent), method=method, rank_or_energy=rank_or_energy), sent)
        for (i, sent) in q
    ]
    db_spaces = [
        (i, subspace_embed_sentence(clean_sent(sent), method=method, rank_or_energy=rank_or_energy), sent)
        for (i, sent) in db
    ]
    db_spaces = [(i, sp, s) for (i, sp, s) in db_spaces if sp is not None]
    log.debug("Saving..")
    with open(fname, "wb") as outf:
        pickle.dump(query_spaces, outf)
    with open("%s.dbspaces" % fname, "wb") as outf:
        pickle.dump(db_spaces, outf)
    log.debug("Done.")

    log.debug("Finally returning %s db spaces and %s query" % (len(db_spaces), len(query_spaces)))

    return db_spaces, query_spaces


def cached_optimize_space(db_spaces, query_spaces, method, db_size, rank_or_energy=0.9, loss_fun="trace"):
    fname = NNCACHE_FOLDER + "/%s.%s.e%s.spaces.%s_optimized" % (method, db_size, rank_or_energy, loss_fun)

    if os.path.exists(fname):
        log.debug("Optimized spaces data is cached, loading..")
        with open(fname, "rb") as inf:
            query_spaces_new = pickle.load(inf)
        with open("%s.dbspaces" % fname, "rb") as inf:
            db_spaces_new = pickle.load(inf)

        return db_spaces_new, query_spaces_new

    log.debug("Optimizing subspaces (query, %s spaces).." % len(query_spaces))
    query_spaces_new = []
    for i, (space_id, (space, imp), sent) in enumerate(query_spaces):
        if i % 100 == 0:
            log.debug("%s / %s optimized" % (i, len(query_spaces)))
        query_spaces_new.append((space_id, (optimize(space, loss_fun=loss_fun), imp), sent))

    log.debug("Optimizing subspaces (db, %s spaces).." % len(db_spaces))
    db_spaces_new = []
    for i, (space_id, (space, imp), sent) in enumerate(db_spaces):
        if i % 100 == 0:
            log.debug("%s / %s optimized" % (i, len(db_spaces)))
        db_spaces_new.append((space_id, (optimize(space, loss_fun=loss_fun), imp), sent))

    # db_spaces = [(space_id, (optimize(space), imp), sent) for (space_id, (space, imp), sent) in db_spaces]

    log.debug("Saving..")
    with open(fname, "wb") as outf:
        pickle.dump(query_spaces_new, outf)
    with open("%s.dbspaces" % fname, "wb") as outf:
        pickle.dump(db_spaces_new, outf)
    log.debug("Done.")

    return db_spaces_new, query_spaces_new


def run_spaces_benchmark(
        db_size, num_queries, top, method="fasttext", subquery_k=10, M=16, efConstruction=500, maxM=32, maxM0=64,
        efSearch=1000, sim="exact", idx_type="normal", random_spaces=5, prefilter=None, optimize_spaces=False,
        signflip=True
):
    db_spaces, query_spaces = cached_generate_spaces_data(method, db_size, num_queries)
    queries_gt, gt_took = load_or_generate_ground_truth_data(db_size, num_queries, top, method, db_spaces, query_spaces)

    if optimize_spaces:
        db_spaces, query_spaces = cached_optimize_space(
            db_spaces, query_spaces, method, db_size, loss_fun=optimize_spaces
        )

    index, index_took = build_anss_index(
        db_spaces, db_size, M=M, efConstruction=efConstruction, maxM=maxM, maxM0=maxM0, idx_type=idx_type,
        random_spaces=random_spaces, optimized=optimize_spaces
    )
    log.debug("Setting query params..")
    index.setQueryTimeParams({
        "ef": efSearch,  # 200 - 1000
    })

    start = time.time()
    log.debug("Running queries..")
    query_ids, query_spaces, queries = zip(*query_spaces)

    res_ids_dists = index.knnQueryBatch(
        query_spaces=query_spaces, subquery_k=subquery_k, k=top, num_threads=16, sim=sim, prefilter_final=prefilter,
        signflip=signflip
    )
    # print(res_ids_dists)
    queries_took = time.time() - start

    approx_results = {
        query_id: list(zip(rids, rdists)) for (query_id, (rids, rdists)) in zip(query_ids, res_ids_dists)
    }

    log.debug("Done. Running evaluation..")

    accuracies, total_accuracy = evaluate_accuracy(approx_results, queries_gt)

    return gt_took, index_took, queries_took, total_accuracy, accuracies


def run_eval():
    dbsize, numqueries = 10000, 1000  # use this for full experiment later
    # dbsize, numqueries = 2000,  100  # just for testing
    # dbsize, numqueries = 1000000, 1000  # use this for full experiment later
    top = 10

    space = {
        "method": ["fasttext"],
        # "subquery_k": [10, 50, 75, 100, 250, 400],  # 400  # this is c
        "subquery_k": [1, 5, 10, 20, 50, 100, 200, 350, 500, 600],  # 400  # this is c
        # "subquery_k": [5, 10, 20, 50, 100, 200],  # 400  # this is c
        "M": [32],
        "efConstruction": [2000],
        "maxM": [64],
        "maxM0": [64],
        # "efSearch": [50, 100],  # 1000
        "efSearch": [50],  # , 100],  # 1000
        "sim":  ["exact", "standard_approx"],  # "hitcount", "extrapolated_approx" is broken
        "idx_type": ["normal"],  # , "rand", ],
        # "random_spaces": [0],
        "prefilter": [None, 0.2],  # 0.25
        "optimize_spaces": [False, "trace", "nearest", "max"],  # "L1+trace", "L1", "trace"
        # "optimize_spaces": ["nearest"],  # "L1+trace", "L1", "trace"
        # "optimize_spaces": ["max"],  # "L1+trace", "L1", "trace"
    }
    results = {}
    full_space = dict_product(space)
    fname = "./nn_vec_res/nn_spaces_db%s_%sqs_top%sparam_results.txt" % (dbsize, numqueries, top)
    ground_truth_time = None
    identifiers = set()
    with open(fname, "w") as outf:
        for i, param_set in enumerate(full_space):
            log.debug("Running combination %s of %s" % (i + 1, len(full_space)))
            for key in sorted(param_set.keys()):
                log.debug("\t%s: %s" % (key, param_set[key]))
            method = param_set["method"]
            M = param_set["M"]
            efConstruction = param_set["efConstruction"]
            maxM = param_set["maxM"]
            maxM0 = param_set["maxM0"]
            efSearch = param_set["efSearch"]
            subquery_k = param_set["subquery_k"]
            sim = param_set["sim"]
            idx_type = param_set["idx_type"]
            # random_spaces = param_set["random_spaces"]
            prefilter = param_set["prefilter"]
            optimize_spaces = param_set["optimize_spaces"]

            if sim != "exact" and prefilter is not None:
                log.debug("Skipping (prefilter not needed with approximate similarity)")
                continue  # we can skip this, it has no effect

            # if idx_type == "rand" and random_spaces != 0:
            #     name = "rand(%s)_%s_efs%s_filt(%s)" % (random_spaces, sim, efSearch, prefilter)
            #     specname = "rand(%s)_%s_subq%s_efs%s_filt(%s)" % (random_spaces, sim, subquery_k, efSearch, prefilter)
            # else:
            if sim == "standard_approx":
                simstr = "approx"
            else:
                simstr = sim
            filt_str = ("_filter=%s" % prefilter) if prefilter else ""
            name = "%s%s" % (simstr, filt_str)
            specname = "%s_sq=%s%s" % (simstr, subquery_k, filt_str)

            if optimize_spaces:
                name += "_" + optimize_spaces
                specname += "_" + optimize_spaces

            identifier = "%s\t%s" % (name, specname)
            if identifier in identifiers:
                continue
            identifiers.add(identifier)
            if i == 0:  # run once so that "warm up" time is not counted
                log.debug("DRY RUN")
                run_spaces_benchmark(
                    dbsize, numqueries, top, subquery_k=subquery_k, sim=sim, M=M, efConstruction=efConstruction,
                    maxM=maxM, maxM0=maxM0, efSearch=efSearch, method=method, idx_type=idx_type,
                    prefilter=prefilter, optimize_spaces=optimize_spaces
                )
                log.debug("DRY RUN COMPLETE")

            gt_took1, idx_took2, qs_took, total_acc, fine_acc = run_spaces_benchmark(
                dbsize, numqueries, top,
                subquery_k=subquery_k, sim=sim,
                M=M, efConstruction=efConstruction, maxM=maxM, maxM0=maxM0, efSearch=efSearch, method=method,
                idx_type=idx_type, prefilter=prefilter, optimize_spaces=optimize_spaces
            )
            if ground_truth_time is None:
                ground_truth_time = gt_took1 / numqueries
                for x in range(1, 11):
                    frac = x / 10.
                    outf.write("brute-force\tbrute-force100\t0.\t%s\t%s\n" % (frac * ground_truth_time, frac))
            else:
                assert ground_truth_time == gt_took1 / numqueries

            sdev = np.std(np.array(fine_acc))
            results[str(param_set)] = (gt_took1, idx_took2, qs_took, total_acc, fine_acc)
            log.debug("Params:")
            for key in sorted(param_set.keys()):
                log.debug("\t%s: %s" % (key, param_set[key]))
            s2 = "accuracy %s, sdev %s, gt %.6s, insert %.6s, lookup %.6s" % (
                total_acc, sdev, gt_took1, idx_took2, qs_took
            )

            log.debug(s2)

            build_time = idx_took2
            search_time = qs_took / numqueries  # seconds per query

            res = [build_time, search_time, total_acc]

            outf.write("%s\t%s" % (identifier, "\t".join(map(str, res)) + "\n"))

    log.debug("Wrote to %s" % fname)


if __name__ == '__main__':
    run_eval()