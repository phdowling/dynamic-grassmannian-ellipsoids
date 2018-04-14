from collections import defaultdict

import numpy as np
from flask import Flask, request, jsonify, Response, Blueprint
from flask import send_from_directory
from werkzeug.serving import run_simple
from subspaces.glove_wrapper import CommonCrawlGloveWrapper
from subspaces.subspaces import subspace_embed_sentence, get_pca_components, _get_word_vector
from subspaces.util import translator
from subspaces.util import tokenize
import requests
import logging

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)
log = logging.getLogger(__name__)

prox_blueprint = Blueprint("prox", __name__)

# TODO get access to the inquire API (sorry, I know this might be difficult :/ )
INQUIRE_URL = "http://some.inquire.server.stanford.edu:9001"

app = Flask("vizserver", static_folder="")



@app.route('/')
def root():
    print("Hello")
    return app.send_static_file('sentviz_static/index.html')


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('sentviz_static', path)


@app.route("/query_weighted")
def query_w():
    print(request.args)
    words = request.args.get("words").split("|")

    if "weights" in request.args:
        weights_raw = request.args.get("weights").split("|")
        weights_raw = list(map(float, weights_raw))
        weights = np.array(weights_raw)
    else:
        weights = np.array([1.] * len(words))

    print(words)
    print(weights)

    assert len(words) == len(weights)
    vecs = [_get_word_vector(word, method="glove_cc") for word in words]
    words_weights_vecs = [(w, w_, v) for w, w_, v in zip(words, weights, vecs) if v is not None]
    words, weights, vecs = zip(*words_weights_vecs)
    weights = np.array(weights)
    vecs = np.vstack(vecs)
    query_v = (vecs * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
    query_v = map(lambda i: str(float(i)), query_v)
    # TODO merge results for negative vector also
    req = requests.get(INQUIRE_URL + "/query?data=x&query_vector=" + "|".join(query_v))
    response = req.json()
    return jsonify(response)


@app.route("/query_by_vec")
def query_v():
    print(request.args)
    query_v = map(float, request.args.get("vec").split("|"))
    query_v = np.array(list(map(float, query_v)))
    # TODO merge results for negative vector also
    req1 = requests.get(INQUIRE_URL + "/query?data=x&query_vector=" + "|".join(map(str, query_v)))
    response1 = req1.json()
    req2 = requests.get(INQUIRE_URL + "/query?data=x&query_vector=" + "|".join(map(str, -query_v)))
    response2 = req2.json()
    results_1 = response1["query_results"][:]
    results_2 = response2["query_results"][:]
    all_results = results_1 + results_2
    all_results.sort(key=lambda item: -item["similarity"])
    response1["query_results"] = all_results
    return jsonify(response1)

def norm_vec(wv, mean):
    wv_norm = wv - mean  # vec is relative to mean, so adjust the words too
    wv_norm /= np.linalg.norm(wv_norm)
    return wv_norm


def toks_word_vecs(sent):
    vecs_ws = [(_get_word_vector(word, method="glove_cc"), word) for word in sent]
    words_exist = [v is not None for (v, w) in vecs_ws]

    vecs_ws = dict([(w, v) for (v, w) in vecs_ws if v is not None]).items()
    return list(zip(*vecs_ws)), words_exist


@app.route("/embed", methods=["GET"])
def get_embed2(_request=None, subtract_mean=True, energy=0.9):
    if _request is None:
        _request = request

    sentence = _request.args.get("sentence")
    alpha = _request.args.get("alpha", 0.001)

    if alpha is not None:
        alpha = float(alpha)

    doc = sentence.translate(translator).strip()
    if subtract_mean:
        vecs, imps, mean = subspace_embed_sentence(
            doc, energy, alpha=alpha, method="glove_cc", _substract_mean=True
        )
    else:
        vecs, imps = subspace_embed_sentence(
            doc, energy, alpha=alpha, method="glove_cc", _substract_mean=False
        )
        mean = np.zeros(shape=(300,))

    doc = tokenize(doc, filter_stopwords=False)
    (toks, wvs), exist_flags = toks_word_vecs(doc)

    result_topics = {}
    word_top_topic_scores = defaultdict(float)
    word_top_topics = {}
    # similarities_by_dim = []
    for i, (vec, imp) in enumerate(zip(vecs, imps)):
        print("topic %s" % i)
        words_ordered = sorted(
            [(tok, np.abs(norm_vec(wv, mean).dot(vec))) for (tok, wv) in zip(toks, wvs)], key=lambda kv: -kv[1]
        )
        # similarities_by_dim.append(sims)
        for word, simil in words_ordered:
            if simil >= word_top_topic_scores[word]:
                word_top_topics[word] = i
                word_top_topic_scores[word] = simil

        result_topics[i] = {"importance": float(imp), "words": dict(words_ordered)}


    return jsonify(
        {
            "words_found": exist_flags, "words": doc, "sent_topics": result_topics, "topics_by_word": word_top_topics,
            "topic_vectors": dict(zip(
                range(len(imps)),
                list(map(lambda v: list(map(float, v)), vecs))
            ))
        }
    )


def get_embed(_request=None):
    if _request is None:
        _request = request

    sentence = _request.args.get("sentence")
    alpha = _request.args.get("alpha", None)

    if alpha is not None:
        alpha = float(alpha)

    word_vecs, words, words_exist = get_word_vectors(sentence)
    if len(word_vecs):
        vecs = word_vecs - word_vecs.mean(axis=0)# [:, np.newaxis]
        vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]

        dim_vectors, expl_var_ratios = get_pca_components(vecs, n_components=0.9)

        similarities_by_dim = np.abs(np.dot(dim_vectors, vecs.T))
    else:
         dim_vectors, similarities_by_dim, expl_var_ratios = [], [], []

    # result_topics = []
    result_topics = {}
    word_top_topic_scores = defaultdict(float)
    word_top_topics = {}
    for i, (topic_vector, importance) in enumerate(zip(similarities_by_dim, expl_var_ratios)):
        topic_vector /= float(topic_vector.mean())
        sim = zip(words, map(float, topic_vector))
        words_ordered = sorted(sim, key=lambda kv: -kv[1])
        for word, simil in words_ordered:
            if simil >= word_top_topic_scores[word]:
                word_top_topics[word] = i
                word_top_topic_scores[word] = simil
        # result_topics.append(((i, float(importance)), words_ordered))
        result_topics[i] = {"importance": float(importance), "words": dict(words_ordered)}

    # similarities /= similarities.sum(axis=1)[:, np.newaxis]

    return jsonify(
        {
            "words_found": words_exist, "words": words, "sent_topics": result_topics, "topics_by_word": word_top_topics,
            "topic_vectors": dict(zip(
                range(len(dim_vectors)),
                list(map(lambda v: list(map(float, v)), dim_vectors))
            ))
        }
    )


def get_word_vectors(sentence):
    sentence = sentence.translate(translator).strip()
    words_orig = tokenize(sentence, filter_stopwords=False)
    print(sentence)
    print(words_orig)

    word_vecs = [_get_word_vector(word, method="glove_cc") for word in words_orig]

    words_exist = [v is not None for v in word_vecs]
    print(words_exist)
    words_and_vecs = [(w, v) for w, v in zip(words_orig, word_vecs) if v is not None]

    if not words_and_vecs:
        return [], words_orig, words_exist

    words, word_vecs = zip(*words_and_vecs)
    word_vecs = np.array(word_vecs)
    return word_vecs, words_orig, words_exist


class Request(object):
    def __init__(self, args):
        self.args = args

app.register_blueprint(prox_blueprint, )


def debug_local():
    while True:
        res = get_embed(Request({"sentence": input("Enter sentence: ")}))

        print("-----------------------------------")
        for (topic_id, importance), word_dist in res:
            word_dist = ["'%s', %.6s" % (word, sim) for word, sim in word_dist]
            print("Topic %s (%.4s)" % (topic_id,importance) + ":\t" + "\t".join(map(str, word_dist[:8])))


# todo create api call to compute similarity of sentence (maybe we can provide more than the raw score?)
if __name__ == "__main__":
    # debug_local()
    run_simple("0.0.0.0", 8080, application=app)
