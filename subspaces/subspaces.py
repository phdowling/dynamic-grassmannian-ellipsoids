from collections import defaultdict

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
import spacy
from sklearn.decomposition.pca import _infer_dimension_
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.wrappers.fasttext import FastText
import logging

from sklearn.utils.extmath import svd_flip

from subspaces.glove_wrapper import GloveWrapper, CommonCrawlGloveWrapper
from subspaces.util import stopwords, tokenize

log = logging.getLogger(__name__)

_nlp = {}
_fasttext = None
_glove_wrapped = None
_glove_cc_wrapped = None
# TODO change this to the appropriate path on your machine (FastText model .bin file)
FASTTEXT_PATH = "/path/to/bookCorpus/fasttext/model.300.bin"


def get_word_proba(word):
    return get_glove().get_proba(word)


def _get_word_vector(word, method="fasttext"):
    if word in stopwords:
        return None
    if method == "spacy":
        if type(word) == str:
            word = get_nlp("en")(word)[0]
        return word.vector if word.has_vector else None
    elif method == "fasttext":
        try:
            return get_fasttext()[str(word)]
        except KeyError:
            return None
    elif method == "glove":
        return get_glove()[str(word)]

    elif method == "glove_cc":
        return get_glove_cc()[str(word)]


def get_glove():
    global _glove_wrapped
    if _glove_wrapped is None:
        _glove_wrapped = GloveWrapper()
    return _glove_wrapped


def get_glove_cc():
    global _glove_cc_wrapped
    if _glove_cc_wrapped is None:
        _glove_cc_wrapped = CommonCrawlGloveWrapper()
    return _glove_cc_wrapped


def get_fasttext():
    global _fasttext
    if _fasttext is None:
        log.debug("Loading fasttext model..")
        _fasttext = FastText.load_fasttext_format(FASTTEXT_PATH)
    return _fasttext


def get_nlp(lang):
    if lang not in _nlp:
        log.debug("Loading %s spacy pipeline.." % lang)
        _nlp[lang] = spacy.load(lang)
    return _nlp[lang]


def vector_embed_sentence(sentence, alpha_embed=None, method="fasttext", normalize=False, deproject_pc=None):
    if type(sentence) == list:
        tokens = sentence
    else:
        tokens = tokenize(sentence)
    if alpha_embed is not None:

        fvecs = [(_get_word_vector(word, method=method), get_word_proba(word)) for word in tokens]
        vecs_frequencies = [item for item in fvecs if item[0] is not None]

        if not vecs_frequencies:
            return None

        vecs, frequencies = zip(*vecs_frequencies)
        weights = alpha_embed / (alpha_embed + np.array(frequencies))  # a / (a + p(w)), like in arora paper
        # r = (np.array(vecs) * weights[:, np.newaxis]).sum(0) / weights.sum()
        r = (np.array(vecs) * weights[:, np.newaxis]).sum(0) / len(vecs)  # paper says to normalize by sent length
    else:
        vecs = [_get_word_vector(word, method=method) for word in tokens]
        vecs = [v for v in vecs if v is not None]
        r = np.array(vecs).mean(0)

    if not vecs:
        return None

    if deproject_pc is not None:
        projection = deproject_pc.dot(r)
        vecs -= deproject_pc.dot(projection)

    if normalize:
        return r / np.linalg.norm(r)

    return r


def compute_vector_sim(
        sent1, sent2, normalized=False, method="fasttext", _embed=True, deproject_pc=None, alpha_embed=None
):
    if _embed:
        sent1 = vector_embed_sentence(sent1, method=method, deproject_pc=deproject_pc, alpha_embed=alpha_embed)
        sent2 = vector_embed_sentence(sent2, method=method, deproject_pc=deproject_pc)
    if sent1 is None or sent2 is None:
        return 0.0
    if not normalized:
        sent1 /= np.linalg.norm(sent1)
        sent2 /= np.linalg.norm(sent1)
    return float(np.dot(sent1, sent2.T))


def get_pca_components(X, n_components, get_S=False):
    """Same as in sklearn, but we don't center the data"""
    n_samples, n_features = X.shape
    U, S, V = linalg.svd(X, full_matrices=False)
    # flip eigenvectors' sign to enforce deterministic output
    U, V = svd_flip(U, V)

    components_ = V

    # Get variance explained by singular values
    explained_variance_ = (S ** 2) / n_samples
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var

    # Postprocess the number of components required
    if n_components == 'mle':
        n_components = _infer_dimension_(explained_variance_, n_samples, n_features)
    elif 0 < n_components < 1.0:
        # number of components for which the cumulated explained
        # variance percentage is superior to the desired threshold
        ratio_cumsum = explained_variance_ratio_.cumsum()
        n_components = np.searchsorted(ratio_cumsum, n_components) + 1

    # Compute noise covariance using Probabilistic PCA model
    # The sigma2 maximum likelihood (cf. eq. 12.46)
    # if n_components < min(n_features, n_samples):
    #     noise_variance_ = explained_variance_[n_components:].mean()
    # else:
    #     noise_variance_ = 0.

    components_ = components_[:n_components]
    # explained_variance_ = explained_variance_[:n_components]
    # explained_variance_ratio_ = explained_variance_ratio_[:n_components]
    # if get_explained_var:
    #     return components_, (explained_variance_ratio_[:n_components])
    if get_S:
        return components_, explained_variance_ratio_[:n_components], S[:n_components]

    return components_, explained_variance_ratio_[:n_components]


def get_wpca_components(X, weights, n_components, get_S=False):  # , xi=0):
    """Same as in sklearn, but we don't center the data"""
    # weights = np.repeat(weights[:, np.newaxis], X.shape[1], axis=1)

    weights = weights[:, np.newaxis]
    X_ = X * weights
    covar = np.dot(X_.T, X_)
    covar /= np.dot(weights.T, weights)
    covar[np.isnan(covar)] = 0

    # enhance weights if desired
    # if xi != 0:
    #     Ws = weights.sum(0)
    #     covar *= np.outer(Ws, Ws) ** xi

    eigvals = (0, X.shape[1] - 1)
    evals, evecs = linalg.eigh(covar, eigvals=eigvals)
    components_ = evecs[:, ::-1].T
    explained_variance_ = evals[::-1]
    explained_variance_ratio_ = evals[::-1] / covar.trace()
    n_samples, n_features = X_.shape

    # Postprocess the number of components required
    if n_components == 'mle':
        n_components = _infer_dimension_(explained_variance_, n_samples, n_features)
    elif 0 < n_components < 1.0:
        # number of components for which the cumulated explained
        # variance percentage is superior to the desired threshold
        ratio_cumsum = explained_variance_ratio_.cumsum()
        n_components = np.searchsorted(ratio_cumsum, n_components) + 1

    components_ = components_[:n_components]
    explained_variance_ratio_ = explained_variance_ratio_[:n_components]

    if get_S:
        return components_, explained_variance_ratio_, evals[:n_components]

    return components_, explained_variance_ratio_


dyn_embedding_sizes = defaultdict(lambda: defaultdict(int))
dyn_embedding_len_sizes = defaultdict(lambda: [[0]*300]*300)


def subspace_embed_sentence(
        sentence, rank_or_energy=0.9, method="spacy", alpha=None, _return_effective_len=False, _substract_mean=False,
        deproject_pc=None, _get_S=False
):
    # nlp = get_nlp(lang)
    sentence = str(sentence)
    sent = tokenize(sentence)
    if alpha is not None:
        vecs_ws = [
            (
                _get_word_vector(word, method=method),
                alpha / (alpha + get_word_proba(word))
            ) for word in sent
            ]
        vecs_ws = [v for v in vecs_ws if v[0] is not None]

        if len(vecs_ws) == 0:
            return None

        vecs, weights = list(map(np.array, zip(*vecs_ws)))
        vecs = np.array(vecs)
        if deproject_pc is not None:
            deproject_pc = deproject_pc[:, np.newaxis]
            projection = vecs.dot(deproject_pc)
            vecs -= projection.dot(deproject_pc.T)

        if _substract_mean:
            # TODO this probably needs to be a weighted mean! (but in practice we don't subtract mean anyway))
            mean = vecs.mean(axis=0)
            vecs -= mean  # [np.newaxis, :]
        representation = get_wpca_components(vecs, weights, rank_or_energy, get_S=_get_S)
    else:
        vecs = [_get_word_vector(word, method=method) for word in tokenize(sentence)]
        vecs = [v for v in vecs if v is not None]

        if len(vecs) == 0:
            return None

        vecs = np.array(vecs)
        if deproject_pc is not None:
            deproject_pc = deproject_pc[:, np.newaxis]
            projection = vecs.dot(deproject_pc)
            vecs -= projection.dot(deproject_pc.T)

        if _substract_mean:
            mean = vecs.mean(axis=0)
            vecs -= mean  # [np.newaxis, :]
        representation = get_pca_components(vecs, rank_or_energy, get_S=_get_S)

    if _return_effective_len:
        return representation, len(vecs)

    if _substract_mean:
        representation = list(representation)
        representation.append(mean)
        representation = tuple(representation)

    return representation


def subspace_similarity(subspace_1, subspace_2, beta=0.):
    if type(subspace_1) == tuple and type(subspace_2) == tuple:
        subspace_1, w1 = subspace_1
        subspace_2, w2 = subspace_2
    else:
        assert beta == 0.

    rank1, rank2 = len(subspace_1), len(subspace_2)
    rank = min(rank1, rank2)

    prod = np.dot(subspace_1, subspace_2.T)

    if beta != 0:  # apply length weight
        minima = np.minimum.outer(w1, w2)
        maxima = np.maximum.outer(w1, w2)
        match = minima / maxima  # percentage difference of axes' importances
        match_adjusted = 1. - beta * (1. - match)  # adjust to assign a lower importance
        prod *= match_adjusted  # apply re-weighting to cossim

    sim = np.linalg.norm(prod) / np.sqrt(rank)
    # u, s, v = svd(prod)
    # sim = np.sqrt((s**2).sum()) / np.sqrt(len(s))
    return sim


def compute_subspace_sim(sent1, sent2, rank_or_energy=0.9, alpha=None, beta=0., method="fasttext", deproject_pc=None):
    """
    Compute the similarity of two strings using grassmanian subspaces with weighted norm-adjusted cosine similarity.
    :param sent1: The first sentence
    :param sent2: Second sentence
    :param rank_or_energy: if int: the number of components. if float (0 < x < 1): the proportion of variance to keep.
    :param alpha: parameter for word-reweighting in computing representation spaces.
        Must be larger than 0, larger values decrease the severity of reweighting (weights always closer to 1)
    :param beta: parameter for norm-adjustment in space similarity, between 0 and 1. Zero means no adjustment,
        1 vector norms are fully multiplied into cosine similarities
    :param method: what vectors to use (prefer fasttext)
    :return: float similarity between 0 and 1
    """
    s1 = subspace_embed_sentence(
        sent1, rank_or_energy=rank_or_energy, method=method, alpha=alpha, deproject_pc=deproject_pc)
    s2 = subspace_embed_sentence(
        sent2, rank_or_energy=rank_or_energy, method=method, alpha=alpha, deproject_pc=deproject_pc)
    if s1 is None:
        log.warn("no embeddings for '%s'" % sent1)
        return 0.0
    if s2 is None:
        log.warn("no embeddings for '%s'" % sent2)
        return 0.0
    return subspace_similarity(s1, s2, beta=beta)

