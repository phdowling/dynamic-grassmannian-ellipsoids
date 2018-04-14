import os
import numpy as np
import logging
import pickle
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


class GloveWrapper(object):
    # TODO set path to glove vectors
    def __init__(self, path="/path/to/bookCorpus/glove/"):
        vocab_file = os.path.join(path, "vocab.txt")
        vector_file = os.path.join(path, "vectors.txt")
        self.vocab = {}
        self.vocab_count = {}
        self.total_words = 0.
        log.debug("Loading from %s.." % path)
        log.debug("Reading vocab..")

        with open(vocab_file, "r") as inf:
            for i, line in enumerate(inf):
                d = line.strip().split(" ")
                if len(d) != 2:
                    log.error("Can't parse line %s: %s" % (i, line))
                    continue
                word, count = d
                self.vocab[word] = i
                c = int(count)
                self.vocab_count[word] = c
                self.total_words += c

        log.debug("Read %s vocab entries (%s lines processed)" % (len(self.vocab), i + 1))
        self.vectors = np.zeros(shape=(i + 2, 300))
        self.vectors_3d = None
        log.debug("Reading %s vectors.." % (i + 2))

        with open(vector_file, "r") as inf:
            for i, line in enumerate(inf):
                raw = line.strip().split(" ")[1:]
                if len(raw) != 300:
                    log.error("Couldn't read vector on line %s." % i)
                    continue
                vector = np.array([float(n) for n in raw])
                self.vectors[i] = vector

    def __getitem__(self, word):
        if word in self.vocab:
            return self.vectors[self.vocab[word]]
        return None  # self.vectors[-1]  # UNK token

    def get_three_dim_projection(self, word):
        if word in self.vocab:
            if self.vectors_3d is None:
                log.debug("Computing 3D glove vector projection")
                self.vectors_3d = PCA(3).fit_transform(self.vectors)
            return self.vectors_3d[self.vocab[word]]
        return None

    def get_proba(self, word):
        return self.vocab_count.get(word, 1.0) / self.total_words

    def get_nns(self, query_vec, k=20):
        q_v = query_vec / np.linalg.norm(query_vec)
        scored = (
            (
                word,
                (self.vectors[self.vocab[word]] / np.linalg.norm(self.vectors[self.vocab[word]])).dot(q_v)
            )
            for word in self.vocab
        )
        return sorted(scored, key=lambda i: -i[1])[:k]

class CommonCrawlGloveWrapper(object):
    # TODO set path to pre-trained glove vectors
    def __init__(self, path="/path/to/glove.840B.300d.txt"):
        self.path = path
        self.vocab = {}

        log.debug("Reading glove vectors..")

        binpath = path + ".saved.pkl"
        if os.path.exists(binpath):
            with open(binpath, "rb") as inf:
                self.vectors, self.vocab = pickle.load(inf)
            log.debug("Loaded cached vectors.")
        else:
            self.vectors = []
            log.debug("Loading raw from file..")

            with open(path, "r") as inf:
                i = 0
                for li, line in enumerate(inf):
                    word, raw = line.strip().split(" ", maxsplit=1)
                    raw = raw.split(" ")
                    if len(raw) != 300:
                        log.error("Couldn't read vector on line %s." % i)
                        continue

                    self.vocab[word] = i
                    vector = np.array([float(n) for n in raw])
                    self.vectors.append(vector)
                    i += 1
            log.debug("Read %s entries (%s lines processed)" % (len(self.vocab), li + 1))
            self.vectors = np.vstack(self.vectors)
            with open(binpath, "wb") as outf:
                pickle.dump((self.vectors, self.vocab), outf, protocol=4)

    def __getitem__(self, word):
        id = self.vocab.get(word, None)
        if id is not None:
            return self.vectors[id]
        return None  # self.vectors[-1]  # UNK token

    def get_nns(self, query_vec, k=20):
        q_v = query_vec / np.linalg.norm(query_vec)
        scored = (
            (
                word,
                (self.vectors[self.vocab[word]] / np.linalg.norm(self.vectors[self.vocab[word]])).dot(q_v)
            )
            for word in self.vocab
        )
        return sorted(scored, key=lambda i: -i[1])[:k]

