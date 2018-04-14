import annoy
from collections import defaultdict
import numpy as np
from subspaces.subspaces import subspace_similarity


class ANSS(object):
    def __init__(self, dimensionality):
        self.dimensionality = dimensionality
        self.ann = annoy.AnnoyIndex(dimensionality, metric="angular")

        self._i_vec = 0
        self._i_sub = 0
        self._subspace_to_data = {}
        self._subspace_ranks = defaultdict(int)
        self._subspaces_to_vectors = {}
        self._vectors_to_subspace = {}

    def add_sample(self, subspace, data=None):
        space = []
        for vector in subspace:
            self.ann.add_item(self._i_vec, vector)
            self._vectors_to_subspace[self._i_vec] = self._i_sub
            self._subspace_ranks[self._i_sub] += 1
            space.append(vector)
            self._i_vec += 1

        self._subspaces_to_vectors[self._i_sub] = np.array(space)

        self._subspace_to_data[self._i_sub] = data
        self._i_sub += 1

    def build(self, n_trees=1000):
        self.ann.build(n_trees)

    def query(self, subspace, max_nns=None, exact=True, _search_nns=100, _search_k=100000):
        rank = len(subspace)
        neighbors = defaultdict(list)
        partial_subspace_similarities = defaultdict(float)  # init all to 0
        # PHASE 1: scan for nearest neighbors of each basis vector of query subspace
        for i, vector in enumerate(subspace):
            for sign in [1, -1]:  # TODO do we need this in our case?
                ids, distances = self.ann.get_nns_by_vector(
                    sign * vector, _search_nns, search_k=_search_k, include_distances=True
                )
                for _id, dist in zip(ids, distances):
                    neighbors[sign * i].append((_id, dist))
        # PHASE 2: aggregate the scores per subspace to approximate their distance
        for _, point_neighbors in neighbors.items():
            for (_id, dist) in point_neighbors:
                # angular distance in annoy is euclidean distance of normalized vectors i.e.
                # dist = sqrt(2.(1.- (p^T . q)))
                # thus we can get the dot product (==cos similarity bc normalized vecs) directly via:
                # (p^T q) = 1. - dist**2 / 2.
                # Further, we want to compute, for each subspace:
                # Sqrt(1 / rank(Sum over i, j (p_i^T q_j^T) ^ 2))
                # iteratively. Therefore we sum up the squares of the dot products!
                dot_prod = (1. - (dist ** 2) / 2.)
                subspace_id = self._vectors_to_subspace[_id]
                partial_subspace_similarities[subspace_id] += dot_prod ** 2
                # later on we will divide this by the rank and then take the square root

        if exact:
            # compute exact similarities for candidates
            final_estimates = list(sorted(
                [
                    (
                        self._subspace_to_data[sub_id],
                        subspace_similarity(subspace, self._subspaces_to_vectors[sub_id])
                    ) for sub_id, _ in partial_subspace_similarities.items()
                ], key=lambda item: -item[1]
            ))
        else:
            # only compute the incremental estimates (avoids full sum)
            # TODO can we get better estimates via expected value?
            final_estimates = list(sorted(
                [
                    (
                        self._subspace_to_data[sub_id],
                        np.sqrt(dist_part / min(rank, self._subspace_ranks[sub_id]))
                    ) for sub_id, dist_part in partial_subspace_similarities.items()
                    ], key=lambda item: -item[1]
            ))

        return list(zip(*final_estimates[:max_nns]))  # two lists, first data, then distances
