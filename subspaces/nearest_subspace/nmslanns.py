import nmslib
from collections import defaultdict
import numpy as np
from subspaces.subspaces import subspace_similarity
import pickle
import logging
import itertools
log = logging.getLogger(__name__)


# index.addDataPointBatch(data=data, ids=ids)
#     log.debug("Insert done, building..")
#     index.createIndex({
#         # 'post': 2
#         "M": M,
#         "efConstruction": efConstruction,  # 200 - 1000
#         "maxM": maxM,
#         "maxM0": maxM0,
#         "delaunay_type": 2
#     }, print_progress=True)
#     took1 = time.time() - start
#     index.saveIndex(fname)


class BetterANSS(object):
    def __init__(self):
        self.ann = nmslib.init()

        self._i_indexed_vecs = 0
        self._i_sub = 0
        self._subspace_id_to_external_id = {}
        self._subspace_ranks = defaultdict(int)
        self._subspaces_to_basis_vectors = {}
        self._indexed_vectors_to_subspace = {}
        self.vecs_per_space_dim = 1

    # def get_vectors_to_index(self, subspace, dim_weights=1):
    #     return subspace

    def get_vectors_for_query(self, subspace, dim_weights=1):
        return subspace

    def addDataPointBatch(self, data, ids):
        for idx, (subspace, dim_weights) in enumerate(data):

            for vector in subspace:
                self.ann.addDataPoint(self._i_indexed_vecs, vector)
                self._indexed_vectors_to_subspace[self._i_indexed_vecs] = self._i_sub
                self._i_indexed_vecs += 1

            self._subspace_ranks[self._i_sub] = len(subspace)
            self._subspaces_to_basis_vectors[self._i_sub] = np.array(subspace)
            self._subspace_id_to_external_id[self._i_sub] = ids[idx]
            self._i_sub += 1

    def createIndex(self, params, print_progress=True):
        self.ann.createIndex(params, print_progress=print_progress)

    def saveIndex(self, fname):
        state = (
            self._i_sub,
            self._i_indexed_vecs,
            self._subspaces_to_basis_vectors,
            self._indexed_vectors_to_subspace,
            self._subspace_id_to_external_id,
            self._subspace_ranks
        )
        with open(fname + ".state", "wb") as outf:
            pickle.dump(state, outf)
        self.ann.saveIndex(fname)

    def loadIndex(self, fname, print_progress=False):
        with open(fname + ".state", "rb") as inf:
            loaded = pickle.load(inf)
            (
                self._i_sub, self._i_indexed_vecs, self._subspaces_to_basis_vectors, self._indexed_vectors_to_subspace,
                self._subspace_id_to_external_id, self._subspace_ranks
            ) = loaded

        self.ann.loadIndex(fname, print_progress=print_progress)

    def setQueryTimeParams(self, params):
        self.ann.setQueryTimeParams(params)

    def knnQuery(self, subspace, *args, **kwargs):
        return self.knnQueryBatch([subspace], *args, **kwargs)[0]

    def knnQueryBatch(
            self, query_spaces, subquery_k=10, k=10, num_threads=16, sim="exact", prefilter_final=None,
            signflip=True
    ):
        ranks = [len(basis_vecs) for (basis_vecs, dim_weights) in query_spaces]
        # if prefilter_final is not None:
            # print("ranks:", ranks)  # TODO remove
        # dim_ws = [dim_weights for (basis_vecs, dim_weights) in query_spaces]
        # print(ranks)
        # print(dim_ws)
        # print("first q shape:", query_spaces[0][0].shape)
        # print("second q shape:", query_spaces[1][0].shape)

        log.debug("preparing query data for ANN")

        # query_vecs = (subspace for subspace, dim_weights in query_spaces)

        if signflip:
            queryies_signflip = (np.vstack([space_vecs, -space_vecs]) for (space_vecs, dim_weights) in query_spaces)
            all_vec_queries = np.vstack(queryies_signflip)
        else:
            all_vec_queries = np.vstack((sp for (sp, dim_weights) in query_spaces))
        # print("final query vectors shape:", all_vec_queries.shape)
        log.debug("Querying ANN")
        allres = self.ann.knnQueryBatch(
            all_vec_queries, k=subquery_k, num_threads=num_threads
        )  # list of tuples [(ids, dists), (ids, dists), ...]

        log.debug("Processing ANN results, sim func is %s" % sim)
        all_result_ids, all_result_dists = zip(*allres)
        all_result_ids, all_result_dists = iter(all_result_ids), iter(all_result_dists)
        # assert len(all_result_ids) == all_vec_queries.shape[0] * self.vecs_per_space_dim
        # maxresults = min([len(reslist) for reslist in all_result_ids])
        # all_result_ids = [reslist[:maxresults] for reslist in all_result_ids]
        # all_result_dists = [reslist[:maxresults] for reslist in all_result_dists]

        res_sets = []
        # start_idx = 0
        for subspace, rank in zip(query_spaces, ranks):
            if signflip:
                offset = 2 * rank * self.vecs_per_space_dim  # because of flipped signs
            else:
                offset = rank * self.vecs_per_space_dim  # no flipped signs

            # end_idx = start_idx + offset
            # ids = np.array(all_result_ids)[start_idx: start_idx + rank]
            # dists = np.array(all_result_dists)[start_idx: start_idx + rank]

            # ids = all_result_ids[start_idx: start_idx + rank]
            # dists = all_result_dists[start_idx: start_idx + rank]
            ids = (next(all_result_ids) for _ in range(offset))
            dists = (next(all_result_dists) for _ in range(offset))

            data_and_sims = self.rank_nns(
                ids=ids, dists=dists, k=k, subspace=subspace, sim=sim, prefilter_final=prefilter_final
            )
            res_sets.append(data_and_sims)

            # start_idx = end_idx

        # just making sure we processed everything
        assert not any(True for _ in all_result_ids)
        assert not any(True for _ in all_result_dists)

        log.debug("ANSS query finished")
        return res_sets

    def exact_sim(self, subspace, cand_sub_id, partial_sims):
        return subspace_similarity(subspace, self._subspaces_to_basis_vectors[cand_sub_id])

    def standard_approx_sim(self, subspace, cand_sub_id, partial_sims):
        # We want to compute, for each subspace:
        # Sqrt(1 / rank * (Sum over i, j (p_i^T q_j^T) ^ 2))
        rank = len(subspace)
        minrank = min(rank, self._subspace_ranks[cand_sub_id])
        return np.sqrt((np.array(partial_sims) ** 2).sum() / minrank)

    def extrapolated_approx_sim(self, subspace, cand_sub_id, partial_sims):
        # We want to compute, for each subspace:
        # Sqrt(1 / rank * (Sum over i, j (p_i^T q_j^T) ^ 2))
        rank = len(subspace)
        minrank = min(rank, self._subspace_ranks[cand_sub_id])
        maxrank = max(rank, self._subspace_ranks[cand_sub_id])
        if maxrank - len(partial_sims) != 0:  # TODO a better interpolation might be possible
            frac_sum = 0
            if len(partial_sims) == 1:
                frac = 0.5
            else:
                prev = partial_sims[0]
                for f in partial_sims[1:]:
                    frac_sum += f / prev
                    prev = f
                frac = frac_sum / (len(partial_sims) - 1)

            for i in range(maxrank - len(partial_sims)):
                partial_sims.append(partial_sims[-1] * frac)

        return np.sqrt((np.array(partial_sims) ** 2).sum() / minrank)

    def hitcount_approx_sim(self, subspace, cand_sub_id, partial_sims):
        rank = float(len(subspace))
        return len(partial_sims) / (rank * self._subspace_ranks[cand_sub_id])

    sim_funcs = {
        "exact": exact_sim,
        "standard_approx": standard_approx_sim,
        "extrapolated_approx": extrapolated_approx_sim,
        "hitcount": hitcount_approx_sim
    }

    def rank_nns(self, ids, dists, k, subspace, sim, prefilter_final=None):
        subspace, dimension_weights = subspace

        partial_subspace_similarities = defaultdict(list)

        ids_flat = (_i for sublist in ids for _i in sublist)
        dists_flat = (d for sublist in dists for d in sublist)

        for _id, dist in zip(ids_flat, dists_flat):
            # cosine simil in nmslib is 1 - cos dist of normalized vectors
            # thus we can get the dot product (==cos similarity bc normalized vecs) directly via:
            # (p^T q) = 1. - dist

            cossim = 1. - dist
            subspace_id = self._indexed_vectors_to_subspace[_id]
            partial_subspace_similarities[subspace_id].append(cossim)

        if prefilter_final is not None:
            approx_estimates = list(sorted(
                [
                    (
                        sub_id,
                        partial_sims
                    ) for sub_id, partial_sims in partial_subspace_similarities.items()
                ],
                key=lambda item: -self.standard_approx_sim(subspace, item[0], item[1])
            )
            )[:max(k, int(prefilter_final * len(partial_subspace_similarities)))]

            candidates = dict(approx_estimates)
        else:
            candidates = partial_subspace_similarities

        # compute exact similarities for candidates
        final_estimates = list(sorted(
            [
                (
                    self._subspace_id_to_external_id[sub_id],
                    self.sim_funcs[sim](self, subspace, sub_id, partial_sims)
                ) for sub_id, partial_sims in candidates.items()
            ],
            key=lambda item: -item[1]
        ))

        data_and_sims = list(zip(*final_estimates[:k]))  # two lists, first data, then distances

        if not data_and_sims:
            data_and_sims = [(), ()]
        else:
            assert data_and_sims[1][0] >= data_and_sims[1][1]

        return data_and_sims
