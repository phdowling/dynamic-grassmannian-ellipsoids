from collections import defaultdict

from subspaces.nearest_subspace.nmslanns import BetterANSS
import numpy as np
import logging
log = logging.getLogger(__name__)


def sample_vectors(subspace, dim_weights, nsamples):
    nvecs, dims = subspace.shape
    mean = np.zeros((nvecs,))
    multivar_cov = dim_weights * np.eye(nvecs)
    samples = np.random.multivariate_normal(mean, multivar_cov, size=nsamples)
    samples = samples / np.linalg.norm(samples, axis=1)[:, np.newaxis]
    samples_in_original_space = np.dot(samples, subspace)
    return samples_in_original_space


def get_random_basis(subspace):
    subspace, dim_weights = subspace  # TODO maybe we can project the weights
    random_span = sample_vectors(subspace, dim_weights, subspace.shape[0])
    q = np.linalg.qr(random_span.T)[0]
    return (q.T, 1)


class RandomizedANSS(BetterANSS):
    def __init__(self, random_bases):
        super().__init__()
        self.num_rand_bases = random_bases

    def knnQueryBatch(
            self, query_spaces, subquery_k=10, k=10, num_threads=16, sim="exact", prefilter_final=None, signflip=True
    ):
        res_sets = []
        channeled_query_spaces = [query_spaces]
        log.debug("generating query channels")
        for i_channel in range(self.num_rand_bases):
            channeled_query_spaces.append([get_random_basis(subspace) for subspace in query_spaces])
        log.debug("Running channeled queries..")
        reduced_data = [defaultdict(float) for _ in range(len(query_spaces))]
        for i_channel, channel_queries in enumerate(channeled_query_spaces):
            log.debug("Running channel %s.." % i_channel)
            data_and_sims_lists = super(RandomizedANSS, self).knnQueryBatch(
                channel_queries, subquery_k=subquery_k, k=None, num_threads=num_threads,
                sim=sim, prefilter_final=prefilter_final, signflip=signflip
            )
            if self.num_rand_bases == 0:
                return data_and_sims_lists  # don't waste time "aggregating" results

            log.debug("Collecting results")
            # We keep the maximum approximate similarity encountered for every rotation
            for subspace_i, data_and_sims in enumerate(data_and_sims_lists):
                for data, sim_score in zip(*data_and_sims):
                    update_dict = reduced_data[subspace_i]
                    update_dict[data] = max(update_dict[data], sim_score)

        log.debug("Reducing results..")
        for reduced_ in reduced_data:
            sorted_sims = list(sorted(reduced_.items(), key=lambda kv: -kv[1]))[:k]
            rdata, rsims = zip(*sorted_sims)
            res_sets.append((np.array(rdata), np.array(rsims)))
        log.debug("done.")
        return res_sets
