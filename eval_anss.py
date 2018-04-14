import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='append')

args = parser.parse_args()

for fn_in in args.input:
    all_data = {}

    for line in open(fn_in):
        algo, algo_name, build_time, search_time, precision = line.strip().split('\t')
        all_data.setdefault(algo, []).append((algo_name, float(build_time), float(search_time), float(precision)))

    print("algo build_time auc best_precision best_prec_qps")
    for algo in all_data:
        algo_names, build_times, search_times, precisions = zip(*all_data[algo])

        search_times = np.array(search_times)
        queries_per_second = 1. / search_times
        precisions = np.array(precisions)
        auc = np.trapz(y=queries_per_second, x=precisions)

        best_precision_idx = np.argmax(precisions)
        best_precision, best_precision_qps = precisions[best_precision_idx], queries_per_second[best_precision_idx]
        print("%s %s %s %s %s" % (
            algo,
            np.round(build_times[0], 2),
            np.round(auc, 2),
            np.round(best_precision, 2),
            np.round(best_precision_qps, 2))
        )
