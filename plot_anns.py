import numpy
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, LogFormatter

mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='append')
parser.add_argument('--output', action='append')

args = parser.parse_args()

# Construct palette by reading all inputs
all_algos = set()
for fn in args.input:
    for line in open(fn):
        all_algos.add(line.strip().split('\t')[0])

colors = plt.cm.Set1(numpy.linspace(0, 1, len(all_algos)))
linestyles = {}
for i, algo in enumerate(all_algos):
    linestyles[algo] = (colors[i], ['--', '-.', '-', ':'][i % 4], ['+', '<', 'o', 'D', '*', 'x', 's'][i % 7])

# Now generate each plot
for fn_in, fn_out in zip(args.input, args.output):
    all_data = {}

    for line in open(fn_in):
        algo, algo_name, build_time, search_time, precision = line.strip().split('\t')
        all_data.setdefault(algo, []).append((algo_name, float(build_time), float(search_time), float(precision)))

    dont_keep = []
    for algo in all_data:
        tokeep = input("%s ? (y/n)" % algo)
        if tokeep == "y":
            continue
        else:
            dont_keep.append(algo)

    for algo in dont_keep:
        del all_data[algo]

    handles = []
    labels = []

    plt.figure(figsize=(8, 8))
    for algo in sorted(all_data.keys(), key=str.lower):
        # for algo in all_data:
        data = all_data[algo]
        data.sort(key=lambda t: t[-2])  # sort by time
        ys = [1.0 / t[-2] for t in data]  # queries per second
        # ys = [t[-2] for t in data]  # seconds per queries
        xs = [t[-1] for t in data]
        ls = [t[0] for t in data]

        # Plot Pareto frontier
        # xs, ys = [], []
        # last_y = float('-inf')
        # for t in data:
        #     y = t[-1]
        #     if y > last_y:
        #         last_y = y
        #         xs.append(t[-1])
        #         ys.append(1.0 / t[-2])
        #         # ys.append(t[-2])
        color, linestyle, marker = linestyles[algo]
        handle, = plt.plot(
            xs, ys, '-', label=algo, color=color,
            ms=5,
            mew=1,
            lw=2,
            linestyle=linestyle, marker=marker
        )
        handles.append(handle)
        labels.append(algo)

    plt.gca().set_yscale('log')
    # plt.gca().set_yscale('linear')
    plt.gca().set_title('Precision-Performance tradeoff - up and to the right is better')
    plt.gca().set_ylabel('Queries per second ($s^{-1}$) - larger is better')
    plt.gca().set_xlabel('10-NN precision - larger is better')
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.xlim([0.0, 1.03])
    plt.ylim(ymin=0, ymax=620)
    start, end = plt.gca().get_ylim()
    # plt.gca().yaxis.set_ticks(numpy.arange(start, end, 10000))
    # plt.gca().yaxis.set_ticks(numpy.arange(0, end, 5))
    plt.gca().xaxis.set_ticks(numpy.arange(0, 1.1, .1))
    # plt.gca().yaxis.set_ticks(numpy.arange(int(start), int(end) + 1, 50))
    numpy.arange(int(20), int(end) + 1, 20)
    ticks1 = list(range(20, 100, 20))
    ticks2 = list(range(100, int(end), 100))
    plt.gca().yaxis.set_ticks([int(start), 10] + ticks1 + ticks2)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    plt.savefig(fn_out, bbox_inches='tight')
