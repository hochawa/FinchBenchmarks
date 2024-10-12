import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from collections import defaultdict
import re

vuduc_mtxs = [
    "Boeing/ct20stif",
    "Simon/olafu",
    "Boeing/bcsstk35",
    "Boeing/crystk02",
    "Boeing/crystk03",
    "Nasa/nasasrb",
    "Simon/raefsky4",
    "Mulvey/finan512",
    "Cote/vibrobox",
    "HB/saylr4",
    "Rothberg/3dtube",
    "Pothen/pwt",
    "Gupta/gupta1"
    "Simon/raefsky3",
    "Simon/venkat01",
    "FIDAP/ex11",
    "Zitney/rdist1",
    "HB/orani678",
    "Goodwin/rim",
    "Hamm/memplus",
    "HB/gemat11",
    "Mallya/lhr10",
    "Grund/bayer02",
    "Grund/bayer10",
    "Brethour/coater2",
    "ATandT/onetone2",
    "Wang/wang4",
    "HB/lnsp3937",
    "HB/sherman5",
    "HB/sherman3",
    "Shyy/shyy161",
    "Wang/wang3",
]
good_mtxs = [
    #willow
    "GHS_indef/exdata_1",
    "Janna/Emilia_923",
    "Janna/Geo_1438",
    "TAMU_SmartGridCenter/ACTIVSg70K"
    "Goodwin/Goodwin_071", 
    "Norris/heart3",
    "Rajat/rajat26", 
    "TSOPF/TSOPF_RS_b678_c1" 
    #permutation
    "permutation_synthetic"
    #graph
    "SNAP/email-Enron", 
    "SNAP/as-735",
    "SNAP/Oregon-1",
    "Newman/as-22july06",
    "SNAP/loc-Brightkite",
    "SNAP/as-Skitter"
    "SNAP/soc-Epinions1",
    "SNAP/wiki-Vote",
    "SNAP/email-EuAll",
    "SNAP/cit-HepPh",
    "SNAP/web-NotreDame",
    "SNAP/amazon0302",
    "SNAP/p2p-Gnutella08",
    "SNAP/email-Eu-core",
    #banded
    "toeplitz_small_band",
    "toeplitz_medium_band",
    "toeplitz_large_band",
    #triangle
    "upper_triangle",
    #taco
    "HB/bcsstk17",
    "Williams/pdb1HYS",
    "Williams/cant",
    "Williams/consph",
    "Williams/cop20k_A",
    "DNVS/shipsec1",
    "Boeing/pwtk",
    "Bova/rma10"
    #blocked
    "blocked_10x10",
]

RESULTS_FILE_PATH = "spmv_results_lanka.json"
CHARTS_DIRECTORY = "charts/"
FORMAT_ORDER = {
    "finch_sym": -1,
    "finch_unsym": -2,
    "finch_unsym_row_maj": -3,
    "finch_vbl": -4,
    "finch_vbl_unsym": -5,
    "finch_vbl_unsym_row_maj": -6,
    "finch_band": -7,
    "finch_band_unsym": -8,
    "finch_band_unsym_row_maj": -9,
    "finch_pattern": -10,
    "finch_pattern_unsym": -11,
    "finch_pattern_unsym_row_maj": -12,
    "finch_point": -13,
    "finch_point_row_maj": -14,
    "finch_point_pattern": -15,
    "finch_point_pattern_row_maj": -16,
    "finch_blocked": -17,
}
FORMAT_LABELS = {
    "finch_sym": "Symmetric SparseList",
    "finch_unsym": "SparseList",
    "finch_unsym_row_maj": "SparseList (Row-Major)",
    "finch_vbl": "Symmetric SparseVBL",
    "finch_vbl_unsym": "SparseVBL",
    "finch_vbl_unsym_row_maj": "SparseVBL (Row-Major)",
    "finch_band": "Symmetric SparseBand",
    "finch_band_unsym": "SparseBand",
    "finch_band_unsym_row_maj": "SparseBand (Row-Major)",
    "finch_pattern": "Symmetric Pattern",
    "finch_pattern_unsym": "Pattern",
    "finch_pattern_unsym_row_maj": "Pattern (Row-Major)",
    "finch_point": "SparsePoint",
    "finch_point_row_maj": "SparsePoint (Row-Major)",
    "finch_point_pattern": "SparsePoint Pattern",
    "finch_point_pattern_row_maj": "SparsePoint Pattern (Row-Major)",
    "finch_blocked": "4D-Blocked"
}

def all_formats_chart(ordered_by_format=False):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    data = defaultdict(lambda: defaultdict(int))
    finch_formats = get_best_finch_format()

    for result in results:
        mtx = result["matrix"]
        method = result["method"]
        time = result["time"]

        if (method == "finch_unsym") or (method == "finch_unsym_row_maj"):
            # finch_baseline_time = data[mtx]["finch_baseline"]
            # data[mtx]["finch_baseline"] = time if finch_baseline_time == 0 else min(time, finch_baseline_time)
            #infinity in python is float('inf')
            #data[mtx]["finch_baseline"] = min(data[mtx].get("finch_baseline", float('inf')), data[mtx][method])
            #the above line doesn't work, but this one does:
            data[mtx]["finch_baseline"] = min(data[mtx].get("finch_baseline", float('inf')), time)
        if "finch" in method and finch_formats[mtx] != method:
            continue
        method = "finch" if "finch" in method else method
        data[mtx][method] = time

    for mtx, times in data.items():
        ref_time = times["taco"]
        for method, time in times.items():
            times[method] = ref_time / time

    if ordered_by_format:
        #ordered_data = sorted(data.items(), key = lambda mtx_results: (mtx_results[1]["finch"] > 1, FORMAT_ORDER[finch_formats[mtx_results[0]]], mtx_results[1]["finch"]), reverse=True)
        ordered_data = sorted(data.items(), key = lambda mtx_results: (FORMAT_ORDER[finch_formats[mtx_results[0]]], mtx_results[1]["finch"]), reverse=True)
    else:
        ordered_data = sorted(data.items(), key = lambda mtx_results: mtx_results[1]["finch"], reverse=True)

    methods = ["finch", "finch_baseline", "julia_stdlib", "mkl", "eigen", "cora"]#, "suite_sparse"]
    legend_labels = ["Finch (Best)", "Finch (Baseline)", "Julia Stdlib", "MKL", "Eigen", "CoRa"]#, "SuiteSparse"]
    colors = {
        "finch": "tab:green",
        "finch_baseline": "tab:gray",
        "julia_stdlib": "tab:blue",
        "mkl": "tab:orange",
        "eigen": "tab:red",
        "cora": "tab:purple"
    }
    all_data = defaultdict(list)
    for i, (mtx, times) in enumerate(ordered_data):
        for method in methods:
            all_data[method].append(times.get(method, 0))  # Use None or 0 as default

    ordered_mtxs = [mtx for mtx, _ in ordered_data]
    labels = [FORMAT_LABELS[finch_formats[mtx]] for mtx, _ in ordered_data]
    short_mtxs = [mtx.rsplit('/',1)[-1] for mtx in ordered_mtxs]
    new_mtxs = {
        "toeplitz_large_band": "large_band",
        "toeplitz_medium_band": "medium_band",
        "toeplitz_small_band": "small_band",
        #"TSOPF_RS_b678_c1": "*RS_b678_c1",
    }
    short_mtxs = [new_mtxs.get(mtx, mtx) for mtx in short_mtxs]

    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, labeled_groups=["finch"], bar_labels_dict={"finch": labels[:]}, title="SpMV Performance (Speedup Over Taco) labeled")
    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, title="SpMV Performance (Speedup Over Taco)")

    # for mtx in mtxs:
        # all_formats_for_matrix_chart(mtx)


def get_best_finch_format():
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    formats = defaultdict(list)
    for result in results:
        if "finch" not in result["method"]:
            continue
        formats[result["matrix"]].append((result["method"], result["time"]))

    best_formats = defaultdict(list)
    for matrix, format_times in formats.items():
        best_format, _ = min(format_times, key=lambda x: x[1])
        best_formats[matrix] = best_format
    
    return best_formats


def get_method_results(method, mtxs=[]):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    mtx_times = {}
    for result in results:
        if result["method"] == method and (mtxs == [] or result["matrix"] in mtxs):
            mtx_times[result["matrix"]] = result["time"]
    return mtx_times


def get_speedups(faster_results, slower_results):
    speedups = {}
    for mtx, slow_time in slower_results.items():
        if mtx in faster_results:
            speedups[mtx] = slow_time / faster_results[mtx]
    return speedups


def order_speedups(speedups):
    ordered = [(mtx, time) for mtx, time in speedups.items()]
    return sorted(ordered, key=lambda x: x[1], reverse=True)


def method_to_ref_comparison_chart(method, ref, title=""):
    method_results = get_method_results(method)
    ref_results = get_method_results("taco")
    speedups = get_speedups(method_results, ref_results)

    x_axis = []
    data = defaultdict(list)
    for matrix, speedup in speedups.items():
        x_axis.append(matrix)
        data[method].append(speedup)
        data[ref].append(1)

    make_grouped_bar_chart([method, ref], x_axis, data, labeled_groups=[method], title=title)


def all_formats_for_matrix_chart(matrix):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    data = {}
    for result in results:
        if result["matrix"] == matrix:
            data[result["method"]] = result["time"]
    
    formats = []
    speedups = []
    bar_colors = []
    for format, time in data.items():
        formats.append(format)
        speedups.append(data["taco"] / time)
        bar_colors.append("orange" if "finch" in format else "green")
    
    fig, ax = plt.subplots()
    ax.bar(formats, speedups, width = 0.2, color = bar_colors)
    ax.set_ylabel("Speedup")
    ax.set_title(matrix + " Performance")
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)

    fig_file = matrix.lower().replace("/", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + "/matrices/" + fig_file, dpi=200, bbox_inches="tight")
    plt.close() 


def make_grouped_bar_chart(labels, x_axis, data, colors = None, labeled_groups = [], title = "", y_label = "", bar_labels_dict={}, legend_labels=None, reference_label = ""):
    x = np.arange(len(data[labels[0]]))
    width = 0.22 
    multiplier = 0
    max_height = 0

    fig, ax = plt.subplots(figsize=(12, 4))
    for label in labels:
        label_data = data[label]
        max_height = max(max_height, max(label_data))
        offset = width * multiplier
        if colors:
            rects = ax.bar(x + offset, label_data, width, label=label, color=colors[label])
        else:
            rects = ax.bar(x + offset, label_data, width, label=label)
        bar_labels = bar_labels_dict[label] if (label in bar_labels_dict) else [round(float(val), 2) if label in labeled_groups else "" for val in label_data]
        ax.bar_label(rects, padding=0, labels=bar_labels, fontsize=5, rotation=90)
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1)/2, x_axis)
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)
    if legend_labels:
        ax.legend(legend_labels, loc='upper left', ncols=3, fontsize='small')
    else:
        ax.legend(loc='upper left', ncols=3, fontsize='small')
    ax.set_ylim(0, max_height + 0.5)

        # Adjusting x-axis limits to make bars go to the edges
    ax.set_xlim(-0.5, len(x_axis) - 0.5 + width * len(labels))

    plt.plot([-1, len(x_axis)], [1, 1], linestyle='--', color="tab:red", linewidth=0.75, label=reference_label)

    fig_file = title.lower().replace(" ", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200, bbox_inches="tight")
    plt.close()
    

all_formats_chart()
all_formats_chart(ordered_by_format=True)
# method_to_ref_comparison_chart("finch", "taco", title="Finch SparseList Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_unsym", "taco", title="Finch SparseList SpMV Performance")
# method_to_ref_comparison_chart("finch_unsym_row_maj", "taco", title="Finch SparseList Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_vbl", "taco", title="Finch SparseVBL Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_vbl_unsym", "taco", title="Finch SparseVBL SpMV Performance")
# method_to_ref_comparison_chart("finch_vbl_unsym_row_maj", "taco", title="Finch SparseVBL Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_band", "taco", title="Finch SparseBand Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_band_unsym", "taco", title="Finch SparseBand SpMV Performance")
# method_to_ref_comparison_chart("finch_band_unsym_row_maj", "taco", title="Finch SparseBand Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_pattern", "taco", title="Finch SparseList Pattern Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_pattern_unsym", "taco", title="Finch SparseList Pattern SpMV Performance")
# method_to_ref_comparison_chart("finch_pattern_unsym_row_maj", "taco", title="Finch SparseList Pattern Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_point", "taco", title="Finch SparsePoint SpMV Performance")
# method_to_ref_comparison_chart("finch_point_row_maj", "taco", title="Finch SparsePoint Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_point_pattern", "taco", title="Finch SparsePoint Pattern SpMV Performance")
# method_to_ref_comparison_chart("finch_point_pattern_row_maj", "taco", title="Finch SparsePoint Pattern Row Major SpMV Performance")