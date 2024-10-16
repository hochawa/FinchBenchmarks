import json
import math
from collections import defaultdict

def geometric_mean(arr):
    n = len(arr)
    return math.exp(sum(math.log(x) for x in arr) / n)

def calculate_speedups(data, operation):
    times_by_matrix = defaultdict(dict)

    for entry in data:
        if entry['operation'] == operation:
            matrix = entry['matrix']
            method = entry['method']
            time = entry['time']
            times_by_matrix[matrix][method] = time

    speedup_by_method_pair = defaultdict(list)

    for matrix, times in times_by_matrix.items():
        methods = list(times.keys())

        for i, baseline_method in enumerate(methods):
            for j, comparison_method in enumerate(methods):
                if i != j:
                    baseline_time = times[baseline_method]
                    comparison_time = times[comparison_method]
                    if comparison_time > 0:
                        speedup = baseline_time / comparison_time
                        speedup_by_method_pair[(baseline_method, comparison_method)].append(speedup)

    geo_speedups = {}
    for (baseline_method, comparison_method), speedups in speedup_by_method_pair.items():
        geo_speedups[(baseline_method, comparison_method)] = geometric_mean(speedups)

    return geo_speedups

def main():
    filename = 'graphs_results.json'
    
    with open(filename, 'r') as file:
        data = json.load(file)

    bfs_speedups = calculate_speedups(data, 'bfs')
    bellmanford_speedups = calculate_speedups(data, 'bellmanford')

    print("Geometric Mean Speedups for BFS:")
    if bfs_speedups:
        for (baseline_method, comparison_method), geo_speedup in bfs_speedups.items():
            print(f"{baseline_method} vs {comparison_method}: {geo_speedup:.4f}")
    else:
        print("No valid BFS data for speedup calculation.")

    print("\nGeometric Mean Speedups for Bellman-Ford:")
    if bellmanford_speedups:
        for (baseline_method, comparison_method), geo_speedup in bellmanford_speedups.items():
            print(f"{baseline_method} vs {comparison_method}: {geo_speedup:.4f}")
    else:
        print("No valid Bellman-Ford data for speedup calculation.")

if __name__ == "__main__":
    main()

