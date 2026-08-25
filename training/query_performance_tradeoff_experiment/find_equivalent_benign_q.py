#!/usr/bin/env python3
"""
find_equivalent_q.py

For each Q in the original "repeat this sequence 5 times" profiling method,
find the smallest number of ordinary (benign) UltraChat-style queries that
produce the EXACT same profiling dataset -- same target tokens, same trace
rows, not just similar coverage.

WHY THIS WORKS
--------------
Every trace assignment in the pipeline draws from the same per-token pool
(shuffled once, from a fixed seed) and always pops from the END of the list:
    pool[token].pop()
So the row given to the i-th time a token is used depends only on i, not on
which query produced it or in what order. Once a token has been used
target_depth times by ANY mix of benign queries, it has exactly the rows the
original "repeat" method would have given it -- no coordination needed
beyond "stop once a token reaches depth 5."

HOW
---
1. Rank the domain vocabulary by frequency (same ranking "uniform" uses).
2. For a given Q, the original method's target is its top Q*10 tokens by
   frequency, each needing 5 occurrences (see plan_uniform in
   1_build_profiles_csv_synthetic.py).
3. Select benign UltraChat queries with the pipeline's existing greedy
   selector (the one that actually builds the real profiling dataset).
4. Walk the selected queries one at a time, counting how many times each
   target token has appeared (capped at 5). The number of queries needed
   before every target token has reached 5 is the answer -- and it's the
   true minimum, since that count can only grow as more queries are added.

Usage:
    python3 find_equivalent_q.py --machine-type desktop
    python3 find_equivalent_q.py --machine-type desktop \
        --uniform-q-list 25 50 100 250 500 1000 --max-benign-q 50000
"""

import argparse
import importlib.util
from collections import Counter
from pathlib import Path


def load_module(filename):
    """Import a sibling pipeline script as a module (its filename starts
    with a digit, so a plain `import` statement won't work)."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_equivalent_q(uniform_q, tokens_by_freq, benign_queries, tokens_per_query=10, target_depth=5):
    """
    Smallest prefix length of benign_queries that gives every target token
    of uniform@uniform_q at least target_depth occurrences.
    Returns None if benign_queries runs out before that happens.
    """
    target_tokens = set(tok for tok, _freq in tokens_by_freq[: uniform_q * tokens_per_query])

    occurrences = Counter()
    tokens_remaining = len(target_tokens)

    for num_queries, query in enumerate(benign_queries, start=1):
        for token in query["token_ids"]:
            if token not in target_tokens or occurrences[token] == target_depth:
                continue
            occurrences[token] += 1
            if occurrences[token] == target_depth:
                tokens_remaining -= 1
        if tokens_remaining == 0:
            return num_queries

    return None  # ran out of benign queries before every target token was covered


def main():
    parser = argparse.ArgumentParser(
        description="Find the benign-query budget equivalent to each original uniform@Q."
    )
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--process-type", default="singleprocess")
    parser.add_argument("--uniform-q-list", nargs="+", type=int,
                         default=[25, 100, 250, 500, 1000])
    parser.add_argument("--max-benign-q", type=int, default=100_000,
                         help="How many benign queries to select and search through.")
    args = parser.parse_args()

    synthetic = load_module("1_build_profiles_csv_synthetic.py")
    benign = load_module("1_build_profiles_csv_benign.py")

    trace_csv = Path(
        f"/<PATH_TO_DATASET>/llamacpp_phi/"
        f"dataset_top50_per_token_{args.machine_type}_{args.process_type}.csv"
    )

    print("Loading traces and building token pools...")
    df_traces = synthetic.read_trace_csv(trace_csv)
    pool, _heldout = synthetic.build_trace_pools(df_traces, synthetic.HELDOUT_PER_TOKEN, synthetic.SEED)

    print("Ranking domain vocabulary by frequency...")
    domain_freq = synthetic.load_domain_token_frequencies(synthetic.FIRST_PARQUET)
    valid_tokens = set(pool.keys()) & set(domain_freq.keys())
    tokens_by_freq = sorted(((t, domain_freq[t]) for t in valid_tokens), key=lambda pair: -pair[1])

    print("Loading benign UltraChat candidates and selecting queries...")
    candidates = benign.load_benign_response_candidates(
        benign.FIRST_PARQUET, valid_token_set=valid_tokens,
        max_candidates=benign.MAX_CANDIDATES, seed=benign.SEED,
    )
    benign_queries, _selected_counts = benign.select_benign_queries(candidates, max_q=args.max_benign_q)

    print(f"\n{'original Q':>12}  {'benign Q needed':>16}  {'multiplier':>11}")
    for uniform_q in args.uniform_q_list:
        equivalent_q = find_equivalent_q(uniform_q, tokens_by_freq, benign_queries)
        if equivalent_q is None:
            print(f"{uniform_q:>12}  {'not reached -- raise --max-benign-q':>16}")
        else:
            print(f"{uniform_q:>12}  {equivalent_q:>16,}  {equivalent_q / uniform_q:>10.1f}x")


if __name__ == "__main__":
    main()
