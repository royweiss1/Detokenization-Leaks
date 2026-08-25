# Detokenization Leaks: Reconstructing Local LLM Outputs From Cache Traces

This repository contains the code used to generate the dataset, collect cache traces from a local LLM, and train/evaluate the models used in the paper.

The project is split into three main pieces:

1. dataset construction
2. cache-trace extraction
3. model training and evaluation experiments

The file `full_list_intel_smt_cpus_gen6_gen14.csv` contains the affected Intel SMT CPU list used to identify compatible desktop/laptop platforms for the side-channel setup.

## 1) Dataset generation

The text data pipeline lives under `datasets/prepare_data/`.

- `1_download_dataset.py`: downloads the base conversational data.
- `2_extract_sentences.py`: extracts assistant responses and splits them into first/middle segments.
- `3_clean_sentences.py`: filters low-quality text, removes refusals, deduplicates prompts, and creates train/test splits.
- `4_trace_mapping.py`: tokenizes the cleaned text and prepares parquet files aligned with the trace data.

This produces the cleaned UltraChat-style examples used for the reconstruction tasks.

## 2) Cache-trace extraction

The low-level side-channel tooling lives under `trace_extractor/`.

- `Makefile`: builds the attacker binary.
- `collect_attacker.c`: the attacker that performs the cache measurement loop.
- `collect_victim.py`: the victim process that detokenizes model outputs under a pinned CPU setup.
- `analyze_timings.py`: validates timing overlap between attacker and victim events.

The core idea is to run attacker and victim on hyper-threaded CPU and recover per-token cache behavior while the victim calls the local detokenizer.

For furthur details, looked at the in depth README.md in `trace_extractor/`.


## 3) Training and experiments

The experiments are under `training/`.

### `training/generality_experiment/`

These scripts evaluate whether the method generalizes across sentence positions and model settings.

- `prepare_symbol_dataset.py`: builds the symbol-level dataset.
- `train_t5_first_sentences.py`: trains the first-sentence model.
- `train_t5_middle_sentences.py`: trains the middle-sentence model.
- `eval_t5_first.py`: evaluates the first-sentence model.
- `eval_t5_middle.py`: evaluates the middle-sentence model.
- `judge_llm_eval.py`: LLM-based judging pipeline for reconstruction quality.

### `training/query_performance_tradeoff_experiment/`

This is the tradeoff study between query budget and performance.

- `1_build_profiles_csv.py`: builds the profile traces.
- `2_prepare_symbols.py`: prepares symbol data.
- `train.py`: trains the model under the chosen query budget.
- `eval.py`: evaluates the trained model.
- `find_equivalent_benign_q.py`: benchmarks equivalent benign queries.
- `orchestrate.py`: coordinates the experiment grid.

### `training/robustness_experiment/`

- `eval_noise_no_train.py`: measures robustness under noisy conditions without retraining.

## Typical workflow

A typical run goes in roughly this order:

```bash
# 1. Prepare data
cd datasets/prepare_data
python 1_download_dataset.py
python 2_extract_sentences.py
python 3_clean_sentences.py
python 4_trace_mapping.py --model phi

# 2. Build and run the cache-trace extractor
cd ../../trace_extractor
make
# start victim + attacker on HT sibling cores

# 3. Train/evaluate experiments
cd ../training
# run the relevant experiment scripts
```

Note that This repository is meant as a research codebase rather than a production package.
