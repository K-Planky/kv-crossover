"""Export Vidur model execution times (not end-to-end request latencies)."""

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 8192
CHUNK = 128
BATCH = 8
BUCKET = 64


def load_source(source, profile):
    def git(*args):
        return subprocess.check_output(['git', '-C', source, *args], text=True).strip()

    if git('rev-parse', 'HEAD') != profile['revision'] or git('status', '--porcelain'):
        sys.exit('expected a clean Vidur checkout at ' + profile['revision'])
    directory = source / 'data/profiling/compute/a100' / profile['model']
    for name, expected in profile['inputs'].items():
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            sys.exit(f'SHA-256 mismatch: {path}')
    sys.path.insert(0, str(source))
    return directory


def predictor(directory, model, scratch):
    from vidur.config import (MetricsConfig, RandomForrestExecutionTimePredictorConfig,
                              ReplicaConfig, SarathiSchedulerConfig)
    from vidur.execution_time_predictor.random_forrest_execution_time_predictor import (
        RandomForrestExecutionTimePredictor,
    )

    np.random.seed(42)  # sklearn's estimator uses the process RNG; train serially.
    config = RandomForrestExecutionTimePredictorConfig(
        compute_input_file=str(directory / 'mlp.csv'),
        attention_input_file=str(directory / 'attention.csv'),
        num_estimators=[32], max_depth=[16], min_samples_split=[2],
        num_training_job_threads=1, k_fold_cv_splits=2,
        kv_cache_prediction_granularity=BUCKET, prediction_max_batch_size=BATCH,
        prediction_max_tokens_per_request=CONTEXT, prediction_max_prefill_chunk_size=CHUNK,
        skip_cpu_overhead_modeling=True,
    )
    return RandomForrestExecutionTimePredictor(
        config, ReplicaConfig(model_name=model, device='a100'), SarathiSchedulerConfig(),
        MetricsConfig(cache_dir=scratch, output_dir=scratch),
    )


def rows(p):
    from vidur.entities import Batch, Request

    def cost(requests, tokens):
        seconds = p.get_execution_time(Batch(0, requests, tokens), 0).total_time
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f'invalid predicted seconds: {seconds}')
        return round(seconds * 1e9)

    for cached in range(0, CONTEXT + 1, BUCKET):
        for new in range(1, min(CHUNK, CONTEXT - cached) + 1):
            req = Request(0, cached + new, 1, cached)
            yield 'prefill', 1, cached, new, cost([req], [new])
        if cached:
            for size in range(1, BATCH + 1):
                # Use equal contexts as a representative of Vidur's mean bucket.
                # Even the last bucket is reachable (e.g. 8191 rounds to 8192).
                processed = min(cached, CONTEXT - 1)
                requests = [Request(0, processed - 1, 2) for _ in range(size)]
                for req in requests:
                    req.on_batch_end(0, processed - 1)  # includes first output token
                yield 'decode', size, cached, 1, cost(requests, [1] * size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vidur_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    profile = tomllib.loads((ROOT / 'experiments/paper.toml').read_text())['vidur_a100']
    directory = load_source(args.vidur_checkout.resolve(), profile)
    with tempfile.TemporaryDirectory() as scratch:
        p = predictor(directory, profile['model'], scratch)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
            candidate = Path(staging) / 'profile.csv'
            with candidate.open('w', newline='') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['phase', 'batch_size', 'cached_bucket_tokens',
                                 'new_tokens_per_request', 'predicted_ns'])
                writer.writerows(rows(p))
            os.replace(candidate, args.output)


if __name__ == '__main__':
    main()
