"""Compare two shipped rows with fresh calls to the pinned predictor."""

import argparse
import csv
from pathlib import Path
import tempfile
import tomllib

from export_a100 import ROOT, load_source, predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vidur_checkout', type=Path)
    args = parser.parse_args()
    config = ROOT / 'experiments/paper.toml'
    profile = tomllib.loads(config.read_text())['vidur_a100']
    directory = load_source(args.vidur_checkout.resolve(), profile)
    from vidur.entities import Batch, Request

    with (config.parent / profile['path']).open() as f:
        table = {(r['phase'], int(r['batch_size']), int(r['cached_bucket_tokens']),
                  int(r['new_tokens_per_request'])): int(r['predicted_ns'])
                 for r in csv.DictReader(f)}
    with tempfile.TemporaryDirectory() as scratch:
        p = predictor(directory, profile['model'], scratch)
        prompt = Request(0, 6472, 8, num_processed_tokens=6144)
        a, b = Request(0, 511, 8), Request(0, 1535, 8)
        a.on_batch_end(0, 511)
        b.on_batch_end(0, 1535)
        for key, batch in [
            (('prefill', 1, 6144, 128), Batch(0, [prompt], [128])),
            (('decode', 2, 1024, 1), Batch(0, [a, b], [1, 1])),
        ]:
            actual_ns = p.get_execution_time(batch, 0).model_time_ms * 1e6
            if abs(table[key] - actual_ns) > 0.5:
                raise ValueError(f'{key}: table={table[key]} vs predictor={actual_ns} ns')
            print(f'{key}: {table[key]} vs {actual_ns:.3f} ns')


if __name__ == '__main__':
    main()
