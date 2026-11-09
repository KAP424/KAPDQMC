"""Run beta points without log files; optionally extend an existing flat dataset."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import threading
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRINT_LOCK = threading.Lock()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--betas', nargs='+', type=float, default=[i / 5 for i in range(11)])
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    data = HERE / 'benchmark_beta_data_10000'
    if len(set(args.betas)) != len(args.betas) or any(not np.isfinite(b) or b < 0 for b in args.betas):
        raise ValueError('Provide distinct finite nonnegative beta values.')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.append:
        metadata = (data / 'metadata.txt').read_text(encoding='utf-8')
        settings = dict(line.split('=', 1) for line in metadata.splitlines() if '=' in line)
        assert int(settings['sweeps']) == 10000 and int(settings['warmup']) == 1000
        assert int(settings['Nlambda']) == 3 and float(settings['Theta_relax']) == 5.0
        previous = ast.literal_eval(settings['betas'])
        for line in metadata.splitlines():
            if ' sha256=' in line:
                relative, expected = line.split(' sha256=')
                assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
        existing = {float(re.search(r'M([0-9.]+)\.csv$', f.name).group(1)) for f in data.glob('*SCEE*.csv')}
        assert existing == set(previous)
        if existing.intersection(args.betas):
            raise ValueError('Requested beta already exists; refusing duplicate samples.')
    else:
        if data.exists():
            raise FileExistsError('Use --append with new beta points to extend the existing dataset.')
        previous = []
        metadata = None

    # Stream stdout to the caller; never create .log files.
    with tempfile.TemporaryDirectory(prefix='.benchmark_beta_pending_', dir=HERE) as temporary:
        staging = Path(temporary).resolve()
        staging.relative_to(HERE.resolve())  # Verify containment before automatic cleanup.

        def run_beta(beta):
            folder = staging / f'beta_{beta:g}'
            env = dict(os.environ, BENCH_SWEEPS='10000', BENCH_WARMUP='1000', BENCH_BETAS=str(beta))
            recent = deque(maxlen=20)
            process = subprocess.Popen(
                ['julia', '--project=.', str(HERE / 'benchmark_beta.jl'), str(folder)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                recent.append(line)
                with PRINT_LOCK:
                    print(f'[beta={beta:g}] {line.rstrip()}', flush=True)
            if process.wait() != 0:
                raise RuntimeError(f'beta={beta} failed: ' + ''.join(recent))
            files = list(folder.glob('*SCEE*.csv'))
            assert len(files) == 1
            with files[0].open(encoding='utf-8', newline='') as handle:
                values = np.asarray(list(csv.reader(handle)), dtype=float)
            assert values.shape == (3, 10001)
            assert np.allclose(values[:, 0], np.arange(3) / 3)
            assert np.isfinite(values).all() and (values[:, 1:] > 0).all()
            return files[0]

        files = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_beta, beta) for beta in args.betas]):
                files.append(future.result())
        if metadata is None:
            metadata = (files[0].parent / 'metadata.txt').read_text(encoding='utf-8')
            data.mkdir(exist_ok=False)
        for source in files:
            source.resolve().relative_to(staging)
            target = data / source.name
            target.resolve().relative_to(data.resolve())
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
        betas = sorted(previous + args.betas)
        metadata = re.sub(r'^betas=.*$', f'betas={betas}', metadata, flags=re.MULTILINE)
        metadata = '\n'.join(line for line in metadata.splitlines()
                             if not line.startswith('Six independent Julia processes;'))
        (data / 'metadata.txt').write_text(metadata.rstrip() + '\n', encoding='utf-8')
    print(f'Completed {len(args.betas)} new beta points; {len(betas)} points in dataset. No log files written.', flush=True)
    subprocess.run([sys.executable, str(HERE / 'build_beta_notebook.py')], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
