"""Check every recorded suppressed-set snapshot against frozen chain metadata.

This does not re-estimate entropy statistics. It validates raw set integrity,
ordering, sampler-count consistency, receipts, and source/model/file hashes.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pilot', 'production'], required=True)
    args = ap.parse_args()
    directory = ROOT / 'data/born_scaling/boundary_check' / args.stage
    planpath = directory / 'plan.toml'
    manifestpath = directory / 'manifest.json'
    output = ROOT / 'analysis/born_scaling/boundary_check' / f'{args.stage}_snapshot_integrity.json'
    if output.exists():
        raise RuntimeError('Refuse to overwrite snapshot integrity receipt')
    plan = tomllib.loads(planpath.read_text(encoding='utf-8'))
    manifest = json.loads(manifestpath.read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    assert manifest['stage'] == plan['stage'] == args.stage
    assert manifest['plan_sha256'] == sha(planpath)
    assert manifest['jobs'] == plan['jobs']
    checked = {}

    def verify(path, digest):
        key = str(path.resolve())
        if key not in checked:
            checked[key] = sha(path)
        assert checked[key] == digest, path

    for source in (plan['source_sha256'], manifest['files']):
        for name, digest in source.items():
            verify(ROOT / name, digest)
    chains = []
    for job in plan['jobs']:
        receipt_path = ROOT / job['receipt_path']
        receipt = tomllib.loads(receipt_path.read_text(encoding='utf-8'))
        assert receipt['status'] == 'complete'
        assert receipt['job'] == job
        assert receipt['plan_sha256'] == sha(planpath)
        assert receipt['source_sha256'] == plan['source_sha256']
        for name, digest in receipt['files'].items():
            verify(ROOT / name, digest)
        modelpath = ROOT / job['model_manifest']
        verify(modelpath, job['model_manifest_sha256'])
        model = tomllib.loads(modelpath.read_text(encoding='utf-8'))
        for name, digest in model['files'].items():
            verify(modelpath.parent / name, digest)
        L = job['L']; n = L * L; N = 2 * n
        count = job['sweeps'] // job['measure_every']
        sets = np.fromfile(ROOT / job['snapshot_path'], dtype='<i4')
        assert sets.size == count * n
        sets = sets.reshape(count, n)
        assert np.all((sets >= 1) & (sets <= N))
        ordered = np.sort(sets, axis=1)
        assert np.all(np.diff(ordered, axis=1) > 0)
        initial = np.asarray(receipt['initial_E'], dtype=np.int32)
        assert initial.shape == (n,)
        assert np.array_equal(sets[-1], receipt['final_E'])
        assert receipt['records'] == count
        rows = list(csv.DictReader((ROOT / job['sample_path']).open(encoding='utf-8', newline='')))
        assert len(rows) == 13 * count
        accepted_total = 0; proposed_total = 0; max_changed = 0
        for ri in range(count):
            group = rows[13 * ri:13 * (ri + 1)]
            reference = group[0]
            for field in ('L', 'beta', 'mu', 'chain', 'seed', 'start', 'sweep', 'record_index',
                          'measure_every', 'accepted', 'proposed', 'log_weight', 'stage'):
                assert len({row[field] for row in group}) == 1
            assert int(reference['L']) == L
            assert int(reference['seed']) == job['seed']
            assert int(reference['chain']) == job['chain']
            assert int(reference['record_index']) == ri + 1
            assert int(reference['sweep']) == (ri + 1) * job['measure_every']
            assert int(reference['measure_every']) == job['measure_every']
            assert reference['start'] == job['start']
            assert reference['stage'] == job['stage']
            assert float(reference['beta']) == job['beta']
            assert float(reference['mu']) == job['mu']
            accepted = int(reference['accepted']); proposed = int(reference['proposed'])
            assert proposed == N * job['measure_every']
            assert 0 <= accepted <= proposed
            previous = initial if ri == 0 else sets[ri - 1]
            changed = int(np.count_nonzero(sets[ri] != previous))
            assert changed <= accepted, 'More E slots changed than accepted swaps'
            max_changed = max(max_changed, changed)
            accepted_total += accepted; proposed_total += proposed
        assert accepted_total == receipt['accepted']
        assert proposed_total == receipt['proposed']
        assert abs(accepted_total / proposed_total - receipt['acceptance']) < 1e-15
        chains.append(dict(run_id=job['run_id'], L=L, chain=job['chain'], snapshots=count,
                           retained_snapshots=job['retained_records'], entropy_rows=len(rows),
                           largest_changed_E_slots_per_record=max_changed,
                           accepted=accepted_total, proposed=proposed_total))
    result = dict(status='PASS', created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  stage=args.stage, plan_sha256=sha(planpath), manifest_sha256=sha(manifestpath),
                  script_sha256=sha(__file__), checked_files=len(checked),
                  checked_file_sha256=checked, chains=chains,
                  total_snapshots=sum(row['snapshots'] for row in chains),
                  scope='Every E set has valid, distinct sites; binary counts/order/final E match receipts; repeated sampler metadata is consistent across 13 rows; changed E slots do not exceed accepted swaps; all source/model/data hashes verified.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    snapshot = output.parent / 'source_snapshots' / f'{sha(__file__)}_{Path(__file__).name}'
    snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ('status', 'stage', 'checked_files', 'total_snapshots')}, indent=2))


if __name__ == '__main__':
    main()
