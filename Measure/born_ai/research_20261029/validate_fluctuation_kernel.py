"""Independent explicit-translation check of the proposed number-variance kernels."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectangle(x, y, L, width, height, tx, ty):
    return ((x - tx) % L < width) & ((y - ty) % L < height)


def variance(C, mask):
    sub = C[np.ix_(mask, mask)]
    return np.trace(sub) - np.square(sub).sum()


def main():
    rows = []
    sources = {Path(__file__).name: digest(Path(__file__))}
    rng = np.random.default_rng(291029315)
    for L in (6, 12):
        a, N, w = L * L, 2 * L * L, L // 3
        path = next((HERE / "pilot/models").glob(f"*_L{L}_*_Q.bin"))
        Q = np.fromfile(path, dtype=np.float64).reshape((N, a), order="F")
        sources[path.relative_to(HERE).as_posix()] = digest(path)
        x = np.repeat(np.arange(a) % L, 2)
        y = np.repeat(np.arange(a) // L, 2)
        dx = (x[:, None] - x[None, :]) % L
        dy = (y[:, None] - y[None, :]) % L
        kernel = np.minimum(np.minimum(dx, L - dx), w) * np.minimum(np.minimum(dy, L - dy), w) / L**2
        subgroup_kernel = ((x[:, None] // w != x[None, :] // w) & (y[:, None] // w != y[None, :] // w)) / 9
        for beta in (0.0, 3.0):
            E = rng.choice(N, a, replace=False)
            filtered = Q.copy()
            filtered[E] *= math.exp(-2 * beta)
            R = np.linalg.qr(filtered, mode="reduced")[0]
            C = R @ R.T
            squared = np.square(C)
            direct_full, direct_subgroup = [], []
            for tx in range(L):
                for ty in range(L):
                    A = rectangle(x, y, L, w, L, tx, ty)
                    B = rectangle(x, y, L, w, 2 * w, tx, ty)
                    D = rectangle(x, y, L, L, w, tx, ty)
                    value = (2 * variance(C, A) + variance(C, D)) / 3 - variance(C, B)
                    direct_full.append(value)
                    if tx % w == 0 and ty % w == 0:
                        direct_subgroup.append(value)
            full = float(np.sum(squared * kernel))
            subgroup = float(np.sum(squared * subgroup_kernel))
            error_full = abs(full - np.mean(direct_full))
            error_subgroup = abs(subgroup - np.mean(direct_subgroup))
            assert error_full < 1e-9 and error_subgroup < 1e-9
            assert full >= 0 and subgroup >= 0
            rows.append(dict(L=L, beta=beta, full=full, subgroup=subgroup,
                             full_error=float(error_full), subgroup_error=float(error_subgroup)))
    output = HERE / "validation/fluctuation_kernel.json"
    if output.exists():
        raise FileExistsError("Preserve existing validation")
    output.write_text(json.dumps(dict(status="PASS", rows=rows, sources_sha256=sources), indent=2) + "\n")
    print(json.dumps(dict(status="PASS", maximum_error=max(max(r["full_error"], r["subgroup_error"]) for r in rows))))


if __name__ == "__main__":
    main()
