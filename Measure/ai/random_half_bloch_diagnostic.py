"""Small Bloch-matrix diagnostics for the fixed random_half_3 product state.

No entropies or large real-space matrices. A mesh is diagnostic evidence, not
a proof that there are no unsampled overlap zeros. Kappa is supercell momentum.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_random_half_bloch"
OUT.mkdir(exist_ok=True)
CATALOGUE = ROOT / "data/patterns.csv"
with CATALOGUE.open(encoding="utf-8-sig", newline="") as handle:
    pattern = next(row for row in csv.DictReader(handle) if row["pattern"] == "random_half_3")
m = int(pattern["m"])
occupied = np.array([bit == "1" for bit in pattern["occupied_bits"]])
P = np.eye(2*m*m)[:, occupied]
edges = []
for y in range(m):
    for x in range(m):
        for dx, dy in ((0,0),(1,0),(0,-1)):
            bx, by = (x+dx) % m, (y+dy) % m
            rx, ry = (x+dx) // m, (y+dy) // m
            edges.append((2*(x+m*y), 1+2*(bx+m*by), rx, ry))


def hamiltonian(kx, ky):
    h = np.zeros((2*m*m, 2*m*m), complex)
    for a, b, rx, ry in edges:
        value = np.exp(1j*(kx*rx+ky*ry))
        h[a,b] += value
        h[b,a] += value.conjugate()
    return h


def overlap(kx, ky):
    energy, vectors = np.linalg.eigh(hamiltonian(kx,ky))
    singular = np.linalg.svd(vectors[:,:m*m].conj().T @ P, compute_uv=False)
    return dict(kx=float(kx),ky=float(ky),radius=float(np.hypot(kx,ky)),
                min_sv=float(singular[-1]), second_min_sv=float(singular[-2]),
                log_overlap2=float(2*np.log(np.maximum(singular,1e-300)).sum()),
                single_particle_gap=float(energy[m*m]-energy[m*m-1]))


def save_csv(name, rows):
    with (OUT/name).open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


mesh_size = 80
mesh = -np.pi + (2*np.arange(mesh_size)+1)*np.pi/mesh_size
mesh_rows = [overlap(kx,ky) for ky in mesh for kx in mesh]
save_csv("overlap_mesh.csv",mesh_rows)
outside = sorted([row for row in mesh_rows if row["radius"] >= .2],key=lambda r:r["min_sv"])
refined = []
def objective(k):
    wrapped = (np.asarray(k)+np.pi) % (2*np.pi)-np.pi
    if np.linalg.norm(wrapped) < .2:
        return 10.+(.2-np.linalg.norm(wrapped))
    return np.log(max(overlap(*wrapped)["min_sv"],1e-16))
for row in outside[:8]:
    point = np.array([row["kx"],row["ky"]])
    step = 2*np.pi/mesh_size
    for iteration in range(2000):
        candidates = [point + step*np.array([dx,dy]) for dx in (-1,0,1) for dy in (-1,0,1)]
        values = [objective(candidate) for candidate in candidates]
        best = int(np.argmin(values))
        if values[best] < values[4]-1e-13:
            point = candidates[best]
        else:
            step *= .6
        if step < 1e-10:
            break
    point = (point+np.pi) % (2*np.pi)-np.pi
    refined.append(dict(**overlap(*point), refinement_step=float(step)))
save_csv("off_dirac_refinement.csv",refined)

energy, vectors = np.linalg.eigh(hamiltonian(0,0))
negative = vectors[:,energy < -1e-10]
zero = vectors[:,abs(energy) <= 1e-10]
_, singular_negative, vh = np.linalg.svd(negative.conj().T @ P,full_matrices=True)
right_null = vh[len(singular_negative):,:].conj().T
zero_overlap = zero.conj().T @ P @ right_null
initial_zero, _ = np.linalg.qr(zero_overlap)
dx_h, dy_h = np.zeros((18,18),complex), np.zeros((18,18),complex)
for a,b,rx,ry in edges:
    dx_h[a,b] += 1j*rx; dx_h[b,a] -= 1j*rx
    dy_h[a,b] += 1j*ry; dy_h[b,a] -= 1j*ry
dx_eff, dy_eff = zero.conj().T@dx_h@zero, zero.conj().T@dy_h@zero
def effective_overlap(phi):
    e, v = np.linalg.eigh(np.cos(phi)*dx_eff+np.sin(phi)*dy_eff)
    singular = np.linalg.svd(v[:,:2].conj().T@initial_zero,compute_uv=False)
    return float(singular[-1])

angles = np.arange(720)*2*np.pi/720
angle_rows = [dict(phi=float(phi),min_sv=effective_overlap(phi)) for phi in angles]
save_csv("effective_dirac_angles.csv",angle_rows)
angular_minima=[]
for j,row in enumerate(angle_rows):
    if row["min_sv"] <= angle_rows[j-1]["min_sv"] and row["min_sv"] <= angle_rows[(j+1)%720]["min_sv"]:
        left,right = row["phi"]-2*np.pi/720,row["phi"]+2*np.pi/720
        ratio = (np.sqrt(5)-1)/2
        a,b = right-ratio*(right-left),left+ratio*(right-left)
        fa,fb = effective_overlap(a),effective_overlap(b)
        for iteration in range(64):
            if fa < fb:
                right,b,fb = b,a,fa
                a = right-ratio*(right-left)
                fa = effective_overlap(a)
            else:
                left,a,fa = a,b,fb
                b = left+ratio*(right-left)
                fb = effective_overlap(b)
        phi = (a+b)/2
        angular_minima.append(dict(phi=float(phi%(2*np.pi)),min_sv=effective_overlap(phi)))
radial_rows=[]
for node_index, node in enumerate(angular_minima):
    for radius in (1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6):
        row=overlap(radius*np.cos(node["phi"]),radius*np.sin(node["phi"]))
        radial_rows.append(dict(node=node_index,phi=node["phi"],**row))
if radial_rows:
    save_csv("dirac_radial_overlaps.csv",radial_rows)
summary=dict(pattern="random_half_3",m=m,occupied_bits=pattern["occupied_bits"],
             convention="supercell kappa; physical Dirac valleys both fold to kappa=(0,0)",
             ground_negative_count_at_gamma=negative.shape[1],zero_count_at_gamma=zero.shape[1],
             minimum_negative_support_sv=float(min(singular_negative)),
             zero_residual_singular_values=np.linalg.svd(zero_overlap,compute_uv=False).tolist(),
             mesh_size=mesh_size,mesh_minimum=min(mesh_rows,key=lambda r:r["min_sv"]),
             off_dirac_mesh_minimum=outside[0],off_dirac_refinements=refined,
             effective_dirac_minima=angular_minima,
             interpretation="Finite mesh/refinement and double-precision diagnostics only; not a proof of no overlap zeros or of entropy scaling.",
             hashes={"script":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "catalogue":hashlib.sha256(CATALOGUE.read_bytes()).hexdigest()})
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
