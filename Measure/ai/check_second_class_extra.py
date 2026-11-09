"""Geometry-only selection and independent Bloch rank audit of new class-2 states.

No entropy, correlation-region matrix, or fitted coefficient is read.  The
Julia pattern catalogue is parsed as the source of the four selected patterns;
the Bloch Hamiltonian and spectral-support energy are reconstructed in NumPy.
Only data/second_class_extra_* outputs are written.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"
CATALOGUE=BASE/"second_class_extra_patterns.jl"
RANK_TOL=1e-10
ZERO_TOL=1e-9
SIZES={4:list(range(24,145,12))+[180],2:list(range(18,91,6))+[108,126,144,180]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def read_catalogue():
    text=CATALOGUE.read_text(encoding="utf-8")
    entries=re.findall(r'\("(pure02_[a-z_0-9]+)",\s*([24]),\s*"([01]+)",\s*(true|false)\)',text)
    assert len(entries)==4
    descriptions={
        "pure02_staircase4":"Doubly occupied row lengths (4,3,1,0), each row filled from x=1.",
        "pure02_notched_stripe4":"Start with columns x=1,2 doubly occupied; move the doubly occupied cell (1,1) to the empty cell (3,2).",
        "pure02_alternating_x":"Doubly occupy odd x and empty even x; invariant along y.",
        "pure02_alternating_checker":"Doubly occupy cells with x+y even and empty cells with x+y odd.",
    }
    patterns=[]
    for name,m,bits,expected in entries:
        m=int(m)
        assert len(bits)==2*m*m and bits.count("1")==m*m
        assert all(bits[2*j]==bits[2*j+1] for j in range(m*m))
        patterns.append(dict(name=name,m=m,bits=bits,occupied_bits=bits,class_id=2,
            n0=m*m//2,n1=0,n2=m*m//2,expected_ground=expected=="true",geometry=descriptions[name]))
    return patterns


def hamiltonian(m,kx,ky):
    h=np.zeros((2*m*m,2*m*m),complex)
    for y in range(m):
        for x in range(m):
            a=2*(x+m*y)
            for dx,dy in ((0,0),(1,0),(0,-1)):
                xx,yy=x+dx,y+dy
                b=1+2*(xx%m+m*(yy%m))
                z=np.exp(1j*(kx*(xx//m)+ky*(yy//m)))
                h[a,b]+=z;h[b,a]+=z.conjugate()
    return h


def spectral_rank(coefficients):
    return int(np.count_nonzero(np.linalg.svd(coefficients,compute_uv=False)>RANK_TOL))


def greedy_energy(energies,coefficients):
    first,previous,result=0,0,0.
    while first<len(energies):
        last=first+1
        while last<len(energies) and abs(energies[last]-energies[first])<1e-10:
            last+=1
        level=float(np.mean(energies[first:last]))
        current=spectral_rank(coefficients[:last])
        assert previous<=current<=coefficients.shape[1]
        result+=(current-previous)*level
        if current==coefficients.shape[1]:return result
        previous=current;first=last
    raise AssertionError("initial occupied rank lost")


def audit_same_size(patterns,L):
    m=patterns[0]["m"];assert all(p["m"]==m for p in patterns)
    assert L%m==0
    q=L//m;pnum=m*m
    output={p["name"]:dict(pattern=p["name"],m=m,L=L,occupied_bits=p["bits"],
        expected_ground=p["expected_ground"],blocks=q*q,blocked_momenta=0,
        negative_rank_deficiency=0,nonpositive_rank_deficiency=0,
        total_negative_rank_deficiency=0,total_nonpositive_rank_deficiency=0,
        min_negative_sv=1.,min_nonpositive_sv=1.,ground_energy=0.,accessible_energy=0.) for p in patterns}
    fs={p["name"]:np.flatnonzero(np.array(list(p["bits"]))=="1") for p in patterns}
    for iy in range(q):
        for ix in range(q):
            energies,vectors=np.linalg.eigh(hamiltonian(m,2*math.pi*ix/q,2*math.pi*iy/q))
            negative,nonpositive=energies < -ZERO_TOL,energies <= ZERO_TOL
            ground=float(np.sum(energies[:pnum]))
            for p in patterns:
                row=output[p["name"]];c=vectors[fs[p["name"]]].conj().T
                sn=np.linalg.svd(c[negative],compute_uv=False)
                sl=np.linalg.svd(c[nonpositive],compute_uv=False)
                dn=int(np.sum(negative))-int(np.count_nonzero(sn>RANK_TOL))
                dl=pnum-int(np.count_nonzero(sl>RANK_TOL))
                row["blocked_momenta"]+=int(dn>0 or dl>0)
                row["negative_rank_deficiency"]=max(row["negative_rank_deficiency"],dn)
                row["nonpositive_rank_deficiency"]=max(row["nonpositive_rank_deficiency"],dl)
                row["total_negative_rank_deficiency"]+=dn
                row["total_nonpositive_rank_deficiency"]+=dl
                row["min_negative_sv"]=min(row["min_negative_sv"],float(sn[-1]))
                row["min_nonpositive_sv"]=min(row["min_nonpositive_sv"],float(sl[-1]))
                row["ground_energy"]+=ground
                # This separate cumulative spectral-rank calculation explicitly
                # checks the minimal accessible energy instead of merely setting
                # it to the ground energy when the two final ranks pass.
                row["accessible_energy"]+=greedy_energy(energies,c)
    rows=list(output.values())
    for row in rows:
        row["excess_energy"]=row["accessible_energy"]-row["ground_energy"]
        row["ground_reachable"]=(row["blocked_momenta"]==0 and abs(row["excess_energy"])<1e-8)
        row["rank_tol"]=RANK_TOL;row["zero_tol"]=ZERO_TOL
    return rows


def induced_degrees(bits,m,value):
    full=lambda x,y:bits[2*((x%m)+m*(y%m))]==value
    counts={k:0 for k in range(4)}
    for y in range(m):
        for x in range(m):
            if full(x,y):
                counts[1+full(x+1,y)+full(x,y-1)]+=1
                counts[1+full(x-1,y)+full(x,y+1)]+=1
    return [counts[k] for k in range(4)]


def inequivalence(patterns):
    old={"pure02_block4":"11111111110011001100110000000000",
         "pure02_irregular4":"11001111001100000011001111001100"}
    allm4=old|{p["name"]:p["bits"] for p in patterns if p["m"]==4}
    witnesses={name:dict(occupied_degrees_0_1_2_3=induced_degrees(bits,4,"1"),
        empty_degrees_0_1_2_3=induced_degrees(bits,4,"0")) for name,bits in allm4.items()}
    signatures=[tuple(sorted((tuple(w["occupied_degrees_0_1_2_3"]),tuple(w["empty_degrees_0_1_2_3"]))))
                for w in witnesses.values()]
    assert len(set(signatures))==4
    return dict(status="PASS",witnesses=witnesses,
        reason="Any translation or space-group operation of the HoneyComb120 hopping graph preserves the degree histogram of the occupied induced nearest-neighbour subgraph. Particle-hole complementation exchanges occupied and empty histograms. The unordered pairs of histograms differ for all four m4 patterns, proving that neither new example is a spatial-symmetry/complement copy of either old example or of the other new example. This graph invariant is stronger than checking a selected list of coordinate transformations.")


def main():
    patterns=read_catalogue();DATA.mkdir(exist_ok=True)
    # Family 1: first fixed staircase shape, row lengths (4,3,1,0).
    # Family 2: start with the width-two vertical strip; exchange (1,1)
    # with (3,1), then (3,2). The first candidate fails and the next passes.
    staircase=next(p for p in patterns if p["name"]=="pure02_staircase4")
    notch=next(p for p in patterns if p["name"]=="pure02_notched_stripe4")
    candidate0=dict(name="notched_stripe_candidate_1",m=4,
        bits="00111100111100001111000011110000",expected_ground=False)
    selection=[]
    for p,order,family in ((staircase,1,"staircase"),(candidate0,1,"notched_stripe"),(notch,2,"notched_stripe")):
        for L in ((12,) if p is candidate0 else (12,24,36)):
            row=audit_same_size([p],L)[0]
            row["candidate_family"]=family;row["candidate_order"]=order
            row["selected"]=p is not candidate0
            selection.append(row)
            assert row["ground_reachable"]==p["expected_ground"]
    witnesses=inequivalence(patterns)
    rows=[]
    for m in (4,2):
        group=[p for p in patterns if p["m"]==m]
        for L in SIZES[m]:
            block=audit_same_size(group,L)
            for row in block:
                assert row["ground_reachable"]==row["expected_ground"],row
            rows.extend(block)
            print("m",m,"L",L,[(r["pattern"],r["blocked_momenta"],r["excess_energy"],r["min_negative_sv"]) for r in block],flush=True)
    out=DATA/"second_class_extra_reachability.csv"
    candidates=DATA/"second_class_extra_candidate_screen.csv"
    write_csv(out,rows);write_csv(candidates,selection)
    receipt=dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),
        patterns=patterns,sizes_by_m={str(m):s for m,s in SIZES.items()},rows=len(rows),
        rank_tol=RANK_TOL,zero_tol=ZERO_TOL,numpy_version=np.__version__,
        protocol="Independent NumPy Bloch Hamiltonian; test rank(V_negative^dagger P)=dim(H_negative) and rank(V_nonpositive^dagger P)=m^2 in every actual finite-size momentum block. Independently obtain accessible energy from cumulative spectral-rank increments. No entropy data or fits are read.",
        selection="Two separate geometry-defined candidate families, with first rank-accessible candidate in each: staircase rows(4,3,1,0); width-two stripe with one cell exchange, destinations(3,1) then(3,2). Selected candidates were checked at L12,24,36 before any entropy run; full requested sizes are audited here.",
        inequivalence=witnesses,
        interpretation="The two m4 examples are ground-accessible at all listed checked sizes, not an unproved all-L theorem. Both alternating m2 controls are strictly blocked and must be evolved with exact spectral support; their accessible infinite-time states are excited states, not equilibrium ground states.",
        files={path.relative_to(BASE).as_posix():digest(path) for path in (CATALOGUE,Path(__file__),out,candidates)})
    (DATA/"second_class_extra_reachability.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print("PASS",len(rows),"pattern-size checks",flush=True)


if __name__=="__main__":main()
