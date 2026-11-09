"""Construct and certify a rational real-Slater corner counterexample.

The final inequality is checked with Fraction arithmetic and a rational
upper bound on log(1+x), not floating-point entropy eigenvalues.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parent

def inv(A):
    n=len(A); W=[list(A[i])+[F(int(i==j)) for j in range(n)] for i in range(n)]
    for k in range(n):
        p=next(i for i in range(k,n) if W[i][k]);W[k],W[p]=W[p],W[k]
        z=W[k][k];W[k]=[x/z for x in W[k]]
        for i in range(n):
            if i!=k:
                z=W[i][k];W[i]=[x-z*y for x,y in zip(W[i],W[k])]
    return [r[n:] for r in W]

def mul(A,B): return [[sum((x*y for x,y in zip(r,c)),F(0)) for c in zip(*B)] for r in A]
def tr(A):return sum((A[i][i] for i in range(len(A))),F(0))
def det(A):
    W=[list(r) for r in A]; result=F(1);n=len(W)
    for k in range(n):
        p=next(i for i in range(k,n) if W[i][k]);W[k],W[p]=W[p],W[k]
        if p!=k: result=-result
        z=W[k][k];result*=z
        for i in range(k+1,n):
            a=W[i][k]/z
            for j in range(k+1,n):W[i][j]-=a*W[k][j]
    return result

def reg(w,h,x,y):return [2*((i+x)%2+2*((j+y)%2))+s for j in range(h) for i in range(w) for s in (0,1)]
REGS=[(reg(1,2,x,y),reg(2,1,x,y),reg(1,1,x,y)) for y in range(2) for x in range(2)]

def numeric(X):
    Q=np.linalg.qr(X)[0];G=Q@Q.T;D=np.zeros(2)
    for triple in REGS:
        vals=[]
        for r in triple:
            ev=np.linalg.eigvalsh(G[np.ix_(r,r)])
            vals.append(np.array([-np.log(ev**2+(1-ev)**2).sum(),(ev*(1-ev)).sum()]))
        D+=(.5*vals[0]+.5*vals[1]-vals[2])/4
    return D

def main():
    rng=np.random.default_rng(2026102917)
    for trial in range(10000):
        X=rng.normal(size=(8,4))*np.exp(rng.uniform(-3,3,size=(8,1)))
        integers=np.round(X*1000).astype(int)
        D=numeric(integers)
        if D[0]-2*D[1]<-1e-4 and 0<D[0]<.08:break
    else:raise RuntimeError('No suitable example')
    X=[[F(int(x)) for x in r] for r in integers]
    Xt=list(map(list,zip(*X)))
    G=mul(mul(X,inv(mul(Xt,X))),Xt)
    assert mul(G,G)==G
    ratio=F(1); DV=F(0); details=[]
    for triple in REGS:
        pur=[];var=[]
        for r in triple:
            C=[[G[i][j] for j in r] for i in r];C2=mul(C,C)
            purity=det([[F(int(i==j))-2*C[i][j]+2*C2[i][j] for j in range(len(r))] for i in range(len(r))])
            pur.append(purity);var.append(tr(C)-tr(C2))
        ratio*=pur[2]**2/(pur[0]*pur[1])
        DV+=(var[0]+var[1])/8-var[2]/4
    # D_S2 = log(ratio)/8.  For 0<x<1 the three-term alternating
    # Taylor sum is an exact upper bound on log(1+x).
    x=ratio-1
    assert 0<x<1
    log_upper=x-x*x/2+x*x*x/3
    margin=16*DV-log_upper
    assert margin>0
    payload=dict(status='EXACT_RATIONAL_CERTIFICATE_PASS',L=2,p=.5,trial=trial,
        integer_orbitals=integers.tolist(),projector_definition='G=X(X^T X)^(-1)X^T',
        numerical_D_S2=math.log(float(ratio))/8,numerical_DV=float(DV),
        numerical_gap=math.log(float(ratio))/8-2*float(DV),
        proof='0<x=ratio-1<1 and log(ratio)<=x-x^2/2+x^3/3<16*DV, all comparisons rational',
        ratio=str(ratio),DV=str(DV),rational_positive_margin=str(margin),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Disproves D_S2>=2 D_V for arbitrary pure Gaussian states after full translations; not a counterexample to a Born ensemble-specific statement')
    (ROOT/'exact_corner_counterexample.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('ratio','DV','rational_positive_margin')},indent=2))

if __name__=='__main__':main()
