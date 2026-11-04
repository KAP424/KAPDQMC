"""Independent small finite-system tests of the accompanying theory review.

No production samples or historical results are modified. The exact sums
enumerate every output; the last section searches, rather than assumes, a
possible entropy-corner inequality.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import math
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(2026102901)


def s2v(G, R):
    ev = np.linalg.eigvalsh(G[np.ix_(R, R)])
    return -float(np.log(ev * ev + (1 - ev) ** 2).sum()), float((ev * (1-ev)).sum())


def normalizer(n, h, k, q):
    return math.fsum(math.comb(n, j)*math.comb(h, k-j)*q**j
                     for j in range(max(0, k-h), min(n,k)+1))


def rect(L, w, h, tx=0, ty=0):
    return np.array([2*((x+tx)%L + L*((y+ty)%L))+s
                     for y in range(h) for x in range(w) for s in (0,1)])


def regions(L, den):
    w = L//den
    return [(rect(L,w,L,tx,ty), rect(L,L,w,tx,ty), rect(L,w,L-w,tx,ty))
            for ty in range(L) for tx in range(L)]


def corner(G, regs, p):
    sv = np.array([[s2v(G,R) for R in triple] for triple in regs])
    return ((1-p)*sv[:,0,:]+p*sv[:,1,:]-sv[:,2,:]).mean(axis=0)


def exact_output_check(Q, k, beta, R):
    M,n = Q.shape; h=M-n; r=len(R)
    G0=Q@Q.conj().T
    t=math.exp(-2*beta); q=t*t
    z=normalizer(n,h,k,q)
    numerator=normalizer(n-1,h-1,k-1,q) if 0<k<M else 0.
    lam=1-(1-t)**2*numerator/z
    reference=lam*G0+(1-lam)*np.diag(np.diag(G0))
    mean_G=np.zeros_like(G0); mean_n2=mean_s=mean_v=mean_n=mean_nsq=0.
    sum_p=0.; EK=EK2=EXK=0.; mse=0.
    Fsets=list(combinations(range(M),n))
    pf=np.array([abs(np.linalg.det(Q[list(F),:]))**2 for F in Fsets])
    x=np.array([len(set(F)&set(R)) for F in Fsets],dtype=float)
    m=float(pf@x); var=float(pf@((x-m)**2))
    jprob=np.array([math.comb(n,j)*math.comb(h,k-j)*q**j/z
                   if 0<=k-j<=h else 0. for j in range(n+1)])
    jbar=float(jprob@np.arange(n+1)); varj=float(jprob@((np.arange(n+1)-jbar)**2))
    d=jbar/n-(k-jbar)/h
    cu=float(jprob@(np.arange(n+1)*(n-np.arange(n+1))))/(n*n*(n-1))
    vj=k-np.arange(n+1)
    cv=float(jprob@(vj*(h-vj)))/(h*h*(h-1))
    vk=cu*(m*(n-m)-var)+cv*((r-m)*(h-r+m)-var)+d*d*var+varj*((1/n+1/h)**2*var+((1/n+1/h)*m-r/h)**2)
    for E in combinations(range(M),k):
        filtered=Q.copy(); filtered[list(E),:]*=t
        P, Rqr=np.linalg.qr(filtered)
        weight=abs(np.linalg.det(Rqr))**2
        prob=weight/z
        G=P@P.conj().T
        ss,v=s2v(G,R); nm=float(np.trace(G[np.ix_(R,R)]).real)
        sum_p+=prob; mean_G+=prob*G; mean_s+=prob*ss; mean_v+=prob*v
        mean_n+=prob*nm; mean_nsq+=prob*nm*nm; mean_n2+=prob*(v+nm*nm)
        ke=len(set(E)&set(R))
        # Independently evaluate the full latent joint for this E.
        pj=pf*np.array([q**len(set(E)&set(F)) for F in Fsets])/z
        assert abs(float(pj.sum())-prob)<1e-11
        EK+=float(pj.sum())*ke; EK2+=float(pj.sum())*ke*ke
        EXK+=float(pj@x)*ke; mse+=float(pj@((x-(r-ke))**2))
    empirical_vk=EK2-EK*EK
    mseformula=var+vk+2*d*var+(m-r+EK)**2
    best=var-d*d*var*var/vk if vk>1e-14 else var
    errors={"normalization":abs(sum_p-1), "mean_G_channel":float(np.max(abs(mean_G-reference))),
            "mean_n":abs(mean_n-m),"mean_n2":abs(mean_n2-(var+m*m)),
            "total_variance":abs(mean_v-(var-(mean_nsq-m*m))),
            "hypergeom_variance":abs(empirical_vk-vk),
            "hypergeom_covariance":abs(EXK-m*EK-d*var),
            "observable_count_estimator_mse":abs(mse-mseformula)}
    assert max(errors.values())<1e-10,errors
    assert 2*mean_v-1e-10<=mean_s<=4*math.log(2)*mean_v+1e-10
    assert lam*lam*var-1e-10<=mean_v<=best+1e-10
    return dict(M=M,n=n,k=k,beta=beta,lambda1=lam,initial_variance=var,
                mean_entropy=mean_s,mean_quantum_variance=mean_v,linear_MMSE_bound=best,
                raw_complement_count_MSE=mse,errors=errors)


def check_geometry(L, den):
    p=1/den; w=L//den; M=2*L*L
    actual=np.zeros((M,M))
    for triple in regions(L,den):
        for factor,R in zip((1-p,p,-1),triple):
            inside=np.zeros(M);inside[R]=1
            actual+=factor*inside[:,None]*(1-inside)[None,:]/(L*L)
    expected=np.zeros_like(actual)
    for i in range(M):
        xi=(i//2)%L; yi=(i//2)//L
        for j in range(M):
            dx=((j//2)%L-xi)%L;dy=((j//2)//L-yi)%L
            expected[i,j]=min(dx,L-dx,w)*min(dy,L-dy,w)/(L*L)
    err=float(np.max(abs(actual-expected)))
    assert err<1e-12
    X=RNG.normal(size=(M,M//2))+1j*RNG.normal(size=(M,M//2))
    Q=np.linalg.qr(X)[0];G=Q@Q.conj().T
    entropy,dv=corner(G,regions(L,den),p)
    contraction=float((abs(G)**2*expected).sum())
    assert abs(dv-contraction)<1e-11
    subgroup=np.zeros((M,M))
    for ty in range(0,L,w):
        for tx in range(0,L,w):
            triple=(rect(L,w,L,tx,ty),rect(L,L,w,tx,ty),rect(L,w,L-w,tx,ty))
            for factor,R in zip((1-p,p,-1),triple):
                inside=np.zeros(M);inside[R]=1
                subgroup+=factor*inside[:,None]*(1-inside)[None,:]/den**2
    x=(np.arange(M)//2)%L; y=(np.arange(M)//2)//L
    subgroup_expected=((x[:,None]//w!=x[None,:]//w)&(y[:,None]//w!=y[None,:]//w))/den**2
    subgroup_error=float(np.max(abs(subgroup-subgroup_expected)))
    assert subgroup_error<1e-12
    return dict(L=L,p=p,ordered_site_pairs=M*M,sublattice_pairs=4,
                weight_max_error=err,min_pair_weight=float(actual.min()),
                subgroup_weight_max_error=subgroup_error,
                direct_D_variance=dv,kernel_D_variance=contraction,
                variance_contraction_error=abs(dv-contraction))


def check_filter_spectrum():
    rng=np.random.default_rng(2026102902)
    worst_shift_excess=0.; worst_entropy_bound_excess=0.
    for trial in range(300):
        M=12;n=6;r=6
        Q=np.linalg.qr(rng.normal(size=(M,n))+1j*rng.normal(size=(M,n)))[0]
        beta=(trial%5)*.5;t=math.exp(-2*beta)
        D=np.exp(rng.uniform(math.log(t),0.,size=M))
        P=np.linalg.qr(D[:,None]*Q)[0]
        ev0=np.linalg.eigvalsh(Q[:r,:]@Q[:r,:].conj().T)
        ev1=np.linalg.eigvalsh(P[:r,:]@P[:r,:].conj().T)
        eps0=np.log((1-ev0)/ev0);eps1=np.log((1-ev1)/ev1)
        excess=float(np.max(abs(eps1-eps0))-4*beta)
        s0=-np.log(ev0**2+(1-ev0)**2).sum()
        s1=-np.log(ev1**2+(1-ev1)**2).sum()
        sexcess=max(t*t*s0-s1,s1-s0/(t*t))
        worst_shift_excess=max(worst_shift_excess,excess)
        worst_entropy_bound_excess=max(worst_entropy_bound_excess,float(sexcess))
    assert worst_shift_excess<1e-8
    assert worst_entropy_bound_excess<1e-10
    return dict(trials=300,maximum_log_odds_shift_bound_excess=worst_shift_excess,
                maximum_entropy_bound_excess=worst_entropy_bound_excess)


def search_corner_inequality():
    best=None;counter=None;trials=0
    for L,den,count in ((2,2,4000),(3,3,1000),(4,2,300)):
        M=2*L*L; regs=regions(L,den)
        for trial in range(count):
            X=RNG.normal(size=(M,M//2))+1j*RNG.normal(size=(M,M//2))
            # Include near-product states with highly inhomogeneous filtering.
            if trial%2:
                X*=np.exp(RNG.normal(0,3,size=M))[:,None]
            Q=np.linalg.qr(X)[0]; G=Q@Q.conj().T
            ds,dv=corner(G,regs,1/den);gap=ds-2*dv;trials+=1
            item=dict(L=L,p=1/den,trial=trial,D_S2=ds,D_variance=dv,D_S2_minus_2_D_variance=gap)
            if best is None or gap<best["D_S2_minus_2_D_variance"]:best=item
            if gap < -1e-7:
                out=ROOT/"corner_inequality_counterexample.npz"
                np.savez(out,Q=Q,L=L,den=den)
                counter=dict(**item,saved_projector=out.name)
                return dict(trials=trials,minimum=best,counterexample=counter)
    return dict(trials=trials,minimum=best,counterexample=counter,
                caveat="No found counterexample is not a proof.")


def main():
    checks=[]
    for M,n in ((8,4),(7,3)):
        Q=np.linalg.qr(RNG.normal(size=(M,n))+1j*RNG.normal(size=(M,n)))[0]
        for k in (2,M//2,M-2):
            for beta in (0.,.3,1.,3.):
                checks.append(exact_output_check(Q,k,beta,list(range(3))))
    geom=[check_geometry(L,den) for L,den in ((2,2),(3,3),(4,2),(6,2),(6,3))]
    payload=dict(status="PASS",exact_output_checks=checks,geometry_checks=geom,
                 entropy_corner_search=search_corner_inequality(),
                 individual_filter_spectrum_check=check_filter_spectrum(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/"theory_checks.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","exact_checks":len(checks),"geometry_checks":geom,
                      "entropy_corner_search":payload["entropy_corner_search"]},indent=2))


if __name__=="__main__":main()
