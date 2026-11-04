"""Deterministic FFT number-fluctuation corner reference, not Born samples.

The microscopic hopping and zero-mode selector match OriginalSwapScaling.
Large-L calculations use O(L^2 log L) work and O(L^2) memory; no entropy
eigenvalues or extrapolated Monte Carlo observations are generated here.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OUT = ROOT / 'baseline'


def kernels(L, mu=1e-8):
    k = 2*np.pi*np.arange(L)/L
    f = 1+np.exp(1j*k)[None,:]+np.exp(-1j*k)[:,None]
    null = np.abs(f)<1e-12
    f[null] = 0
    eps = np.hypot(np.abs(f), mu)
    if mu == 0:
        eps[null] = 1
    aa = .5*(1-mu/eps)
    bb = .5*(1+mu/eps)
    aa[null] = 0.; bb[null] = 1.
    ab = -.5*f/eps
    return np.array([[np.fft.ifft2(aa), np.fft.ifft2(ab)],
                     [np.fft.ifft2(ab.conj()), np.fft.ifft2(bb)]])


def corners(K, den):
    L=K.shape[-1]; w=L//den
    d=np.arange(L); h=np.minimum(np.minimum(d,L-d),w)/L
    sq=(np.abs(K)**2).sum(axis=(0,1))
    dv=float(L*L*np.sum(sq*h[:,None]*h[None,:]))
    va=float(L*L*np.sum(sq*h[None,:]))
    vc=float(L*L*np.sum(sq*h[:,None]))
    return dv,va,vc


def attenuation(a,beta):
    t=math.exp(-2*beta)
    j=np.arange(a+1,dtype=float)
    logp=np.zeros(a+1)
    logp[1:]=np.cumsum(2*(np.log(a-j[1:]+1)-np.log(j[1:]))-4*beta)
    logp-=np.max(logp); p=np.exp(logp); p/=np.sum(p)
    # Positive expression, avoiding subtraction of two almost equal numbers.
    lam=float(p@(2*j*(a-j)+j*j/t+(a-j)**2*t)/(a*a))
    return lam,float(p@j)


def dense_validation():
    rows=[]
    for L in (6,12,18):
        path=BASE/f'data/born_scaling/pilot/models/state_L{L}_staggered_mu1.0e-8_G.bin'
        G=np.fromfile(path,dtype=np.float64).reshape((2*L*L,2*L*L),order='F')
        K=kernels(L)
        cells=np.arange(L*L); xx=cells%L; yy=cells//L
        dx=(xx[:,None]-xx[None,:])%L; dy=(yy[:,None]-yy[None,:])%L
        reconstructed=np.empty(G.shape,dtype=complex)
        for s in (0,1):
            for u in (0,1):
                reconstructed[s::2,u::2]=K[s,u,dy,dx]
        err=float(np.max(np.abs(G-reconstructed)))
        assert err<1e-12,(L,err)
        for den in (3,2):
            w=L//den; p=1/den
            regs=[]
            for wx,hy in ((w,L),(L,w),(w,L-w)):
                regs.append(np.array([2*(x+L*y)+s for y in range(hy) for x in range(wx) for s in (0,1)]))
            v=[np.trace(G[np.ix_(r,r)])-np.sum(np.abs(G[np.ix_(r,r)])**2) for r in regs]
            direct=(1-p)*v[0]+p*v[1]-v[2]
            fft=corners(K,den)[0]
            assert abs(direct-fft)<2e-11,(L,den,direct,fft)
            rows.append(dict(L=L,p=p,projector_error=err,corner_error=abs(direct-fft),DV=fft,
                reference=str(path.relative_to(BASE)),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    validation=dense_validation()
    sizes=[6,12,18,24,36,48,72,96,144,192,288,384,576,768,1152,1536]
    rows=[]; atten=[]
    for L in sizes:
        tick=time.monotonic()
        for mu in (1e-8,0.):
            K=kernels(L,mu)
            for den in (3,2):
                dv,va,vc=corners(K,den)
                rows.append(dict(L=L,mu=mu,p=1/den,DV=dv,VA=va,VC=vc))
            del K
        for beta in (.5,1.,2.,3.,4.):
            lam,j=attenuation(L*L,beta)
            atten.append(dict(L=L,beta=beta,lambda1=lam,thermodynamic_lambda1=1/math.cosh(beta)**2,mean_j=j))
        print(f'L={L} finished in {time.monotonic()-tick:.3f}s',flush=True)
    for name,data in [('ground_corners.csv',rows),('attenuation.csv',atten)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fits=[]
    for mu in (0.,1e-8):
        for p in (1/3,.5):
            for lo in (48,96,192,384):
                rr=[r for r in rows if r['mu']==mu and r['p']==p and r['L']>=lo]
                L=np.array([r['L'] for r in rr],float); y=np.array([r['DV'] for r in rr])
                for order in (1,2):
                    X=np.array([np.ones(len(L)),np.log(L)]+[L**(-i) for i in range(1,order+1)]).T
                    coeff=np.linalg.lstsq(X,y,rcond=None)[0]
                    fits.append(dict(mu=mu,p=p,L_min=lo,L_max=int(L[-1]),correction_order=order,
                         coefficient=coeff.tolist(),max_abs_residual=float(np.max(np.abs(y-X@coeff))),
                         residual_dof=len(L)-len(coeff),role='deterministic finite-window fit, no MC uncertainty'))
    payload=dict(status='PASS',dense_validation=validation,fits=fits,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')},
        scope='Ground number variance only; not finite-beta Born entropy or its corner coefficient')
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


if __name__=='__main__': main()
