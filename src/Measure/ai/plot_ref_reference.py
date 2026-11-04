"""Standalone, source-backed uniform-A beta=0 Ref calibration figure.

Exports PNG/PDF plus exact plotted rows, independently fitted coefficients and
source receipts. The scientific plotting grids are explicit, never selected by
their agreement with the equilibrium coefficient.
"""
from pathlib import Path
import csv, hashlib, io, json
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis_ref_calibration'
FIG=HERE/'figures_ref_calibration'
EARLY=[2,2.5,3,3.5,4,4.5,5,5.5,6]
DRAW=EARLY+[7,8]
SIZES=[54,72,90,108]

def sha(raw):return hashlib.sha256(raw).hexdigest()
def write_csv(name,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=columns);w.writeheader();w.writerows(rows)

def corrected_fit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    a=np.column_stack((np.log(x),np.ones(len(x)),1/x))
    norm=np.linalg.norm(a,axis=0)
    z=np.linalg.lstsq(a/norm,y,rcond=None)[0]/norm
    return dict(c=float(z[0]),b=float(z[1]),d=float(z[2]),n=len(x),
                grid=','.join(f'{v:g}' for v in x),model='c ln(x) + b + d/x',
                rmse=float(np.sqrt(np.mean((a@z-y)**2))))

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    lookup={};receipts=[]
    for path in [HERE/'data/ground_observations_baseline.csv',
                 HERE/'data/ref_calibration_observations.csv']:
        raw=path.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
        receipts.append(dict(path=str(path.relative_to(HERE)),sha256=sha(raw),bytes=len(raw)))
        for lineno,row in enumerate(csv.DictReader(io.StringIO(complete.decode('utf-8-sig'))),2):
            if row['pattern']!='uniform_A' or float(row['beta'])!=0:continue
            key=(int(row['L']),float(row['theta']))
            if key in lookup:
                assert abs(float(row['D'])-lookup[key]['D'])<1e-9
            lookup[key]=dict(L=key[0],theta=key[1],D=float(row['D']),source=str(path.relative_to(HERE)),source_row=lineno)
    refpath=HERE/'Ref/neq-eqC_H0.csv'
    ref=np.loadtxt(refpath,delimiter=',')
    receipts.append(dict(path=str(refpath.relative_to(HERE)),sha256=sha(refpath.read_bytes()),bytes=refpath.stat().st_size))
    eqrows=[]
    for l,ec in ref:
        if (int(l),float('inf')) not in lookup:continue
        current=lookup[int(l),float('inf')]
        eqrows.append(dict(L=int(l),ref_EC_over_2=ec/2,current_D=current['D'],
                           current_minus_ref=current['D']-ec/2,source_row=current['source_row']))
    eqfitrows=[r for r in eqrows if r['L']>=18]
    xeq=[r['L'] for r in eqfitrows]
    feqref=corrected_fit(xeq,[r['ref_EC_over_2'] for r in eqfitrows])
    feqnew=corrected_fit(xeq,[r['current_D'] for r in eqfitrows])
    fits=[dict(series='Ref equilibrium / 2',L='',window='18-72',**feqref),
          dict(series='Current equilibrium',L='',window='18-72',**feqnew)]
    time_rows=[];timefits={}
    for l in SIZES:
        for t in DRAW:
            assert (l,t) in lookup,(l,t)
            time_rows.append(dict(lookup[l,t]))
        for label,grid in [('early 2-6',EARLY)]:
            result=corrected_fit(grid,[lookup[l,t]['D'] for t in grid])
            timefits[l,label]=result
            fits.append(dict(series='Current time',L=l,window=label,**result))
    # Reproduce the saved equilibrium figure from its actual L<=69 snapshot.
    old=ref[(ref[:,0]>=6)&(ref[:,0]<=69)]
    mins=[];slopes=[]
    for i in range(5,len(old)-6):
        a=np.column_stack((np.log(old[i:,0]),np.ones(len(old[i:]))))
        slopes.append(np.linalg.lstsq(a,old[i:,1],rcond=None)[0][0]);mins.append(old[i,0])
    saved_eq=float(np.polyfit(1/np.asarray(mins),slopes,5)[-1]/2)
    estimates=[
        ('Ref eq, same grid',feqref['c'],'computed'),
        ('Current eq, same grid',feqnew['c'],'computed'),
        ('Current time, 2-6, L108',timefits[108,'early 2-6']['c'],'computed'),
        ('Ref eq, cutoff poly-5',saved_eq,'reproduced saved snapshot'),
        ('Ref time, cutoff poly-2',.330/2,'saved label only'),
        ('Ref time, cutoff poly-3',.328/2,'saved label only'),
        ('Ref time, cutoff poly-4',.314/2,'saved label only'),
        ('Ref time, cutoff power',.310/2,'saved label only'),
    ]
    write_csv('reference_plot_equilibrium.csv',eqrows)
    write_csv('reference_plot_time_values.csv',time_rows)
    write_csv('reference_plot_fits.csv',fits)
    write_csv('reference_plot_method_comparison.csv',[dict(label=l,c=c,provenance=p) for l,c,p in estimates])

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12,
        'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig,axs=plt.subplots(2,2,figsize=(14.2,10.5))
    fig.subplots_adjust(left=.075,right=.965,bottom=.13,top=.89,wspace=.42,hspace=.43)
    fig.suptitle(r'Ref calibration: uniform A, $\beta=0$',fontsize=18,y=.965)
    fig.text(.5,.922,'Single-species second Renyi entropy, natural logarithms; identical hopping and imaginary-time units',ha='center',fontsize=11,color='#464646')
    blue='#2466A3';gold='#B46D21';grey='#4A4A4A';purple='#8B5895'
    colors={54:'#6B7D8B',72:'#B46D21',90:'#70833D',108:'#2466A3'}
    markers={54:'s',72:'^',90:'D',108:'o'}
    for ax in axs.flat:
        ax.grid(axis='y',color='#E3E5E7',linewidth=.7)
        ax.set_axisbelow(True)
    a=axs[0,0]
    a.set_title('A  Equilibrium observations at identical sizes',loc='left',pad=12)
    xx=np.log([r['L'] for r in eqrows])
    a.plot(xx,[r['current_D'] for r in eqrows],'-o',color=blue,lw=1.5,ms=4,label='Current D')
    a.plot(xx,[r['ref_EC_over_2'] for r in eqrows],'s',mfc='none',mec=gold,ms=6,mew=1.1,label=r'Ref $E_C/2$')
    a.set(xlabel=r'$\ln L$',ylabel='D',xlim=np.log([10,75]))
    a.xaxis.set_major_locator(MaxNLocator(6));a.legend(loc='upper left',frameon=False)
    a.text(.97,.05,r'$\max|D-E_C/2| = 1.15\times10^{-6}$',transform=a.transAxes,ha='right',fontsize=10,color=grey)
    inset=a.inset_axes([.52,.17,.43,.27])
    inset.plot(xx,[r['current_minus_ref']*1e6 for r in eqrows],'o-',color=grey,ms=3,lw=1)
    inset.set_title(r'Current $-$ Ref/2 ($\times10^{-6}$)',fontsize=8.5,pad=4)
    inset.xaxis.set_major_locator(MaxNLocator(3));inset.tick_params(labelsize=8)
    inset.spines['top'].set_visible(False);inset.spines['right'].set_visible(False)

    a=axs[0,1]
    a.set_title('B  Unshifted imaginary-time observations',loc='left',pad=12)
    for l in SIZES:
        a.plot(np.log(DRAW),[lookup[l,t]['D'] for t in DRAW],marker=markers[l],color=colors[l],ms=4,lw=1.35,
               mfc='white' if l!=108 else colors[l],label=f'L={l}')
    a.xaxis.set_major_locator(MaxNLocator(6))
    a.set(xlabel=r'$\ln\theta$',ylabel=r'$D(\theta)$',xlim=np.log([1.9,8.4]))
    a.legend(loc='lower right',frameon=False,ncol=2)
    a.text(.03,.97,'Every symbol is an evaluated point\nNo vertical shift or size averaging',transform=a.transAxes,va='top',fontsize=9.5,color=grey)

    a=axs[1,0]
    a.set_title('C  Fixed-model coefficient versus size',loc='left',pad=12)
    for label,color,marker in [('early 2-6',blue,'o')]:
        a.plot(SIZES,[timefits[l,label]['c'] for l in SIZES],marker=marker,color=color,lw=1.7,ms=6,
               label='Time 2-6 (9 points)')
    a.axhline(feqnew['c'],color=grey,lw=1.25,ls='--',label='Eq, common L18-72 grid')
    a.set(xlabel='L',ylabel=r'$c$ in $D=c\ln\theta+b+d/\theta$',xticks=SIZES,xlim=(50,112),ylim=(.07,.174))
    a.legend(loc='lower right',frameon=False,fontsize=9)
    a.annotate(f"{timefits[108,'early 2-6']['c']:.6f}",(108,timefits[108,'early 2-6']['c']),
               xytext=(-8,12),textcoords='offset points',ha='right',color=blue,fontsize=9)
    a.text(.03,.97,'Same three-parameter model at every size',transform=a.transAxes,va='top',fontsize=9.3,color=grey)

    a=axs[1,1]
    a.set_title('D  Normalized estimates and method sensitivity',loc='left',pad=12)
    for i,(label,value,provenance) in enumerate(estimates):
        y=len(estimates)-1-i
        color=blue if i<3 else (grey if i==3 else purple)
        marker='o' if i<3 else ('D' if i==3 else 's')
        a.plot(value,y,marker=marker,mfc=color if i<3 else 'white',mec=color,ms=6,mew=1.3)
        number=f'{value:.6f}' if i<4 else f'{value:.4f}'
        a.text(value+.00035,y,number,va='center',fontsize=9.2,color=color)
    a.axvline(feqnew['c'],color=grey,lw=1,ls='--',zorder=0)
    a.axhline(3.5,color='#D3D6D9',lw=.8)
    a.set_yticks(range(len(estimates)));a.set_yticklabels([v[0] for v in estimates][::-1],fontsize=9.2)
    a.set(xlabel='Single-species c',xlim=(.1528,.1685),ylim=(-.7,7.7))
    a.set_xticks([.155,.160,.165]);a.grid(False)
    a.text(.98,.04,'Open squares: rounded saved Ref labels\nDifferent methods, not confidence bounds',
           transform=a.transAxes,ha='right',va='bottom',fontsize=8.6,color=grey,
           bbox=dict(facecolor='white',edgecolor='none',pad=2))

    fig.text(.075,.075,'Fit grids: time = 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.  Eq common grid = 18, 24, ..., 72.  A/B axes show natural-log coordinates.',fontsize=10,color=grey)
    fig.text(.075,.050,'Ref uses twice the single-species entropy. Its saved equilibrium extrapolation used L <= 69; its nonequilibrium CSV is unavailable.',fontsize=10,color=grey)
    fig.text(.075,.025,'The figure compares declared fits and their size/model dependence; agreement is not used to choose the fitting window.',fontsize=10,color=grey)
    for ext in ('png','pdf'):
        fig.savefig(FIG/f'reference_calibration.{ext}',dpi=180)
    plt.close(fig)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__).read_bytes()),
        inputs=receipts,pattern='uniform_A',beta=0,normalization='one species, natural log Renyi2',
        grids=dict(early=EARLY,raw_time=DRAW,eq_common=xeq),
        scaling_coordinates='Panels A/B: x=ln(L) / ln(theta), linear axes; y=D',
        reference_neq_coefficients='Transcribed rounded labels only; source CSV missing; no fabricated reconstruction.',
        c_eq_ref=feqref['c'],c_eq_current=feqnew['c'],c_time_L108_early=timefits[108,'early 2-6']['c'],
        figures={ext:sha((FIG/f'reference_calibration.{ext}').read_bytes()) for ext in ('png','pdf')})
    (OUT/'reference_plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
