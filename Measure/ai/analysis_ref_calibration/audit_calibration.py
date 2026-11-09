"""Independent source/coverage/80-digit fit audit of the revised calibration.

Run after analyze_ref_calibration.py. Preliminary mode reproduces that analysis's
captured file prefixes while scans may append. --stage=final additionally requires
unchanged full input hashes and the declared final coverage. No primary code is
imported. All model columns, canonical grids, arithmetic means and stability
criteria are reconstructed from raw observations and the saved protocol.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
from datetime import datetime, timezone
import argparse,csv,hashlib,io,json,math

HERE=Path(__file__).resolve().parent
AI=HERE.parent
MODELS={'log':('c','b'),'log_plus_inv1':('c','b','d'),
        'log_plus_inv2':('c','b','e'),'log_plus_inv1_inv2':('c','b','d','e')}
ALL_MODELS={**MODELS,'log_plus_inv1_inv2_inv3':('c','b','d','e','f')}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def readcsv(path):return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
def savecsv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def key(r):return r['pattern'],float(r['beta']),int(r['L']),float(r['theta'])
def fkey(r):return r['domain'],r['pattern'],float(r['beta']),str(r['L']),r['window'],r['model']
def solve(a,b):
    n=len(b);m=[list(x)+[y] for x,y in zip(a,b)]
    for c in range(n):
        q=max(range(c,n),key=lambda i:abs(m[i][c]));m[c],m[q]=m[q],m[c]
        assert m[c][c]
        for i in range(c+1,n):
            z=m[i][c]/m[c][c]
            for j in range(c,n+1):m[i][j]-=z*m[c][j]
    out=[D(0)]*n
    for i in range(n-1,-1,-1):out[i]=(m[i][-1]-sum(m[i][j]*out[j] for j in range(i+1,n)))/m[i][i]
    return out
def fit80(grid,values,model):
    with localcontext() as c:
        c.prec=80
        names=ALL_MODELS[model];rows=[]
        for raw in grid:
            x=D(str(raw));v={'c':x.ln(),'b':D(1),'d':1/x,'e':1/x**2,'f':1/x**3}
            rows.append([v[n] for n in names])
        n=len(names)
        gram=[[sum(r[i]*r[j] for r in rows) for j in range(n)] for i in range(n)]
        rhs=[sum(r[i]*v for r,v in zip(rows,values)) for i in range(n)]
        co=solve(gram,rhs);res=[v-sum(x*z for x,z in zip(r,co)) for r,v in zip(rows,values)]
        return dict(zip(names,co)),(sum(r*r for r in res)/len(res)).sqrt(),max(abs(r) for r in res)

def audit_selection(merged,protocol,summary,fitcoeff,errors):
    """Qualification uses only finite-time coefficients, never equilibrium data."""
    rules=protocol['controlled_window_selection'];floor=D(str(rules['denominator_floor']))
    trim_rows=readcsv(HERE/'calibration_selection_fits.csv')
    trim_actual={(r['pattern'],float(r['beta']),r['window'],r['trim']):r for r in trim_rows}
    candidate_rows=readcsv(HERE/'calibration_candidates.csv')
    candidate_actual={(r['pattern'],float(r['beta']),r['window']):r for r in candidate_rows}
    selected_rows=readcsv(HERE/'calibration_selection.csv')
    selected_actual={(r['pattern'],float(r['beta'])):r for r in selected_rows}
    trimmed=[];candidates=[];selections=[];trim_expected=set();candidate_expected=set();selection_expected=set()
    coefficient_error=D(0);metric_error=D(0);metric_worst={}
    def number_check(actual,field,value,context):
        nonlocal metric_error,metric_worst
        if value is None:
            if actual[field]!='':errors.append(f'Nonempty missing selection field {context}/{field}')
        else:
            if actual[field]=='':errors.append(f'Missing selection value {context}/{field}');return
            delta=abs(D(str(value))-D(actual[field]))
            if delta>metric_error:
                metric_error=delta
                metric_worst=dict(context=str(context),field=field,recomputed=str(value),published=actual[field],
                    absolute_error=float(delta),relative_error=float(delta/max(abs(D(str(value))),D('1e-30'))))
            if delta>D('1e-8'):errors.append(f'Selection number mismatch {context}/{field}: {delta}')
    def bool_check(actual,field,value,context):
        if (actual[field].lower()=='true')!=value:errors.append(f'Selection gate mismatch {context}/{field}')
    def cget(p,b,L,w,model='log_plus_inv1'):
        return fitcoeff.get(('time',p,b,str(L),w,model),{}).get('c')
    for p in protocol['pattern_order']:
        for b in summary['analyzed_betas']:
            group=[];selection_expected.add((p,b))
            for window in rules['candidate_windows']:
                low,high=protocol['time_windows'][window]
                grid=[t for t in protocol['time_grid'] if low<=t<=high]
                ck=(p,b,window);candidate_expected.add(ck)
                actual=candidate_actual[ck]
                main=cget(p,b,108,window);small=cget(p,b,90,window)
                four=cget(p,b,108,window,'log_plus_inv1_inv2');trims={}
                if main is not None:
                    for name,g in [('delete_first',grid[1:]),('delete_last',grid[:-1])]:
                        if len(g)<5:continue
                        tk=(p,b,window,name);trim_expected.add(tk)
                        co,rmse,maxres=fit80(g,[merged[p,b,108,t]['D'] for t in g],'log_plus_inv1')
                        trims[name]=co['c'];r=trim_actual[tk]
                        if r['domain']!='time_trim' or r['L']!='108' or r['model']!='log_plus_inv1':errors.append(f'Trim identity mismatch {tk}')
                        if [float(z) for z in r['grid'].split(',')]!=g:errors.append(f'Trim grid mismatch {tk}')
                        err=max(abs(co[k]-D(r[k])) for k in co);coefficient_error=max(coefficient_error,err)
                        if err>D('1e-8'):errors.append(f'Trim fit mismatch {tk}: {err}')
                        trimmed.append(dict(pattern=p,beta=b,window=window,trim=name,grid=r['grid'],
                            n=len(g),c_high_precision=str(co['c']),c_published=r['c'],c_error=float(abs(co['c']-D(r['c']))),
                            maximum_parameter_error=float(err),rmse_error=float(abs(rmse-D(r['rmse'])))))
                other=[small,four,trims.get('delete_first'),trims.get('delete_last')]
                den=abs(main) if main is not None and abs(main)>=floor else None
                drift=[abs(x-main)/den if x is not None and den is not None else None for x in other]
                thresholds=[rules[k] for k in ('size_relative_difference_max','nested_four_parameter_relative_difference_max',
                            'delete_first_relative_difference_max','delete_last_relative_difference_max')]
                gate=[v is not None and v<=D(str(t)) for v,t in zip(drift,thresholds)]
                complete=main is not None and all(x is not None for x in other)
                sign=complete and den is not None and all(x*main>0 for x in other)
                qualified=complete and den is not None and all(gate) and sign
                item=dict(pattern=p,beta=b,window=window,theta_min=low,theta_max=high,
                    c=main,c_90=small,c_four108=four,c_delete_first=other[2],c_delete_last=other[3],
                    diagnostic_denominator=den,size_drift=drift[0],four_parameter_drift=drift[1],
                    delete_first_drift=drift[2],delete_last_drift=drift[3],qualified=qualified)
                diffs=[(abs(merged[p,b,108,t]['D']-merged[p,b,90,t]['D']),abs(merged[p,b,108,t]['D']))
                       for t in grid if (p,b,108,t) in merged and (p,b,90,t) in merged]
                item['max_absolute_D_difference']=max((x for x,y in diffs),default=None)
                item['max_relative_D_difference']=max((x/y for x,y in diffs if y>=floor),default=None)
                for field in ('c','c_90','c_four108','c_delete_first','c_delete_last','diagnostic_denominator',
                              'size_drift','four_parameter_drift','delete_first_drift','delete_last_drift'):
                    number_check(actual,field,item[field],ck)
                for field in ('max_absolute_D_difference','max_relative_D_difference'):
                    if field in actual:number_check(actual,field,item[field],ck)
                for field,value in zip(('size_gate','four_parameter_gate','delete_first_gate','delete_last_gate'),gate):bool_check(actual,field,value,ck)
                for field,value in [('checks_available',complete),('sign_gate',sign),('qualified',qualified)]:bool_check(actual,field,value,ck)
                if [float(z) for z in actual['grid'].split(',')]!=grid:errors.append(f'Candidate grid mismatch {ck}')
                candidates.append(item);group.append(item)
            # Sorting uses only declared bounds of qualified finite-time fits.
            qualified=sorted((r for r in group if r['qualified']),key=lambda r:(r['theta_min'],r['theta_max']),reverse=True)
            best=qualified[0] if qualified else None
            actual=selected_actual[p,b]
            expected_window=best['window'] if best else ''
            if actual['window']!=expected_window or actual['selected_window']!=expected_window:errors.append(f'Selected window mismatch {(p,b)}')
            if int(actual['qualified_candidate_count'])!=len(qualified):errors.append(f'Qualified count mismatch {(p,b)}')
            if actual['qualified_windows']!=','.join(r['window'] for r in qualified):errors.append(f'Qualified order mismatch {(p,b)}')
            if actual['status']!=('qualified_finite_scale_candidate' if best else 'no_stable_candidate'):errors.append(f'Selection status mismatch {(p,b)}')
            bool_check(actual,'selection_is_posthoc',True,(p,b))
            for field in ('c','c_90','c_four108','c_delete_first','c_delete_last','size_drift','four_parameter_drift','delete_first_drift','delete_last_drift'):
                number_check(actual,field,best[field] if best else None,(p,b))
            for field in ('max_absolute_D_difference','max_relative_D_difference'):
                if field in actual:number_check(actual,field,best[field] if best else None,(p,b))
            aliases={'selected_c':'c','selected_model':'model','selected_L':'L','selected_size_drift':'size_drift',
                'selected_model_drift':'four_parameter_drift','selected_lower_trim_drift':'delete_first_drift','selected_upper_trim_drift':'delete_last_drift'}
            for alias,field in aliases.items():
                if actual[alias]!=actual[field]:errors.append(f'Selection alias mismatch {(p,b,alias)}')
            for item in group:bool_check(candidate_actual[p,b,item['window']],'selected',item is best,(p,b,item['window']))
            selections.append(dict(pattern=p,beta=b,selected_window=expected_window,selected_c=str(best['c']) if best else '',
                qualified_count=len(qualified),qualified_windows=','.join(r['window'] for r in qualified),
                selection_inputs='finite-time fits only; no equilibrium coefficient or raw-D threshold'))
    for name,expected,actual in [('trim',trim_expected,set(trim_actual)),('candidate',candidate_expected,set(candidate_actual)),('selection',selection_expected,set(selected_actual))]:
        if expected!=actual:errors.append(f'{name} key mismatch')
    savecsv('audit_calibration_trim_fits.csv',trimmed)
    savecsv('audit_calibration_candidates.csv',candidates)
    savecsv('audit_calibration_selection.csv',selections)
    return dict(checked_trim_fits=len(trimmed),checked_candidates=len(candidates),checked_selections=len(selections),
                max_trim_parameter_error=float(coefficient_error),max_candidate_or_selection_numeric_error=float(metric_error),
                worst_candidate_or_selection_numeric_difference=metric_worst,
                selected_combinations=sum(bool(r['selected_window']) for r in selections),
                target_independence='Independent selection consumes only finite-time coefficients, point-count rules, relative drift thresholds, signs and window-bound ordering. Equilibrium values never enter qualification or ranking.')

def audit_order(merged,protocol,summary,errors):
    rules=protocol['model_order_diagnostic'];floor=D(str(rules['denominator_floor']))
    rows=readcsv(HERE/'calibration_order_fits.csv');published={fkey(r):r for r in rows}
    diagnostic=readcsv(HERE/'calibration_order_diagnostics.csv')
    expected={};coeffs={};comparisons=[];max_c=D(0);max_param=D(0);max_metric=D(0)
    for p in rules['patterns']:
        for b in summary['analyzed_betas']:
            for window,(low,high) in rules['windows'].items():
                grid=[t for t in protocol['time_grid'] if low<=t<=high]
                if any((p,b,L,t) not in merged for L in rules['sizes'] for t in grid):continue
                for model in rules['models']:
                    assert len(grid)>=len(ALL_MODELS[model])+2
                    for L in rules['sizes']:
                        k=('time_order',p,b,str(L),window,model)
                        expected[k]=(grid,[merged[p,b,L,t]['D'] for t in grid])
    if set(expected)!=set(published):errors.append('Model-order fit key mismatch')
    for k,(grid,values) in expected.items():
        r=published[k];co,rmse,mr=fit80(grid,values,k[-1]);coeffs[k]=co
        if [float(x) for x in r['grid'].split(',')]!=grid:errors.append(f'Model-order grid mismatch {k}')
        delta={name:abs(value-D(r[name])) for name,value in co.items()}
        max_c=max(max_c,delta['c']);max_param=max(max_param,max(delta.values()))
        if max(delta.values())>D('1e-8'):errors.append(f'Model-order coefficient mismatch {k}')
        if abs(rmse-D(r['rmse']))>D('1e-10'):errors.append(f'Model-order residual mismatch {k}')
        comparisons.append(dict(domain=k[0],pattern=k[1],beta=k[2],L=k[3],window=k[4],model=k[5],grid=r['grid'],
            n=len(grid),c_high_precision=str(co['c']),c_published=r['c'],c_error=float(delta['c']),
            maximum_parameter_error=float(max(delta.values())),rmse_error=float(abs(rmse-D(r['rmse'])))))
    def get(p,b,L,w,m):return coeffs['time_order',p,b,str(L),w,m]
    def relative(value,den):return value/abs(den) if abs(den)>=floor else None
    diagnostic_keys=set()
    for r in diagnostic:
        p,b,w,m=r['pattern'],float(r['beta']),r['window'],r['model'];diagnostic_keys.add((p,b,w,m))
        if r['status']!='available':
            if ('time_order',p,b,'108',w,m) in expected:errors.append(f'Order diagnostic incorrectly missing {(p,b,w,m)}')
            continue
        lo=get(p,b,90,w,m);hi=get(p,b,108,w,m)
        first=get(p,b,108,'3-8',m)['c'];last=get(p,b,108,'4-8',m)['c']
        four=get(p,b,108,w,'log_plus_inv1_inv2')['c'];five=get(p,b,108,w,'log_plus_inv1_inv2_inv3')['c']
        grid=expected['time_order',p,b,'108',w,m][0]
        nums=dict(c_90=lo['c'],c_108=hi['c'],size_drift=relative(abs(lo['c']-hi['c']),hi['c']),
            c_window3_8=first,c_window4_8=last,window_drift=relative(abs(first-last),last),
            c_four108=four,c_five108=five,order_drift=relative(abs(four-five),four),
            max_absolute_D_difference=max(abs(merged[p,b,108,t]['D']-merged[p,b,90,t]['D']) for t in grid))
        with localcontext() as ctx:
            ctx.prec=80
            span=abs(hi['c'])*(D(str(max(grid)))/D(str(min(grid)))).ln()
            for power,name in [(1,'d'),(2,'e'),(3,'f')]:
                field=f'inverse{power}_to_log_span'
                nums[field]=relative(abs(hi[name])*(1/D(str(min(grid)))**power-1/D(str(max(grid)))**power),span) if name in hi else None
        for name,value in nums.items():
            if value is None:
                if r[name]!='':errors.append(f'Nonempty missing order metric {(p,b,w,m,name)}')
            else:
                delta=abs(value-D(r[name]));max_metric=max(max_metric,delta)
                if delta>D('1e-7'):errors.append(f'Order metric mismatch {(p,b,w,m,name)}: {delta}')
        for name,value in [('size_sign_changed',lo['c']*hi['c']<=0),('window_sign_changed',first*last<=0),('order_sign_changed',four*five<=0)]:
            if (r[name].lower()=='true')!=value:errors.append(f'Order sign mismatch {(p,b,w,m,name)}')
    expected_diag={(p,b,w,m) for p in rules['patterns'] for b in summary['analyzed_betas'] for w in rules['windows'] for m in rules['models']}
    if diagnostic_keys!=expected_diag:errors.append('Order diagnostic key mismatch')
    savecsv('audit_calibration_order_fits.csv',comparisons)
    return dict(checked_fits=len(comparisons),checked_diagnostics=len(diagnostic),max_c_error=float(max_c),
        max_parameter_error=float(max_param),max_diagnostic_numeric_error=float(max_metric),
        role='Model-order sensitivity only; no fit from this table enters local candidate qualification.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preliminary','final'],default='preliminary')
    args=parser.parse_args()
    summary=json.loads((HERE/'calibration_summary.json').read_text())
    protocol=json.loads((HERE/'protocol.json').read_text())
    errors=[];receipts=[];merged={};duplicates=0
    assert sha((HERE/'protocol.json').read_bytes())==summary['protocol_sha256'],'protocol changed'
    assert sha((AI/'analyze_ref_calibration.py').read_bytes())==summary['script_sha256'],'main script changed since output'
    for receipt in summary['sources']:
        if receipt.get('missing'):continue
        path=AI/receipt['path'];now=path.read_bytes();raw=now[:receipt['bytes']]
        if sha(raw)!=receipt['sha256']:errors.append(f'Cannot reconstruct source snapshot: {path}')
        if args.stage=='final' and sha(now)!=receipt['sha256']:errors.append(f'Final source changed: {path}')
        complete=raw[:raw.rfind(b'\n')+1]
        rows=list(csv.DictReader(io.StringIO(complete.decode('utf-8-sig'))))
        receipts.append(dict(path=str(path),snapshot_matches=sha(raw)==receipt['sha256'],
            current_full_matches=sha(now)==receipt['sha256'],snapshot_sha256=sha(raw),snapshot_rows=len(rows)))
        for lineno,r in enumerate(rows,2):
            p,b,L,t=key(r)
            if p not in protocol['pattern_order'] or b not in summary['analyzed_betas']:continue
            if (math.isinf(t) and not 18<=L<=144) or (math.isfinite(t) and not 0<t<=16):continue
            if str(r.get('premeasurement_ground',r.get('ground','true'))).lower() not in ('true','1'):
                errors.append(f'Non-ground row {path}:{lineno}')
            excess=D(r.get('premeasurement_excess') or r.get('premeasurement_limit_excess') or r.get('accessible_excess') or '0')
            if abs(excess)>=D('1e-8'):errors.append(f'GS excess {path}:{lineno}')
            values={z:D(r[z]) for z in ('D','S_A','S_C','S_B')}
            if abs(values['D']-(2*values['S_A']+values['S_C'])/3+values['S_B'])>D('1e-9'):
                errors.append(f'D identity {path}:{lineno}')
            k=(p,b,L,t)
            if k in merged:
                duplicates+=1
                if max(abs(values[z]-merged[k][z]) for z in values)>D('1e-8'):errors.append(f'Conflicting duplicate {k}')
            merged[k]=values
    published=readcsv(HERE/'calibration_points.csv')
    pubmap={key(r):r for r in published}
    if set(pubmap)!=set(merged):errors.append('Published point keys differ from independent source merge')
    point_error=max((abs(D(r[z])-merged[k][z]) for k,r in pubmap.items() if k in merged for z in ('D','S_A','S_C','S_B')),default=D(0))
    if point_error>D('1e-12'):errors.append(f'Published values changed {point_error}')
    expected={};coverage=[]
    def add(domain,p,b,L,window,grid,values):
        for model,names in MODELS.items():
            if len(grid)>=len(names)+2:expected[(domain,p,b,str(L),window,model)]=(grid,values)
    def time_group(domain,p,b,sizes,windows):
        for window,grid in windows.items():
            if any((p,b,L,t) not in merged for L in sizes for t in grid):continue
            vectors=[[merged[p,b,L,t]['D'] for t in grid] for L in sizes]
            for L,values in zip(sizes,vectors):add(domain,p,b,L,window,grid,values)
            with localcontext() as c:
                c.prec=80
                avg=[sum(v[i] for v in vectors)/len(vectors) for i in range(len(grid))]
            add(domain,p,b,'average'+'_'.join(map(str,sizes)),window,grid,avg)
    for p in protocol['pattern_order']:
        for b in summary['analyzed_betas']:
            tw={label:[x for x in protocol['time_grid'] if bounds[0]<=x<=bounds[1]] for label,bounds in protocol['time_windows'].items()}
            time_group('time',p,b,protocol['time_sizes'],tw)
            for L in protocol['time_sizes']:
                missing=[t for t in protocol['time_grid'] if (p,b,L,t) not in merged]
                coverage.append(dict(domain='time',pattern=p,beta=b,L=L,complete=not missing,missing=missing))
            eq={L:v['D'] for (pp,bb,L,t),v in merged.items() if pp==p and bb==b and math.isinf(t)}
            for L in protocol['equilibrium_extension_required_sizes']:
                coverage.append(dict(domain='equilibrium_extension',pattern=p,beta=b,L=L,complete=L in eq,missing=[] if L in eq else [L]))
            for label,(low,high) in protocol['equilibrium_windows'].items():
                grid=sorted(L for L in eq if low<=L<=high)
                if not grid or grid[0]!=low or grid[-1]!=high:continue
                if high==144 and any(L not in eq for L in protocol['equilibrium_extension_required_sizes']):continue
                add('equilibrium',p,b,'',label,grid,[eq[L] for L in grid])
    late=protocol.get('late_beta0_diagnostic',{})
    if 0 in summary['analyzed_betas'] and late:
        for p in late['patterns']:
            time_group('time_late',p,0,late['sizes'],late['windows'])
            required=sorted(set(t for g in late['windows'].values() for t in g))
            for L in late['sizes']:
                missing=[t for t in required if (p,0,L,t) not in merged]
                coverage.append(dict(domain='time_late',pattern=p,beta=0,L=L,complete=not missing,missing=missing))
    tables=readcsv(HERE/'calibration_fits.csv');actual={fkey(r):r for r in tables}
    if len(actual)!=len(tables):errors.append('Duplicate published fit keys')
    if set(actual)!=set(expected):errors.append(f'Fit key mismatch missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}')
    comparison=[];fitcoeff={};max_c=0.;max_coef=0.;max_rmse=0.
    for k,(grid,values) in expected.items():
        if k not in actual:continue
        row=actual[k]
        if [float(x) for x in row['grid'].split(',')]!=list(grid):errors.append(f'Grid differs {k}')
        coeff,rmse,maxres=fit80(grid,values,k[-1]);fitcoeff[k]=coeff
        delta={name:abs(co-D(row[name])) for name,co in coeff.items()}
        max_c=max(max_c,float(delta['c']));max_coef=max(max_coef,max(map(float,delta.values())))
        rmse_error=abs(rmse-D(row['rmse']));max_rmse=max(max_rmse,float(rmse_error))
        comparison.append(dict(domain=k[0],pattern=k[1],beta=k[2],L=k[3],window=k[4],model=k[5],
            n=len(grid),grid=row['grid'],c_high_precision=str(coeff['c']),c_published=row['c'],
            c_error=float(delta['c']),maximum_parameter_error=max(map(float,delta.values())),rmse_error=float(rmse_error)))
        if max(delta.values())>D('1e-8') or rmse_error>D('1e-10'):errors.append(f'Numerical fit mismatch {k}')
    # Recheck the main diagnostic from independently recomputed coefficients.
    gate_mismatches=[];comparison_max_error=0.
    def coef(p,b,L=108,w='2-6',m='log_plus_inv1',domain='time'):
        return fitcoeff.get((domain,p,b,str(L),w,m),{}).get('c')
    criteria=protocol['diagnostics'];floor=D(str(criteria['denominator_floor']))
    for r in readcsv(HERE/'calibration_comparison.csv'):
        p,b=r['pattern'],float(r['beta']);c=coef(p,b)
        check={'c_main108':c,'c_90':coef(p,b,90),'c_average':coef(p,b,'average90_108'),
               'c_purelog108':coef(p,b,m='log'),'c_purelog_average':coef(p,b,'average90_108',m='log'),
               'c_inverse_square108':coef(p,b,m='log_plus_inv2'),'c_four108':coef(p,b,m='log_plus_inv1_inv2'),
               'c_upper_window108':coef(p,b,w='2-5'),'c_lower_window108':coef(p,b,w='3-6'),
               'c_eq_old':coef(p,b,'','18-90',domain='equilibrium'),
               'c_eq_extended':coef(p,b,'','18-144',domain='equilibrium')}
        for field,value in check.items():
            if value is None:
                if r[field]!='':errors.append(f'Expected missing coefficient {p}/{b}/{field}')
            else:comparison_max_error=max(comparison_max_error,float(abs(value-D(r[field]))))
        others=[check[n] for n in ('c_90','c_upper_window108','c_lower_window108','c_four108')]
        complete=c is not None and all(x is not None for x in others)
        den=abs(c) if c is not None and abs(c)>=floor else None
        stable=False
        if complete and den is not None:
            thresholds=[criteria[z] for z in ('size_coefficient_difference_max','upper_cut_relative_difference_max',
                        'lower_cut_relative_difference_max','four_parameter_relative_difference_max')]
            stable=all(abs(x-c)/den<=D(str(t)) for x,t in zip(others,thresholds)) and all(x*c>0 for x in others)
        if stable!=(r['stable_diagnostic'].lower()=='true'):gate_mismatches.append((p,b))
    if gate_mismatches:errors.append(f'Stability classification mismatches: {gate_mismatches}')
    if comparison_max_error>1e-8:errors.append(f'Coefficient comparison table mismatch {comparison_max_error}')
    selection_audit=audit_selection(merged,protocol,summary,fitcoeff,errors) if 'controlled_window_selection' in protocol else {}
    order_audit=audit_order(merged,protocol,summary,errors) if 'model_order_diagnostic' in protocol else {}
    required_missing=[r for r in coverage if not r['complete'] and ((r['domain']=='time_late' and summary['require_late']) or
                      (r['domain']=='time' and r['beta'] in summary['required_betas']) or
                      (r['domain']=='equilibrium_extension' and r['beta'] in summary.get('required_equilibrium_betas',summary['required_betas'])))]
    if args.stage=='final':
        if summary['stage']!='final':errors.append('Primary stage is not final')
        if required_missing:errors.append(f'Required grid incomplete: {len(required_missing)}')
    pubcov={(r['domain'],r['pattern'],float(r['beta']),int(r['L'])):r for r in readcsv(HERE/'calibration_coverage.csv')}
    for r in coverage:
        k=r['domain'],r['pattern'],r['beta'],r['L']
        if k not in pubcov or (pubcov[k]['complete'].lower()=='true')!=r['complete']:errors.append(f'Coverage mismatch {k}')
    savecsv('audit_calibration_fits.csv',comparison)
    savecsv('audit_calibration_coverage.csv',coverage)
    report=dict(stage=args.stage,status='PASS' if not errors else 'FAIL',created_utc=datetime.now(timezone.utc).isoformat(),
        errors=errors,source_receipts=receipts,independent_selected_rows=len(merged),published_selected_rows=len(pubmap),
        duplicates=duplicates,point_max_error=float(point_error),expected_fits=len(expected),checked_fits=len(comparison),
        max_c_error=max_c,max_parameter_error=max_coef,max_rmse_error=max_rmse,
        comparison_max_c_error=comparison_max_error,classification_mismatches=gate_mismatches,
        controlled_selection_audit=selection_audit,
        model_order_audit=order_audit,
        required_missing=required_missing,method='80-digit Decimal logarithms and pivoted normal-equation elimination; independent grids and source merge',
        hashes={str(p.relative_to(AI)):sha(p.read_bytes()) for p in [Path(__file__),HERE/'protocol.json',
            HERE/'calibration_summary.json',HERE/'calibration_fits.csv',HERE/'calibration_comparison.csv',
            HERE/'calibration_candidates.csv',HERE/'calibration_selection.csv',HERE/'calibration_selection_fits.csv',
            HERE/'calibration_order_fits.csv',HERE/'calibration_order_diagnostics.csv',
            AI/'analyze_ref_calibration.py']})
    (HERE/'audit_calibration_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_receipts','hashes','required_missing')},indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
