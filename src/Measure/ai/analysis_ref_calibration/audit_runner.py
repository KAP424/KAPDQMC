"""Independent light audit of Ref calibration runner outputs; no entropy work."""
from pathlib import Path
import csv, hashlib, io, json, math, tomllib
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
AI=HERE.parent
DATA=AI/'data'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def key(r):return (r['pattern'],int(r['L']),float(r['theta']),float(r['beta']))
def recorded_code_source(filename,expected_hash):
    """Verify the source actually recorded by the run, including archived versions."""
    for path in (AI/filename,DATA/'provenance'/'code'/expected_hash/filename):
        if path.is_file() and digest(path.read_bytes())==expected_hash:return path
    return None
def snapshot(path):
    raw=path.read_bytes()
    complete=raw[:raw.rfind(b'\n')+1]
    return list(csv.DictReader(io.StringIO(complete.decode('utf-8-sig')))),dict(
        path=str(path),bytes_captured=len(raw),complete_bytes=len(complete),
        complete_sha256=digest(complete),incomplete_tail_bytes=len(raw)-len(complete))
def main():
    errors=[];receipts=[];comparisons=[];manifest_checks=[]
    original=list(csv.DictReader((DATA/'patterns.csv').open()))
    bits={r['pattern']:r['occupied_bits'] for r in original}
    bits.update(special_matched_a='100110011010101010',special_matched_b='101001011010101010',
                general_matched_a='011010111001011000',general_matched_b='101001010110101100')
    baseline,ar=snapshot(DATA/'ground_observations_baseline.csv')
    prior={key(r):r for r in baseline}
    manifests={}
    for path in sorted(DATA.glob('ref_calibration_run_*.toml')):
        m=tomllib.loads(path.read_text(encoding='utf-8'))
        manifests[m['run_id']]=(path,m)
    codefiles=('ref_calibration_experiments.jl','fermion_experiment.jl','real_entropy_extension.jl',
               'experiment_patterns.jl','arrangement_patterns.jl')
    current={f:digest((AI/f).read_bytes()) for f in codefiles}
    stats={};all_seen={};perrun={}
    paths=[DATA/'ref_calibration_validation.csv',DATA/'ref_calibration_observations.csv']
    # Additional outputs are included if referenced by a run manifest.
    paths=sorted(set(paths)|{Path(m['output']) for _,m in manifests.values()})
    for path in paths:
        if not path.exists():continue
        rows,receipt=snapshot(path);receipts.append(receipt)
        seen=set(); maxima={name:0. for name in ('D_error','delta_error','ground_excess','ground_residual',
            'projector_error','reconstruction_error','imag_error','real_entropy_bound','matched_entropy_difference')}
        for row in rows:
            k=key(row)
            if k in seen:errors.append(f'duplicate key in {path.name}: {k}')
            seen.add(k);perrun[row['run_id']]=perrun.get(row['run_id'],0)+1
            L=int(row['L']); m=int(row['m'])
            if L%(3*m):errors.append(f'cut phase not aligned: {k}')
            if row['occupied_bits']!=bits.get(row['pattern']):errors.append(f'pattern mismatch: {k}')
            b=row['occupied_bits']
            if len(b)!=2*m*m or b.count('1')!=m*m:errors.append(f'not half filling: {k}')
            if (all(b[2*j:2*j+2] in ('01','10') for j in range(m*m))) != (row['family']=='special'):
                errors.append(f'family mismatch: {k}')
            if row['premeasurement_ground'].lower()!='true':errors.append(f'ground flag false: {k}')
            run_record=manifests.get(row['run_id'])
            if run_record is None:errors.append(f'missing run manifest: {k}')
            elif row['code_sha256']!=run_record[1]['code_sha256']['ref_calibration_experiments.jl']:
                errors.append(f'row runner hash differs from its run manifest: {k}')
            sa,sc,sb,d=[float(row[z]) for z in ('S_A','S_C','S_B','D')]
            values=dict(D_error=abs(d-((2*sa+sc)/3-sb)),delta_error=abs(float(row['delta'])-(sa-sb)),
                ground_excess=abs(float(row['premeasurement_excess'])),ground_residual=float(row['premeasurement_residual']),
                projector_error=float(row['projector_error']),reconstruction_error=float(row['reconstruction_error']),
                imag_error=float(row['rectangle_imag_error']),real_entropy_bound=float(row['rectangle_real_error_bound']))
            for name,value in values.items():maxima[name]=max(maxima[name],value)
            if min(sa,sc,sb)<-1e-8:errors.append(f'negative S2: {k}')
            if float(row['excess_energy']) < -1e-7:errors.append(f'below GS energy: {k}')
            if not 0<=float(row['min_initial_sv'])<=1+1e-12:errors.append(f'overlap singular value out of range: {k}')
            if not -1e-9<=float(row['empty_occupancy'])<=L*L+1e-9:errors.append(f'empty occupancy out of range: {k}')
            if k in prior:
                diff=max(abs(float(row[z])-float(prior[k][z])) for z in ('S_A','S_C','S_B','D'))
                maxima['matched_entropy_difference']=max(maxima['matched_entropy_difference'],diff)
                comparisons.append(dict(source=path.name,pattern=k[0],L=k[1],theta=k[2],beta=k[3],max_S_D_error=diff))
            if k in all_seen and path.name!='ref_calibration_validation.csv':
                other=all_seen[k]
                if max(abs(float(row[z])-float(other[z])) for z in ('S_A','S_C','S_B','D'))>1e-9:
                    errors.append(f'conflicting outputs: {k}')
            all_seen[k]=row
        limits=dict(D_error=1e-11,delta_error=1e-11,ground_excess=1e-8,ground_residual=1e-9,
                    projector_error=1e-10,reconstruction_error=1e-9,imag_error=2e-10,real_entropy_bound=1e-9,
                    matched_entropy_difference=1e-9)
        for name,value in maxima.items():
            if value>limits[name]:errors.append(f'{path.name}: {name} exceeds tolerance ({value})')
        stats[path.name]=dict(rows=len(rows),patterns=sorted({r['pattern'] for r in rows}),maxima=maxima)
    for runid,(path,m) in manifests.items():
        sources={f:recorded_code_source(f,h) for f,h in m['code_sha256'].items()}
        unverified=[f for f,path in sources.items() if path is None]
        if unverified:errors.append(f'recorded source cannot be verified for {runid}: {unverified}')
        changed=[f for f,h in m['code_sha256'].items() if current.get(f)!=h]
        baseline_hash=m.get('baseline_sha256',m.get('prior_atlas_sha256'))
        baseline_match=baseline_hash==ar['complete_sha256']
        if not baseline_match:errors.append(f'baseline hash differs from recorded input for {runid}')
        count=perrun.get(runid,0)
        if m['status']=='complete' and count!=m['new_jobs']:errors.append(f'incomplete completed run {runid}: {count}/{m["new_jobs"]}')
        output=Path(m['output'])
        # A completion hash describes the prefix before subsequent runs appended.
        hash_match=None
        if output.exists() and m['status']=='complete':
            lines=output.read_bytes().splitlines(keepends=True)
            header=next(csv.reader([lines[0].decode()]));rid=header.index('run_id')
            last=max((i for i,l in enumerate(lines[1:],1) if len(list(csv.reader([l.decode()]))[0])>rid and list(csv.reader([l.decode()]))[0][rid]==runid),default=0)
            prefix=b''.join(lines[:last+1])
            hash_match=digest(prefix)==m['output_sha256_at_completion']
            if not hash_match:errors.append(f'completion prefix hash mismatch for {runid}')
        manifest_checks.append(dict(run_id=runid,status=m['status'],expected_rows=m['new_jobs'],captured_rows=count,
            current_code_differs=changed,unverified_recorded_code=unverified,
            recorded_code_sources={f:str(path.relative_to(AI)) if path is not None else None for f,path in sources.items()},
            recorded_baseline_sha256=baseline_hash,baseline_hash_matches=baseline_match,
            completion_prefix_hash_matches=hash_match,manifest_sha256=digest(path.read_bytes())))
    if comparisons:
        with (HERE/'runner_validation_matches.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=comparisons[0]);writer.writeheader();writer.writerows(comparisons)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PASS_SNAPSHOT' if not errors else 'FAIL',
        errors=errors,outputs=stats,receipts=receipts,baseline_receipt=ar,manifests=manifest_checks,
        code_sha256=current,audit_script_sha256=digest(Path(__file__).read_bytes()),
        static_review={
          'ground_semantics':'premeasurement_ground means the unmeasured theta=Inf accessible limit has global ground energy and zero residual; it does not assert the finite-theta state is already a ground state.',
          'evolution':'state=FE.evolve(prepared,theta), cached only within exact pattern,L,theta; every beta measures the same unmeasured state, not a previously measured state.',
          'measurement':'FE.measure copies orbitals and scales original empty rows by exp(-2beta), then thin QR; beta zero returns unchanged state.',
          'geometry':'L divisible by 3m, fixed origin (1,1), A=L/3 by L, C=L by L/3, B=L/3 by 2L/3; D=(2SA+SC)/3-SB.',
          'entropy':'one species, natural log, real covariance checked before discarding imaginary part; Cholesky of .5I+2(C-.5I)^2 computes -logdet and has spectrum [.5,1].',
          'diagnostics':'energy/excess/residual/overlap/empty occupancy describe measured state; premeasurement_excess/residual describe theta=Inf unmeasured limit.',
          'provenance':'Each row hash must match its run manifest. Each recorded source hash is verified against current or archived exact source bytes. The relocated baseline must match the original recorded input hash. Completion prefix hashes accommodate later appends.',
          'scope':'Read-only static review and existing numeric record checks. No new large-region entropy computation.'})
    (HERE/'runner_audit_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],errors=errors,outputs=stats,manifests=manifest_checks),indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
