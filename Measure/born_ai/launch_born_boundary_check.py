"""Run each frozen boundary-check chain in a fresh hidden Julia process."""
import argparse, datetime as dt, hashlib, json, os, shutil, subprocess, time, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2),encoding='utf-8');tmp.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);args=ap.parse_args()
    directory=ROOT/'data/born_scaling/boundary_check'/args.stage;planpath=directory/'plan.toml'
    plan=tomllib.loads(planpath.read_text(encoding='utf-8'))
    for f,h in plan['source_sha256'].items():assert sha(ROOT/f)==h,f
    startpath=directory/'launch_receipt.json'
    with startpath.open('x',encoding='utf-8') as s:json.dump(dict(started_at=utc(),plan_sha256=sha(planpath)),s)
    env=dict(os.environ,JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    hidden=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    pending=list(plan['jobs']);running=[];done=[];failed=[];processes=[];started=utc()
    # Larger sizes first so their progress can overlap small jobs in the tail.
    pending.sort(key=lambda j:(-j['L'],j['chain']))
    while pending or running:
        while pending and len(running)<plan['execution_workers'] and not failed:
            job=pending.pop(0)
            logpath=directory/(job['run_id']+'.log')
            cmd=[shutil.which('julia'),'--project=.',str(ROOT/'run_born_boundary_check.jl'),
                 '--plan='+str(planpath),'--run-id='+job['run_id']]
            with logpath.open('xb') as log:
                process=subprocess.Popen(cmd,cwd=ROOT.parents[2],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=hidden)
            info=dict(run_id=job['run_id'],pid=process.pid,started_at=utc(),command=cmd)
            processes.append(info);running.append((job,process,info))
            print('LAUNCHED '+job['run_id']+' pid='+str(process.pid),flush=True)
        for job,process,info in running[:]:
            code=process.poll()
            if code is None:continue
            running.remove((job,process,info));info['exit_code']=code;info['finished_at']=utc()
            if code:failed.append(info)
            else:
                rec=tomllib.loads((ROOT/job['receipt_path']).read_text(encoding='utf-8'))
                assert rec['status']=='complete' and rec['plan_sha256']==sha(planpath)
                for f,h in rec['files'].items():assert sha(ROOT/f)==h
                assert rec['records']==job['sweeps']//job['measure_every']
                done.append(job)
            print('FINISHED '+job['run_id']+' exit='+str(code),flush=True)
        state=dict(status='running',started_at=started,updated_at=utc(),completed=len(done),total=len(plan['jobs']),
            running=[j['run_id'] for j,_,_ in running],failed=failed,pending=len(pending),processes=processes)
        save(directory/'launch_status.json',state)
        if failed and not running:break
        if pending or running:time.sleep(5)
    if failed:raise RuntimeError('Diagnostic run failed; see preserved logs')
    files={planpath.relative_to(ROOT).as_posix():sha(planpath)}
    for job in done:
        recpath=ROOT/job['receipt_path'];files[job['receipt_path']]=sha(recpath)
        rec=tomllib.loads(recpath.read_text(encoding='utf-8'));files.update(rec['files'])
        modelpath=ROOT/job['model_manifest'];files[job['model_manifest']]=sha(modelpath)
    manifest=dict(status='complete',stage=args.stage,completed_at=utc(),plan_path=planpath.relative_to(ROOT).as_posix(),
        plan_sha256=sha(planpath),jobs=plan['jobs'],files=files,source_sha256=plan['source_sha256'],
        total_records=sum(j['sweeps']//j['measure_every'] for j in done),
        retained_records=sum(j['retained_records'] for j in done))
    save(directory/'manifest.json',manifest)
    state.update(status='complete',completed_at=utc(),manifest_sha256=sha(directory/'manifest.json'))
    save(directory/'launch_status.json',state);print('COMPLETE '+args.stage,flush=True)

if __name__=='__main__':main()
