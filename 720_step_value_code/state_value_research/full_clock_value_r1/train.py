"""Frozen global teachers; OOF full-future labels; no preference training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,hashlib,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,joblib
from catboost import CatBoostRegressor
HERE=Path(__file__).resolve().parent; DATA=HERE/'data'; RUN=HERE/'run0'
SEEDS=(93027,93028,93029); ARMS=('direct','equal','decay48')
SETTINGS=dict(iterations=600,depth=6,learning_rate=.05,l2_leaf_reg=20,loss_function='RMSE',thread_count=8,verbose=False,allow_writing_files=False,allow_const_label=True)

def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def info():
    return json.loads((DATA/'episodes.json').read_text()),json.loads((DATA/'features.json').read_text())
def predict(b,x):
    shape=x.shape[:-1]; z=np.asarray(x).reshape(-1,x.shape[-1]); out=z[:,b['cash_col']].astype(np.float64)+b['model'].predict(z,thread_count=2)*10000
    terminal=z[:,b['names'].index('cash.time_left')]==0; out[terminal]=z[terminal,b['cash_col']]
    return out.reshape(shape)
def fit_job(job):
    tag=job['tag']; path=RUN/'models'/(tag+'.joblib'); rec=RUN/'fits'/(tag+'.json')
    if path.exists() and rec.exists(): return json.loads(rec.read_text())|{'reused':True}
    start=time.monotonic(); rows,names=info(); xall=np.load(DATA/'real.npy',mmap_mode='r'); yall=np.load(DATA/'terminal.npy')
    freeze=json.loads((RUN/'FREEZE.json').read_text()); held=set(freeze['folds'][job['fold']]) if job.get('fold') is not None else set()
    ii=[i for i,r in enumerate(rows) if r['split']=='train' and r['episode_id'] not in held]; ids=[rows[i]['episode_id'] for i in ii]
    lo,hi=job.get('span',(0,719)); c=names.index('cash.own')
    # Index sequentially avoids copying the full memmap for every model.
    x=np.concatenate([np.asarray(xall[i,lo:hi]) for i in ii]).reshape(-1,len(names))
    if job['arm']=='direct': target=np.stack([np.broadcast_to(yall[i],(hi-lo,2)) for i in ii]).reshape(-1)
    else:
        labels=np.load(RUN/'labels'/f"{job['arm']}.npy",mmap_mode='r'); target=np.asarray(labels[ii,lo:hi]).reshape(-1)
    assert np.isfinite(target).all() and len(target)==len(x)
    settings=dict(SETTINGS)
    if 'span' in job: settings.update(iterations=500,depth=5)
    model=CatBoostRegressor(**settings,random_seed=job['seed']); model.fit(x,(target-x[:,c])/10000)
    b=dict(model=model,names=names,cash_col=c,training_episodes=ids,excluded_episodes=sorted(held),tag=tag,job=job,settings=settings,
        target='own terminal cash or frozen future-value mixture',rank_loss=False,terminal_exact=True)
    tmp=path.with_suffix('.tmp.joblib'); joblib.dump(b,tmp,compress=3); tmp.replace(path)
    r=dict(tag=tag,seconds=time.monotonic()-start,rows=len(x),games=len(ids),seed=job['seed'],span=[lo,hi],iterations=model.tree_count_,
        train_rmse_scaled=float(model.get_best_score()['learn']['RMSE']),target_deviation_mae=float(np.mean(abs(target-np.stack([np.broadcast_to(yall[i],(hi-lo,2)) for i in ii]).reshape(-1)))),
        model_sha=sha(path),settings=settings,learn_rmse_curve=model.get_evals_result()['learn']['RMSE'])
    write(rec,r); print(json.dumps(dict(fit=r)),flush=True)
    if held:
        jj=[i for i,r in enumerate(rows) if r['episode_id'] in held]; assert all(rows[i]['split']=='train' for i in jj)
        p=np.stack([predict(b,xall[i]) for i in jj]); np.savez_compressed(RUN/'oof'/f"fold{job['fold']}.npz",indices=np.array(jj),value=p)
    return r

def future_targets(p,y,half_life=None):
    """p [episodes,719,2]. All later nonterminal slots; O(T), no truncation."""
    target=np.broadcast_to(y[:,None,:],p.shape).copy(); acc=np.zeros_like(y); w=0.; decay=1. if half_life is None else 2**(-1/half_life)
    for t in range(717,-1,-1):
        acc=decay*(acc+p[:,t+1]); w=decay*(w+1)
        target[:,t]=.5*y+.5*acc/w
    return target

def freeze():
    RUN.mkdir(exist_ok=True)
    for sub in ('models','fits','oof','labels'): (RUN/sub).mkdir(exist_ok=True)
    rows,n=info(); ids=[r['episode_id'] for r in rows if r['split']=='train']; assert len(ids)==505
    order=sorted(ids,key=lambda e:hashlib.sha256(('full-clock-value-r1:'+e).encode()).hexdigest()); folds=[order[f::5] for f in range(5)]
    obj=dict(settings=SETTINGS,seeds=SEEDS,arms=ARMS,folds=folds,train_ids=ids,protocol_sha=sha(HERE/'PROTOCOL.md'),source_sha=sha(__file__),
        data_audit=json.loads((DATA/'ASSEMBLED.json').read_text()),teacher_iterations=1,teacher_seed=93027,terminal_beta=.5,half_life=48,
        rank_training=False,all_future_nonterminal_times=True,confirmation_reused=True,final11_read=False)
    path=RUN/'FREEZE.json'
    if path.exists(): assert json.loads(path.read_text())==obj
    else: write(path,obj)

def labels():
    rows,n=info(); y=np.load(DATA/'terminal.npy'); p=np.full((len(rows),720,2),np.nan)
    seen=set(); records=[]
    for f in range(5):
        b=joblib.load(RUN/'models'/f'teacher_fold{f}.joblib'); z=np.load(RUN/'oof'/f'fold{f}.npz'); jj=z['indices']; ids={rows[i]['episode_id'] for i in jj}
        assert ids==set(b['excluded_episodes']) and ids.isdisjoint(b['training_episodes']) and not seen&ids; seen|=ids
        assert b['job']['arm']=='direct'; p[jj]=z['value']; records.append(dict(fold=f,teacher_sha=sha(RUN/'models'/f'teacher_fold{f}.joblib'),annotated_episodes=len(ids),all_ancestors_excluded=True))
    ii=[i for i,r in enumerate(rows) if r['split']=='train']; assert len(seen)==505 and np.isfinite(p[ii]).all()
    np.testing.assert_array_equal(p[ii,719],y[ii]); np.save(RUN/'labels/oof_values.npy',p)
    stats={}
    for arm,half in [('equal',None),('decay48',48)]:
        a=np.full((len(rows),719,2),np.nan); a[ii]=future_targets(p[ii,:719],y[ii],half)
        # Verify streaming full-window sums against explicit weighted sums.
        for t in (0,300,696,717):
            w=np.ones(718-t) if half is None else np.exp2(-np.arange(1,719-t)/half)
            expected=.5*y[ii]+.5*np.einsum('etp,t->ep',p[ii,t+1:719],w/w.sum())
            np.testing.assert_allclose(a[ii,t],expected,rtol=1e-12,atol=1e-7)
        np.testing.assert_array_equal(a[ii,718],y[ii]); np.save(RUN/'labels'/f'{arm}.npy',a)
        stats[arm]=dict(mae_to_terminal=float(np.mean(abs(a[ii]-y[ii,None,:]))),sha=sha(RUN/'labels'/f'{arm}.npy'))
    write(RUN/'LABEL_AUDIT.json',dict(teachers=records,labels=stats,teachers_train_terminal_only=True,all_505_training_games_annotated_out_of_fold=True,
        eval_values_used_for_training=False,rank_loss=False,teacher_times_uncapped=True))

def jobs_run(jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(fit_job,j) for j in jobs]): f.result()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--jobs',type=int,default=2); a=ap.parse_args(); start=time.monotonic(); freeze()
    teachers=[dict(tag=f'teacher_fold{f}',arm='direct',seed=93027,fold=f) for f in range(5)]
    jobs_run(teachers,a.jobs); labels()
    jobs=[dict(tag=f'{arm}_s{seed}',arm=arm,seed=seed) for arm in ARMS for seed in SEEDS]
    jobs += [dict(tag=f'stage{j}_direct',arm='direct',seed=93027,span=(j*72,min((j+1)*72,719))) for j in range(10)]
    jobs_run(jobs,a.jobs)
    files={str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'models').glob('*.joblib'))}; assert len(files)==24
    write(RUN/'EVAL_FREEZE.json',files); write(RUN/'TRAIN_DONE.json',dict(seconds=time.monotonic()-start,models=len(files),no_rank_loss=True))
    print('TRAIN_DONE',flush=True)
if __name__=='__main__': main()
