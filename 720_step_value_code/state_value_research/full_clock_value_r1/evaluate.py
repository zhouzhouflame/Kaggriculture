"""All720 real states and evaluation-only temporal transplants."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json,time,sys
from pathlib import Path
import numpy as np,joblib
from train import HERE,DATA,RUN,SEEDS,ARMS,info,predict,sha,write
NAMES=None; BUNDLES=None

def bootstrap(v):
    """One value per episode, clustered even if averaged over seats/times."""
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if not len(v): return None
    rng=np.random.default_rng(9373); b=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),episodes=len(v))
def per_game(v): return np.asarray(v).reshape(len(v),-1).mean(1)
def quantile(v):
    v=np.asarray(v); v=v[np.isfinite(v)]
    return dict(zip(('p05','median','p95'),map(float,np.quantile(v,[.05,.5,.95])))) if len(v) else None
def infer_all(x):
    out={}
    for arm in ARMS:
        ps=[]
        for seed in SEEDS:
            v=predict(BUNDLES[f'{arm}_s{seed}'],x); out[f'{arm}_s{seed}']=v; ps.append(v)
        out[arm]=np.mean(ps,axis=0)
    shape=x.shape[:-1]; flat=x.reshape(-1,x.shape[-1]); stage=np.minimum(flat[:,NAMES.index('cash.day')].astype(int)//3,9); v=np.empty(len(flat))
    for k in np.unique(stage):
        mask=stage==k; v[mask]=predict(BUNDLES[f'stage{k}_direct'],flat[mask])
    out['stage_direct']=v.reshape(shape); out['cash']=x[...,NAMES.index('cash.own')].astype(np.float64)
    return out

def winner(v,y):
    diff=v[...,0]-v[...,1]; true=np.sign(y[:,0]-y[:,1]); true=true.reshape((len(y),)+(1,)*(diff.ndim-1))
    sign=np.sign(diff); strict=sign==true; ties=(sign==0)&(true!=0)
    return strict,strict.astype(float)+.5*ties,ties
def real_report(v,y):
    assert np.isfinite(v).all()
    truth=y[:,None,:]; err=v-truth; correct,credit,tie=winner(v,y)
    perturn=[]
    for t in range(720):
        perturn.append(dict(step=t,day=t//24+1,hour=t%24,correct=int(correct[:,t].sum()),games=len(y),accuracy=float(correct[:,t].mean()),
            tie_half_credit=float(credit[:,t].mean()),predicted_ties=int(tie[:,t].sum()),mae=float(abs(err[:,t]).mean()),bias=float(err[:,t].mean()),negative_values=int((v[:,t]<0).sum())))
    days={}
    for d in range(30):
        sl=slice(d*24,min((d+1)*24,719))
        days[str(d+1)]=dict(accuracy=bootstrap(per_game(correct[:,sl])),tie_half_credit=bootstrap(per_game(credit[:,sl])),
            mae=bootstrap(per_game(abs(err[:,sl]))),bias=bootstrap(per_game(err[:,sl])),predicted_tie_rate=float(tie[:,sl].mean()))
    return dict(all_nonterminal=dict(accuracy=bootstrap(per_game(correct[:,:719])),tie_half_credit=bootstrap(per_game(credit[:,:719])),
        mae=bootstrap(per_game(abs(err[:,:719]))),bias=bootstrap(per_game(err[:,:719]))),days=days,turns=perturn,
        terminal_exact=bool(np.array_equal(v[:,719],y)))

def probe_report(v,cash,terminal,offsets,identical,resource_change=None):
    # value [episode,root,seat,slot]. Strict order is a proxy, not ground truth.
    dv=np.diff(v,axis=-1); dc=np.diff(cash,axis=-1); positive=dv>1e-8; ties=abs(dv)<=1e-8
    root=list(offsets).index(0); perday=[]
    r=dict(adjacent=bootstrap(per_game(positive)),full=bootstrap(per_game(np.all(positive,axis=-1))),tie_rate=float(ties.mean()),
        half_credit_ties=bootstrap(per_game(positive+.5*ties)),feature_identical_pairs=int(identical.sum()),pairs=int(identical.size),scale={})
    changed=~identical
    r['nonidentical_positive']=float(positive[changed].mean()) if changed.any() else None
    strata=[('cash_down',dc<0),('cash_flat',dc==0),('cash_up',dc>0)]
    if resource_change is not None: strata.append(('cash_down_resource_change',(dc<0)&resource_change))
    for name,mask in strata:
        if mask.any():
            count=mask.reshape(len(mask),-1).sum(1); numer=(positive&mask).reshape(len(mask),-1).sum(1)
            r[name]=dict(pairs=int(mask.sum()),positive=float(positive[mask].mean()),delta=quantile(dv[mask]),
                positive_per_episode=bootstrap(numer[count>0]/count[count>0]))
    for j,k in enumerate(offsets):
        if k==0: continue
        delta=v[...,j]-v[...,root]; ref=terminal[:,None,:]*abs(k)/720
        ratio=np.divide(np.sign(k)*delta,ref,out=np.full_like(delta,np.nan),where=ref>0)
        r['scale'][str(k)]=dict(delta=quantile(delta),oriented_ratio=quantile(ratio),correct_direction=bootstrap(per_game(delta*np.sign(k)>1e-8)))
    return r

def main():
    global NAMES,BUNDLES
    start=time.monotonic(); rows,NAMES=info(); freeze=json.loads((RUN/'EVAL_FREEZE.json').read_text()); assert all(sha(RUN/k)==v for k,v in freeze.items())
    BUNDLES={p.stem:joblib.load(p) for p in (RUN/'models').glob('*.joblib') if not p.stem.startswith('teacher')}
    xall=np.load(DATA/'real.npy',mmap_mode='r'); yall=np.load(DATA/'terminal.npy'); out=RUN/'evaluation'; out.mkdir(exist_ok=True)
    summary={}; integrity={}; local={}
    for split in ('development','confirmation'):
        ii=[i for i,r in enumerate(rows) if r['split']==split]; ids={rows[i]['episode_id'] for i in ii}; assert len(ids)==168
        assert all(ids.isdisjoint(b['training_episodes']) for b in BUNDLES.values())
        x=np.asarray(xall[ii]); y=yall[ii]; path=out/f'{split}_real.npz'
        if path.exists(): pred=dict(np.load(path))
        else: pred=infer_all(x); np.savez_compressed(path,**pred)
        reports={k:real_report(v,y) for k,v in pred.items()}; write(out/f'{split}_real.json',reports)
        summary[split]={k:r['all_nonterminal'] for k,r in reports.items()}; summary[split]['paired_vs_direct']={}
        base=winner(pred['direct'],y)[1][:,:719]
        for arm in (*ARMS[1:],'stage_direct','cash'):
            delta=winner(pred[arm],y)[1][:,:719]-base; summary[split]['paired_vs_direct'][arm]=bootstrap(per_game(delta))
        print(json.dumps(dict(evaluated=split,real=True,seconds=time.monotonic()-start)),flush=True)
        # Three probe families, loaded one at a time to bound memory.
        prows=[]
        for i in ii:
            p=DATA/'probes'/f"{rows[i]['episode_id']}.npz"; assert p.exists(); prows.append(p)
        for kind in ('seven','novel','cross'):
            px=[]
            for p in prows:
                with np.load(p) as z: px.append(z[kind]); steps=z[kind+'_steps']; offsets=z[kind+'_offsets']
            px=np.array(px); center=list(offsets).index(0); np.testing.assert_array_equal(px[:,:,:,center],x[:,steps])
            identical=np.all(np.diff(px,axis=-2)==0,axis=-1)
            own_cols=[j for j,n in enumerate(NAMES) if n.startswith('res.own.')]
            resource_change=np.any(np.diff(px[...,own_cols],axis=-2)!=0,axis=-1)
            cash=px[...,NAMES.index('cash.own')]; dest=out/f'{split}_{kind}.npz'
            if dest.exists(): pp=dict(np.load(dest))
            else: pp=infer_all(px); np.savez_compressed(dest,**pp)
            results={k:probe_report(v,cash,y,offsets,identical,resource_change) for k,v in pp.items()}; byday={}
            for day in sorted(set(steps//24+1)):
                jj=np.where(steps//24+1==day)[0]
                byday[str(day)]={k:probe_report(v[:,jj],cash[:,jj],y,offsets,identical[:,jj],resource_change[:,jj]) for k,v in pp.items() if '_s' not in k}
            paired={k:bootstrap(per_game(np.mean(np.diff(v,axis=-1)>1e-8,axis=-1)-np.mean(np.diff(pp['direct'],axis=-1)>1e-8,axis=-1))) for k,v in pp.items() if '_s' not in k and k!='direct'}
            write(out/f'{split}_{kind}.json',dict(models=results,days=byday,paired_vs_direct=paired,steps=steps.tolist(),offsets=offsets.tolist()))
            summary[split][kind]={k:dict(adjacent=v['adjacent'],full=v['full'],tie_rate=v['tie_rate'],nonidentical_positive=v['nonidentical_positive']) for k,v in results.items()}
            summary[split][kind+'_paired']=paired
            if kind=='cross':
                actual={k:np.stack([v[:,steps+int(o)] for o in offsets],axis=-1) for k,v in pred.items()}
                np.savez_compressed(out/f'{split}_cross_actual_time.npz',**actual)
                write(out/f'{split}_cross_actual_time.json',{k:dict(adjacent=bootstrap(per_game(np.diff(v,axis=-1)>1e-8)),delta=quantile(np.diff(v,axis=-1))) for k,v in actual.items()})
            print(json.dumps(dict(evaluated=split,probe=kind,seconds=time.monotonic()-start)),flush=True)
            del px,pp
        local[split]={}
        for h in range(16,23):
            point=696+h; legacy=[]; legacy_pooled=[]
            for seed in SEEDS:
                b=joblib.load(HERE.parent/f'tiny_intraday_chain_r1/run0/{seed}/h{h}_direct_excludenone.joblib'); assert ids.isdisjoint(b['training_episodes'])
                legacy.append(predict(b,x[:,point]))
                pooled=joblib.load(HERE.parent/f'tiny_intraday_chain_r1/run0/{seed}/pooled.joblib'); assert ids.isdisjoint(pooled['training_episodes'])
                legacy_pooled.append(predict(pooled,x[:,point]))
            p=np.mean(legacy,axis=0); truth=np.sign(y[:,0]-y[:,1]); v={k:v[:,point] for k,v in pred.items() if '_s' not in k}; v['legacy_per_hour']=p
            v['legacy_pooled16_22']=np.mean(legacy_pooled,axis=0)
            local[split][str(h)]={k:dict(correct=int((np.sign(a[:,0]-a[:,1])==truth).sum()),games=len(y),mae=float(abs(a-y).mean())) for k,a in v.items()}
        integrity[split]=dict(episodes=len(ii),steps=720,model_training_disjoint=True,zero_offset_exact=True,terminal_exact={k:r['terminal_exact'] for k,r in reports.items()},
            feature_identical_seat_states_by_turn=np.all(x[:,:,0]==x[:,:,1],axis=-1).sum(axis=0).tolist())
        write(out/f'{split}_episodes.json',[rows[i] for i in ii])
    write(out/'SUMMARY.json',summary); write(out/'LOCAL_CONTROLS.json',local); write(out/'INTEGRITY.json',integrity)
    assert all(sha(RUN/k)==v for k,v in freeze.items())
    write(out/'DONE.json',dict(seconds=time.monotonic()-start,real_steps=720,learned_nonterminal_steps=719,models=len(freeze),all_model_hashes_unchanged=True))
    print('EVAL_DONE',flush=True)
if __name__=='__main__': main()
