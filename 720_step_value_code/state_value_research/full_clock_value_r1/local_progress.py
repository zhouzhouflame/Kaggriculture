"""Exactly matched old Day30 h16..22 probes; evaluation only."""
import json
import numpy as np,joblib
from train import HERE,DATA,RUN,SEEDS,ARMS,info,predict,write
import evaluate as ev
def main():
    rows,names=info(); ev.NAMES=names
    ev.BUNDLES={p.stem:joblib.load(p) for p in (RUN/'models').glob('*.joblib') if not p.stem.startswith('teacher')}
    old=HERE.parent/'tiny_intraday_chain_r1'; data=np.load(old/'data/probes.npz'); meta=json.loads((old/'data/probes.metadata.json').read_text())
    pooled=[joblib.load(old/f'run0/{s}/pooled.joblib') for s in SEEDS]; result={}
    for split in ('development','confirmation'):
        ii=[i for i,r in enumerate(meta) if r['split']==split]; ids={str(meta[i]['episode_id']) for i in ii}; assert len(ids)==168
        x=data['X'][ii]; values=ev.infer_all(x)
        values['legacy_pooled16_22']=np.mean([predict(b,x) for b in pooled],axis=0)
        pv=[]
        for j,h in enumerate(range(16,23)):
            bundles=[joblib.load(old/f'run0/{s}/h{h}_direct_excludenone.joblib') for s in SEEDS]
            assert all(ids.isdisjoint(b['training_episodes']) for b in bundles+pooled)
            pv.append(np.mean([predict(b,x[:,j]) for b in bundles],axis=0))
        values['legacy_per_hour']=np.stack(pv,axis=1); result[split]={}
        for arm,v in values.items():
            if '_s' in arm: continue
            scores=np.diff(v,axis=-1)>1e-8; game=scores.reshape(168,2,7,2).mean((1,2,3))
            legacy_score=(np.diff(values['legacy_pooled16_22'],axis=-1)>1e-8).reshape(168,2,7,2).mean((1,2,3))
            delta=v[:,0,2]-v[:,0,1]; ref=data['terminal_own'][ii]*2/720
            result[split][arm]=dict(adjacent=ev.bootstrap(game),paired_vs_legacy_pooled=ev.bootstrap(game-legacy_score),
                h16_plus2_delta=ev.quantile(delta),h16_plus2_ratio=ev.quantile(delta/ref),
                per_hour={str(h):float(scores[:,j].mean()) for j,h in enumerate(range(16,23))})
        np.savez_compressed(RUN/f'evaluation/{split}_matched_late_probes.npz',**values)
    write(RUN/'evaluation/MATCHED_LATE_PROGRESS.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
