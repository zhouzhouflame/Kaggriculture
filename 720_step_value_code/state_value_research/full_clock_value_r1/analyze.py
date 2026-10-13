"""Frozen-results synthesis, uncertainty/drift/support diagnostics; no fitting."""
import json,time,csv
from pathlib import Path
import numpy as np
from train import HERE,DATA,RUN,info,write
from evaluate import bootstrap,per_game,winner,quantile

def main():
    rows,names=info(); yall=np.load(DATA/'terminal.npy'); xall=np.load(DATA/'real.npy',mmap_mode='r'); c=names.index('cash.own')
    out=RUN/'evaluation'; result={}; arms=('direct','equal','decay48','stage_direct','cash'); train_names={name for r in rows if r['split']=='train' for name in r['names']}
    csv_rows=[]
    for split in ('development','confirmation'):
        ii=[i for i,r in enumerate(rows) if r['split']==split]; y=yall[ii]; z=dict(np.load(out/f'{split}_real.npz')); result[split]={}
        detail=json.loads((out/f'{split}_real.json').read_text())
        for arm,report in detail.items():
            csv_rows.extend(dict(split=split,model=arm,**r) for r in report['turns'])
        blocks={f'Day{lo+1}-{hi}':np.arange(lo*24,min(hi*24,719)) for lo,hi in [(0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21),(21,24),(24,27),(27,30)]}
        for title,times in blocks.items():
            result[split][title]={}
            base=winner(z['direct'][:,times],y)[1]
            for arm in arms:
                p=z[arm][:,times]; err=p-y[:,None,:]; correct,credit,ties=winner(p,y)
                # Common per-game/clock shift cancels in winner comparison;
                # gap residual measures side-differential errors separately.
                common=err.mean(axis=-1); relative=err[...,0]-err[...,1]
                result[split][title][arm]=dict(winner_accuracy=bootstrap(per_game(correct)),winner_tie_half_credit=bootstrap(per_game(credit)),
                    paired_credit_vs_direct=bootstrap(per_game(credit-base)),mae=float(abs(err).mean()),bias=float(err.mean()),
                    common_shift_mae=float(abs(common).mean()),relative_margin_error_mae=float(abs(relative).mean()))
        gap=abs(y[:,0]-y[:,1]); strata={'close_lt3000':gap<3000,'middle_3000_15000':(gap>=3000)&(gap<15000),'large_ge15000':gap>=15000,
            'some_unseen_name':np.array([any(n not in train_names for n in rows[i]['names']) for i in ii])}
        result[split]['strata']={}
        for key,mask in strata.items():
            if not mask.any(): continue
            result[split]['strata'][key]=dict(games=int(mask.sum()),models={arm:bootstrap(per_game(winner(z[arm][mask,:719],y[mask])[1])) for arm in arms})
        for kind in ('seven','novel','cross'):
            p=dict(np.load(out/f'{split}_{kind}.npz')); center=3 if kind=='seven' else 1
            result[split][kind+'_seed_consistency']={}
            for arm in ('direct','equal','decay48'):
                vals={str(seed):float((np.diff(p[f'{arm}_s{seed}'],axis=-1)>1e-8).mean()) for seed in (93027,93028,93029)}
                result[split][kind+'_seed_consistency'][arm]=vals
            result[split][kind+'_anchor_relative_magnitude']={arm:dict(delta_median=quantile(p[arm]-p[arm][...,center,None]),
                root_prediction_median=float(np.median(p[arm][...,center]))) for arm in arms}
    timings={}
    for p in (RUN/'fits').glob('*.json'):
        r=json.loads(p.read_text()); curve=r['learn_rmse_curve']; timings[p.stem]=dict(seconds=r['seconds'],rows=r['rows'],first_rmse=curve[0],last_rmse=curve[-1],
            last100_relative_drop=(curve[-101]-curve[-1])/max(curve[-101],1e-12) if len(curve)>100 else None)
    write(out/'DIAGNOSTICS.json',result); write(out/'TIMINGS.json',timings)
    with (out/'ALL_720_TURNS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps(dict(model_fit_seconds=sum(r['seconds'] for r in timings.values()),models=len(timings))),flush=True)
if __name__=='__main__': main()
