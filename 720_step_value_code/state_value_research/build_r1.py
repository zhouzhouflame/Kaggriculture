"""Development-only matched physical/pricing ablations. R0 sources unchanged."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from features import E, STEPS, state_features, reference_schedule, price_schedule
from mechanics import ROOT, ENGINE_SHA

HERE=Path(__file__).resolve().parent


def monetary_channels(prefix, obs, cfg, gross, cost):
    start=obs['day']*24+obs['hour'];last=cfg.get('episodeSteps',720)-1
    me=obs['player'];result={}
    for name,h in [('24',24),('48',48),('remaining',last-start)]:
        stop=min(last,start+h)
        for role,s in [('own',me),('opp',1-me)]:
            p=f'{prefix}.{name}.{role}.'
            result[p+'gross']=sum(v for t,v in gross[s].items() if start<=t<stop)
            result[p+'feed_cost']=sum(v for t,v in cost[s].items() if start<=t<stop)
            result[p+'net_feed']=result[p+'gross']-result[p+'feed_cost']
        result[f'{prefix}.{name}.margin_net_feed']=result[f'{prefix}.{name}.own.net_feed']-result[f'{prefix}.{name}.opp.net_feed']
    return result


def extended_features(obs,cfg):
    f=state_features(obs,cfg)
    schedules=[reference_schedule(obs,s,cfg) for s in (0,1)]
    gross,cost=price_schedule(obs,cfg,schedules)
    f.update(monetary_channels('income',obs,cfg,gross,cost))
    start=obs['day']*24+obs['hour'];last=cfg.get('episodeSteps',720)-1;me=obs['player']
    for name,h in [('24',24),('48',48),('remaining',last-start)]:
        stop=min(last,start+h)
        for role,s in [('own',me),('opp',1-me)]:
            f[f'income.{name}.{role}.service_actions']=sum(v for t,v in schedules[s][2].items() if start<=t<stop)
            f[f'feed.{name}.{role}.units']=sum(v for t,v in schedules[s][1].items() if start<=t<stop)
            for item in E.PRODUCTS:
                f[f'income.{name}.{role}.units.{item}']=sum(v.get(item,0) for t,v in schedules[s][0].items() if start<=t<stop)
        f[f'grossdiff.{name}']=f[f'income.{name}.own.gross']-f[f'income.{name}.opp.gross']
    f['income.opp_hidden_stock_known']=0
    frozen_gross=[Counter(),Counter()];frozen_cost=[Counter(),Counter()]
    for s in (0,1):
        for t,goods in schedules[s][0].items():
            frozen_gross[s][t]=sum(v*obs['market']['prices'][k] for k,v in goods.items())
        for t,n in schedules[s][1].items():frozen_cost[s][t]=n*obs['market']['prices']['WHEAT']
    f.update(monetary_channels('quote',obs,cfg,frozen_gross,frozen_cost))
    solo_gross=[Counter(),Counter()];solo_cost=[Counter(),Counter()]
    for s in (0,1):
        independent=[(defaultdict(Counter),Counter(),Counter()) for _ in (0,1)]
        independent[s]=schedules[s]
        g,c=price_schedule(obs,cfg,independent);solo_gross[s]=g[s];solo_cost[s]=c[s]
    f.update(monetary_channels('solo',obs,cfg,solo_gross,solo_cost))
    return f


def process(row):
    assert row['split'] in ('legacy','development','validation')
    assert not set(row['names'])&{'by','forever young','kwa'}
    with zipfile.ZipFile(ROOT/'expansion_research/corpus_r0/raw_replays.zip') as z:
        raw=z.read(row['member'])
    assert hashlib.sha256(raw).hexdigest()==row['sha256']
    replay=json.loads(raw);cfg={k:v for k,v in replay['configuration'].items() if k!='seed'}
    assert replay['module_version']=='1.32.7' and replay['statuses']==['DONE','DONE'] and len(replay['steps'])==720
    cash=[f['money'] for f in replay['steps'][-1][0]['observation']['farms']]
    assert list(map(float,replay['rewards']))==list(map(float,cash))
    rows=[]
    for t in STEPS:
        for seat in (0,1):
            f=extended_features(replay['steps'][t][seat]['observation'],cfg)
            margin=cash[seat]-cash[1-seat]
            rows.append({'f':f,'margin':margin,'win':int(margin>0),'tie':int(margin==0),
                         'meta':{'episode_id':row['episode_id'],'step':t,'day_display':t//24+1,
                                 'seat':seat,'names':row['names'],'split':row['split']}})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--jobs',type=int,default=8)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'expansion_research/corpus_r0/manifest.json').read_text(encoding='utf8'))
    entries=[r for r in manifest['episodes'] if r['split'] in ('legacy','development','validation')]
    assert len(entries)==140
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i,r in enumerate(pool.map(process,entries),1):
            rows.extend(r)
            if i%20==0:print(json.dumps({'episodes':i,'seconds':time.monotonic()-started}),flush=True)
    names=sorted(rows[0]['f']);assert all(sorted(r['f'])==names for r in rows)
    x=np.array([[r['f'][k] for k in names] for r in rows]);assert np.isfinite(x).all()
    meta=[r['meta'] for r in rows]
    # No evaluation of held-out labels: only compare development rows to the R0
    # matrix to ensure ablations did not quietly change the shared baseline.
    old_meta=json.loads((HERE/'dataset_r0/metadata.json').read_text(encoding='utf8'))
    old_names=json.loads((HERE/'dataset_r0/features.json').read_text())
    old=np.load(HERE/'dataset_r0/dataset.npz')
    old_index={(m['episode_id'],m['step'],m['seat']):i for i,m in enumerate(old_meta) if m['split']!='source_holdout'}
    old_rows=np.array([old_index[(m['episode_id'],m['step'],m['seat'])] for m in meta])
    cols=[names.index(k) for k in old_names]
    np.testing.assert_array_equal(x[:,cols],old['X'][old_rows])
    np.testing.assert_array_equal([r['margin'] for r in rows],old['margin'][old_rows])
    np.savez_compressed(out/'dataset.npz',X=x,margin=np.array([r['margin'] for r in rows]),
                        win=np.array([r['win'] for r in rows]),tie=np.array([r['tie'] for r in rows]))
    (out/'features.json').write_text(json.dumps(names,indent=2),encoding='utf8')
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False),encoding='utf8')
    audit={'episodes':140,'rows':len(rows),'features':len(names),'r0_development_parity':True,
           'source_holdout_raw_read':False,'final_test_raw_read':False,'engine_sha256':ENGINE_SHA,
           'seconds':time.monotonic()-started,'sources':{r['episode_id']:r['sha256'] for r in entries},
           'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
           'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL_R1.md').read_bytes()).hexdigest()}
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources','code_sha256')},indent=2),flush=True)


if __name__=='__main__':main()
