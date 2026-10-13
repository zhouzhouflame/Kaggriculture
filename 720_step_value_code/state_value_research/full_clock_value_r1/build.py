"""All real clocks plus held-only progress probes, resumable episode shards."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,zipfile,time,copy,argparse,hashlib,lzma
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent; PARENT=HERE.parent; sys.path.insert(0,str(PARENT))
import numpy as np
from features import reference_schedule,shed_distance,E
from features_fullstage import stage_features
from build_r1 import extended_features
SEVEN=(-8,-4,-2,0,2,4,8); THREE=(-2,0,2); CROSS=(-1,0,1)

def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load_names(): return json.loads((PARENT/'dataset_r10/real/features.json').read_text())
def read(row):
    with zipfile.ZipFile(HERE/'corpus/transport.zip') as z: p=json.loads(lzma.decompress(z.read(row['episode_id']+'.json.xz')))
    assert p['split']==row['split'] and len(p['snapshots'])==720
    return p

def with_inventory(base,o,seat):
    # Farm biology is identical with and without private inventory. Add only
    # the existing stock and carried delivery terms from reference_schedule.
    goods=defaultdict(Counter,{t:Counter(v) for t,v in base[0].items()})
    t=o['day']*24+o['hour']; farm=o['farms'][seat]; n=len(farm['tiles'])
    goods[t].update({k:v for k,v in o['private']['shed'].items() if k in E.PRODUCTS})
    for pos,inv in zip([farm['farmer']]+farm['hands'],o['private']['inventories']): goods[t+shed_distance(pos,n)+1].update({k:v for k,v in inv.items() if k in E.PRODUCTS})
    return goods,base[1],base[2]

def real_task(row):
    dest=HERE/'data/shards'/f"{row['episode_id']}.npz"
    if dest.exists(): return dict(episode=row['episode_id'],reused=True)
    started=time.monotonic(); p=read(row); cfg=p['configuration']; names=load_names()
    x=np.empty((720,2,len(names)),np.float32); parity=[]
    for t in range(720):
        pair=p['snapshots'][str(t)]; assert pair[0]['farms']==pair[1]['farms'] and pair[0]['market']==pair[1]['market'] and pair[0]['town']==pair[1]['town']
        public=pair[0]|{'player':-1}; base=[reference_schedule(public,s,cfg) for s in (0,1)]
        for s,o in enumerate(pair):
            sched=list(base); sched[s]=with_inventory(base[s],o,s)
            f=stage_features(o,cfg,16,sched); assert sorted(f)==names
            if t in (0,152,488,712,718):
                expected=stage_features(o,cfg,16)
                np.testing.assert_allclose([f[n] for n in names],[expected[n] for n in names],rtol=0,atol=1e-8)
                if t>=712:
                    old=extended_features(o,cfg); np.testing.assert_allclose([f[n] for n in names],[old[n] for n in names],rtol=0,atol=1e-8)
                parity.append((t,s))
            x[t,s]=[f[n] for n in names]
    assert np.isfinite(x).all(); y=np.array(p['EVALUATOR_ONLY']['terminal_cash'],np.float64)
    np.testing.assert_array_equal(x[719,:,names.index('cash.own')],y)
    tmp=dest.with_suffix('.tmp.npz'); np.savez_compressed(tmp,X=x,terminal_own=y); tmp.replace(dest)
    return dict(episode=row['episode_id'],seconds=time.monotonic()-started,parity_checks=len(parity),bytes=dest.stat().st_size)

def splice_rebased(root,source):
    assert root['player']==source['player']; s=root['player']; out=copy.deepcopy(root)
    out['farms'][s]=copy.deepcopy(source['farms'][s]); out['private']=copy.deepcopy(source['private'])
    dd=root['day']-source['day']; dt=dd*24
    # Rebase cross-day absolute ages; within-day progress uses original ages.
    if dd:
        for line in out['farms'][s]['tiles']:
            for tile in line:
                if not isinstance(tile,dict): continue
                for key in ('planted_day','placed_day'):
                    if key in tile:
                        age=source['day']-tile[key]; tile[key]+=dd
                        assert root['day']-tile[key]==age
                if tile.get('fertilized_until_day',-1)!=-1: tile['fertilized_until_day']+=dd
                if tile.get('max_lifespan_step',-1)!=-1:
                    left=tile['max_lifespan_step']-(source['day']*24+source['hour']); tile['max_lifespan_step']+=dt
                    assert tile['max_lifespan_step']-(root['day']*24+root['hour'])==left
    assert out['farms'][1-s]==root['farms'][1-s]
    return out

def probe_task(row):
    assert row['split']!='train'; dest=HERE/'data/probes'/f"{row['episode_id']}.npz"
    if dest.exists(): return dict(episode=row['episode_id'],reused=True)
    started=time.monotonic(); p=read(row); names=load_names(); cfg=p['configuration']; cache={}
    real=np.load(HERE/'data/shards'/f"{row['episode_id']}.npz")['X']
    fixed=[i for i,n in enumerate(names) if n in ('cash.day','cash.hour','cash.time_left','cash.opp') or n.startswith(('res.opp.','res.market.','res.shop.'))]
    def feature(t,u,s):
        if t==u: return real[t,s]
        key=t,u,s
        if key not in cache:
            o=splice_rebased(p['snapshots'][str(t)][s],p['snapshots'][str(u)][s]); f=stage_features(o,cfg,16)
            cache[key]=np.array([f[n] for n in names],np.float32)
            np.testing.assert_array_equal(cache[key][fixed],real[t,s,fixed])
        return cache[key]
    arrays={}
    for kind,hours,offsets,days,unit in [('seven',(8,12),SEVEN,range(30),1),('novel',(6,14),THREE,range(30),1),('cross',(8,16),CROSS,range(1,29),24)]:
        steps=[d*24+h for d in days for h in hours]
        arrays[kind]=np.array([[[feature(t,t+k*unit,s) for k in offsets] for s in (0,1)] for t in steps],np.float32)
        arrays[kind+'_steps']=np.array(steps); arrays[kind+'_offsets']=np.array(offsets)*unit
        np.testing.assert_array_equal(arrays[kind][:,:,offsets.index(0)],real[steps])
    assert all(np.isfinite(v).all() for v in arrays.values())
    tmp=dest.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**arrays); tmp.replace(dest)
    return dict(episode=row['episode_id'],seconds=time.monotonic()-started,bytes=dest.stat().st_size)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['real','probes','assemble']); ap.add_argument('--jobs',type=int,default=24); ap.add_argument('--limit',type=int); a=ap.parse_args()
    data=HERE/'data'; data.mkdir(exist_ok=True); (data/'shards').mkdir(exist_ok=True); (data/'probes').mkdir(exist_ok=True)
    rows=json.loads((HERE/'corpus/manifest.json').read_text())['episodes']; rows=sorted(rows,key=lambda r:r['episode_id']); names=load_names(); write(data/'features.json',names)
    config=dict(builder_sha=sha(__file__),features_sha=sha(PARENT/'features_fullstage.py'),biology_sha=sha(PARENT/'features.py'),market_sha=sha(PARENT/'market_scenarios.cpp'),corpus_manifest_sha=sha(HERE/'corpus/manifest.json'),scenarios=16)
    path=data/'CONFIG.json'
    if path.exists(): assert json.loads(path.read_text())==config
    else: write(path,config)
    if a.mode=='assemble':
        out=np.lib.format.open_memmap(data/'real.npy',mode='w+',dtype=np.float32,shape=(len(rows),720,2,len(names)))
        y=np.empty((len(rows),2),np.float64)
        for i,r in enumerate(rows):
            z=np.load(data/'shards'/f"{r['episode_id']}.npz"); out[i]=z['X']; y[i]=z['terminal_own']
        out.flush(); np.save(data/'terminal.npy',y); write(data/'episodes.json',rows)
        write(data/'ASSEMBLED.json',dict(shape=list(out.shape),dtype=str(out.dtype),episodes=len(rows),training_steps=719,terminal_step=719,feature_sha=sha(data/'features.json'),data_sha=sha(data/'real.npy'),labels_sha=sha(data/'terminal.npy')))
        print('ASSEMBLED',flush=True); return
    jobs=[r for r in rows if a.mode=='real' or r['split']!='train']
    if a.limit: jobs=[r for r in jobs if r['split']=='train'][:a.limit] if a.mode=='real' else jobs[:a.limit]
    start=time.monotonic(); records=[]; fn=real_task if a.mode=='real' else probe_task
    # Long tasks first for load balance; no outcome values used for ordering.
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for i,f in enumerate(as_completed([pool.submit(fn,r) for r in jobs]),1):
            records.append(f.result())
            if i%10==0 or i==len(jobs): print(json.dumps(dict(mode=a.mode,complete=i,total=len(jobs),seconds=time.monotonic()-start)),flush=True)
    write(data/(a.mode+('_pilot' if a.limit else '_DONE')+'.json'),dict(seconds=time.monotonic()-start,records=records,probes_training_use=False))
if __name__=='__main__': main()
