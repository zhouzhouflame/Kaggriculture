"""Legal all-stage expected-shop features, same483 schema as historical model.

Current cohort reference is unchanged. Multiple unknown shops use fixed balanced
common scenarios, never actual future shops, game seed, identity or opponent policy.
"""
from pathlib import Path
from collections import Counter,defaultdict
import ctypes
import numpy as np
from features import E,state_features,reference_schedule
from build_r1 import monetary_channels

HERE=Path(__file__).resolve().parent
PRODUCTS=list(E.PRODUCTS);SHOPS=sorted(E.SHOPS);PI={k:i for i,k in enumerate(PRODUCTS)}
SHAPE={'linear':0,'sq':1,'sqrt':2,'log':3,'log10':4,'hinge':5}
LIB=None


def library():
    global LIB
    if LIB is None:
        LIB=ctypes.CDLL(str(HERE/'market_scenarios.so'))
        arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        LIB.prices.argtypes=[ctypes.c_int,ints,arr,arr,arr];LIB.prices.restype=None
        LIB.market.argtypes=[ctypes.c_int]*9+[arr,arr,ints,ints,ints,ints,ints,ints,arr,arr];LIB.market.restype=None
    return LIB


def parameters(obs):
    p=obs['market'].get('params') or E.MARKET_PARAMS
    return np.asarray([[p[k]['base'],p[k]['I0'],p[k]['T'],SHAPE[p[k]['below_func']],SHAPE[p[k]['above_func']],
        p[k]['below_target']*p[k]['base']/E._shape(p[k]['below_func'],p[k]['T'],p[k]['T']),
        p[k]['above_target']*p[k]['base']/E._shape(p[k]['above_func'],p[k]['T'],p[k]['T'])] for k in PRODUCTS],dtype=np.float64)


def scenarios(count=16):
    assert count%len(SHOPS)==0
    rng=np.random.default_rng(62931)
    return np.ascontiguousarray(np.stack([rng.permutation(np.tile(np.arange(len(SHOPS)),count//len(SHOPS))) for _ in range(8)],axis=1),dtype=np.int32)


def scenario_accounting(obs,cfg,schedules,count=16):
    start=obs['day']*24+obs['hour'];last=cfg.get('episodeSteps',720)-1;known=len(obs['town']['unlocked_shops'])
    draws=scenarios(count) if known<8 else np.zeros((1,8),np.int32);ns=len(draws)
    goods=np.zeros((last,2,len(PRODUCTS)),np.int32);feeds=np.zeros((last,2),np.int32)
    for s in (0,1):
        for t,items in schedules[s][0].items():
            if start<=t<last:
                for k,v in items.items():
                    if k in PI:goods[t,s,PI[k]]=v
        for t,v in schedules[s][1].items():
            if start<=t<last:feeds[t,s]=v
    sm=np.zeros((len(SHOPS),len(PRODUCTS)),np.int32)
    for j,k in enumerate(SHOPS):
        for product in E.SHOPS[k]:sm[j,PI[product]]=2 if len(E.SHOPS[k])==1 else 1
    current=np.sum([sm[SHOPS.index(k)] for k in obs['town']['unlocked_shops']],axis=0,dtype=np.int32) if known else np.zeros(len(PRODUCTS),np.int32)
    town=np.array([k in E.TOWN_CENTER_PRODUCTS for k in PRODUCTS],np.int32)
    inv=np.array([obs['market']['inventory'][k] for k in PRODUCTS],np.float64)
    gross=np.zeros((ns,2,last));cost=np.zeros_like(gross)
    library().market(start,last,ns,len(PRODUCTS),known,PI['WHEAT'],max(1,cfg.get('townShopUnlockInterval',3))*24,
        max(1,cfg.get('townShopSellInterval',4)),max(1,cfg.get('townCenterSellInterval',24)),inv,parameters(obs),goods,feeds,sm,current,draws,town,gross,cost)
    return gross,cost


def expected_schedule(obs,cfg,schedules,count=16):
    g,c=scenario_accounting(obs,cfg,schedules,count);g=g.mean(axis=0);c=c.mean(axis=0)
    return ([Counter({t:float(v) for t,v in enumerate(row) if v}) for row in g],
        [Counter({t:float(v) for t,v in enumerate(row) if v}) for row in c])


def stage_features(obs,cfg,count=16,schedules=None):
    f=state_features(obs,cfg)
    if schedules is None:schedules=[reference_schedule(obs,s,cfg) for s in (0,1)]
    gross,cost=expected_schedule(obs,cfg,schedules,count);f.update(monetary_channels('income',obs,cfg,gross,cost))
    start=obs['day']*24+obs['hour'];last=cfg.get('episodeSteps',720)-1;me=obs['player']
    for name,h in [('24',24),('48',48),('remaining',last-start)]:
        stop=min(last,start+h)
        for role,s in [('own',me),('opp',1-me)]:
            f[f'income.{name}.{role}.service_actions']=sum(v for t,v in schedules[s][2].items() if start<=t<stop)
            f[f'feed.{name}.{role}.units']=sum(v for t,v in schedules[s][1].items() if start<=t<stop)
            for item in E.PRODUCTS:f[f'income.{name}.{role}.units.{item}']=sum(v.get(item,0) for t,v in schedules[s][0].items() if start<=t<stop)
        f[f'grossdiff.{name}']=f[f'income.{name}.own.gross']-f[f'income.{name}.opp.gross']
    f['income.opp_hidden_stock_known']=0;g=[Counter(),Counter()];c=[Counter(),Counter()]
    for s in (0,1):
        for t,goods in schedules[s][0].items():g[s][t]=sum(v*obs['market']['prices'][k] for k,v in goods.items())
        for t,v in schedules[s][1].items():c[s][t]=v*obs['market']['prices']['WHEAT']
    f.update(monetary_channels('quote',obs,cfg,g,c));g=[Counter(),Counter()];c=[Counter(),Counter()]
    for s in (0,1):
        independent=[(defaultdict(Counter),Counter(),Counter()) for _ in (0,1)];independent[s]=schedules[s]
        gg,cc=expected_schedule(obs,cfg,independent,count);g[s]=gg[s];c[s]=cc[s]
    f.update(monetary_channels('solo',obs,cfg,g,c));return f
