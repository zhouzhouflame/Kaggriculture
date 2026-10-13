"""Observed state coverage, initial symmetry, no model fitting."""
import json
import numpy as np
from train import DATA,info,write
def main():
    rows,names=info(); x=np.load(DATA/'real.npy',mmap_mode='r')
    wanted=['cash.own','cash.hour','cash.day','res.own.land','res.own.empty','res.own.workers','res.own.hires_today','res.own.worker_distance']
    cc=[names.index(n) for n in wanted]; result={}
    for split in ('train','development','confirmation'):
        ii=[i for i,r in enumerate(rows) if r['split']==split]; result[split]={}
        for d in range(30):
            z=np.stack([np.asarray(x[i,d*24:min((d+1)*24,719),:,cc]).reshape(len(cc),-1) for i in ii])
            result[split][str(d+1)]={n:dict(min=float(z[:,j].min()),max=float(z[:,j].max()),unique=int(len(np.unique(z[:,j])))) for j,n in enumerate(wanted)}
    write(DATA/'STATE_SUPPORT.json',result)
if __name__=='__main__': main()
