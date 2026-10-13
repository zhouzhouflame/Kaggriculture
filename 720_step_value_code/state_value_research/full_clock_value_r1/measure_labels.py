"""Measure cached full-window aggregation separately from teacher fitting."""
import time,json
import numpy as np
from train import RUN,DATA,info,future_targets,write
def main():
    rows,_=info(); ii=[i for i,r in enumerate(rows) if r['split']=='train']
    start=time.perf_counter(); p=np.load(RUN/'labels/oof_values.npy')[ii,:719]; y=np.load(DATA/'terminal.npy')[ii]; load=time.perf_counter()-start
    result={}
    for arm,half in [('equal',None),('decay48',48)]:
        secs=[]
        for _ in range(5):
            t=time.perf_counter(); a=future_targets(p,y,half); secs.append(time.perf_counter()-t)
        np.testing.assert_array_equal(a,np.load(RUN/'labels'/f'{arm}.npy')[ii])
        result[arm]=dict(seconds=secs,median_seconds=float(np.median(secs)),rows=int(np.prod(a.shape)),future_times_uncapped=True)
    write(RUN/'LABEL_TIMING.json',dict(cache_load_seconds=load,aggregation=result,teacher_fitting_not_included=True))
    print(json.dumps(result),flush=True)
if __name__=='__main__': main()
