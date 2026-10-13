"""按日期、时刻、视角和区域查看快照切片。"""
import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT=Path(__file__).resolve().parents[1]/'data/samples/observations.jsonl'


def region_view(obs,region):
    farm=obs['farms'][obs['player']]
    size=len(farm['tiles']);mid=size//2
    bounds={'NW':(0,mid,0,mid),'NE':(mid,size,0,mid),
            'SW':(0,mid,mid,size),'SE':(mid,size,mid,size)}
    x0,x1,y0,y1=bounds[region]
    return {'region':region,'coordinate_system':'row,column both 1-based',
            'row_range':[y0+1,y1],'column_range':[x0+1,x1],
            'cells':[{'row':y+1,'column':x+1,'tile':farm['tiles'][y][x]}
                     for y in range(y0,y1) for x in range(x0,x1)],
            'context':{k:obs[k] for k in ('step','day','hour','player','market','town','private')},
            'workers':{'farmer':farm['farmer'],'hands':farm['hands']},
            'money':farm['money']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT)
    p.add_argument('--episode')
    p.add_argument('--day',type=int,help='展示日期，1..30')
    p.add_argument('--seat',type=int,choices=[0,1])
    p.add_argument('--start-step',type=int,default=0)
    p.add_argument('--end-step',type=int,default=719)
    p.add_argument('--region',choices=['NW','NE','SW','SE'])
    p.add_argument('--summary',action='store_true')
    args=p.parse_args()
    rows=[]
    with args.input.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);m=row['metadata']
            if args.episode and m['episode_id']!=args.episode:continue
            if args.day is not None and m['day_display']!=args.day:continue
            if args.seat is not None and m['seat']!=args.seat:continue
            if not args.start_step<=m['step']<=args.end_step:continue
            rows.append(row)
    if args.summary:
        out={'records':len(rows),'episodes':sorted({r['metadata']['episode_id'] for r in rows}),
             'steps':sorted({r['metadata']['step'] for r in rows}),
             'splits':dict(Counter(r['metadata']['split'] for r in rows)),
             'own_cash':[{'sample_id':r['metadata']['sample_id'],
                          'cash':r['observation']['farms'][r['metadata']['seat']]['money']} for r in rows]}
    elif args.region:
        out=[{'metadata':r['metadata'],'region_view':region_view(r['observation'],args.region)} for r in rows]
    else:
        out=rows
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
