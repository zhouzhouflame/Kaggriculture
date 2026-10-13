"""比较短周期访问、完整生命周期访问与终局运输条件下的作物合同。"""
import argparse
import json
from pathlib import Path
import ev_core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,help='official_engine.py；默认使用同级 720_step_value_code 中的固定引擎')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.engine:
        ev_core.ENGINE=args.engine.resolve()
    if not ev_core.ENGINE.is_file():
        p.error('请保留同级 720_step_value_code，或用 --engine 指定固定官方引擎源码')
    E=ev_core.engine_kernel()
    result=ev_core.ev_examples(E)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    concise={name:{'selected':row['winner'],'candidates':[
        {k:c[k] for k in ('crop','feasible','cash_net','score','reasons') if k in c}
        for c in row['contracts']]} for name,row in result.items()}
    print(json.dumps(concise,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
