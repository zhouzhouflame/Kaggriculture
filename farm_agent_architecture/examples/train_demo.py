"""训练任务链或出售 Q 表，并在独立种子上进行配对评估。"""
import argparse
import json
from pathlib import Path

from rl_core import MobileEnv, SaleEnv, evaluate, train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['workers','sales'],required=True)
    p.add_argument('--episodes',type=int)
    p.add_argument('--test-seeds',type=int,default=400)
    p.add_argument('--output',type=Path)
    p.add_argument('--policy-output',type=Path)
    args=p.parse_args()
    env=MobileEnv if args.mode=='workers' else SaleEnv
    episodes=args.episodes if args.episodes is not None else (18000 if args.mode=='workers' else 30000)
    if episodes<1 or args.test_seeds<1:
        p.error('episodes and test-seeds must be positive')
    seed=2026101201 if args.mode=='workers' else 2026101202
    q,training=train(env,episodes,seed)
    result={'mode':args.mode,'training':training,
            'evaluation':evaluate(env,q,list(range(10_000_000,10_000_000+args.test_seeds)))}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.policy_output:
        args.policy_output.parent.mkdir(parents=True,exist_ok=True)
        args.policy_output.write_text(json.dumps([{'state':list(s),'q':a} for s,a in q.items()],indent=2),encoding='utf-8')
    print(json.dumps({'training':training,'comparisons':result['evaluation']['comparisons']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
