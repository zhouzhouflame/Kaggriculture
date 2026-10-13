"""核验样本校验值、特征标签对应关系、终局边界与示例可执行性。"""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'examples'))
from ev_core import FeedbackExecutor, ReservationLedger, engine_kernel, ev_examples
from rl_core import MobileEnv, SaleEnv, run_episode
from slice_snapshots import region_view


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    folder=HERE/'data/samples'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        blob=(folder/name).read_bytes()
        assert len(blob)==info['bytes'] and hashlib.sha256(blob).hexdigest()==info['sha256'],name
    observations=rows(folder/'observations.jsonl')
    features={r['sample_id']:r['values'] for r in rows(folder/'features.jsonl')}
    labels={r['sample_id']:r for r in rows(folder/'labels.jsonl')}
    names=json.loads((folder/'feature_names.json').read_text(encoding='utf-8'))
    configs=json.loads((folder/'configurations.json').read_text(encoding='utf-8'))
    assert len(observations)==manifest['sample_count']==48
    assert len(features)==len(labels)==48 and len(names)==483
    pair={}
    split={}
    for row in observations:
        m,o=row['metadata'],row['observation'];sid=m['sample_id']
        assert set(row)=={'metadata','observation'}
        assert set(o)=={'farms','market','town','day','hour','player','private','step'}
        assert 'seed' not in configs[m['episode_id']]
        assert m['step']==o['step']==o['day']*24+o['hour']
        assert m['seat']==o['player'] and m['day_display']==o['day']+1
        assert split.setdefault(m['episode_id'],m['split'])==m['split']
        money=o['farms'][o['player']]['money']
        vector=features[sid]
        assert len(vector)==483 and all(math.isfinite(x) for x in vector)
        assert vector[names.index('cash.own')]==money
        assert labels[sid]['future_net_gain']==labels[sid]['terminal_cash']-money
        if o['step']==719:
            assert labels[sid]['terminal_cash']==money
        group=(m['episode_id'],m['step'])
        public={k:o[k] for k in ('farms','market','town','day','hour','step')}
        assert pair.setdefault(group,public)==public
        view=region_view(o,'NE')
        assert len(view['cells'])==25 and view['row_range']==[1,5] and view['column_range']==[6,10]
    expected=json.loads((HERE/'data/local_experiment_summary.json').read_text(encoding='utf-8'))
    actual=ev_examples(engine_kernel())
    assert json.loads(json.dumps(actual))==expected['ev_contract_cases']
    for env in (MobileEnv,SaleEnv):
        for seed in range(20):
            r,_=run_episode(env,seed,q={},mode='rl')
            fields=('missed_required','unrealized') if env is MobileEnv else ('lost_units','missing_feed','unsold_units')
            assert all(r[k]==0 for k in fields)
    ledger=ReservationLedger()
    assert ledger.reserve('A','WHEAT',2,3) and not ledger.reserve('B','WHEAT',2,3)
    ex=FeedbackExecutor([['HARVEST'],['FEED'],['CARE']])
    tile={'animal':'SHEEP','yield_units':0,'fed_today':False,'cared_today':False}
    assert ex.next_action(tile,{'WHEAT':1},{})==['FEED']
    assert ex.tasks==[['FEED'],['CARE']]
    print(json.dumps({'sample_rows':len(observations),'feature_width':len(names),
                      'episodes':len(split),'paired_clocks':len(pair),'ev_cases':len(actual),
                      'rl_fallback_runs':40,'shared_material_and_task_continuity':'passed'},indent=2))


if __name__=='__main__':
    main()
