"""Compact inspected-result export; no model or hyperparameter selection."""
import json
from train import HERE,RUN,write
ARMS=('direct','equal','decay48','stage_direct','cash')
def read(p): return json.loads(p.read_text())
def main():
    folder=RUN/'evaluation'; summary=read(folder/'SUMMARY.json'); diag=read(folder/'DIAGNOSTICS.json'); result={}
    for split in ('development','confirmation'):
        real=read(folder/f'{split}_real.json'); seven=read(folder/f'{split}_seven.json'); cross=read(folder/f'{split}_cross.json'); alternate=read(folder/f'{split}_novel.json')
        result[split]={}
        for arm in ARMS:
            result[split][arm]=dict(winner=summary[split][arm]['accuracy'],tie_credit=summary[split][arm]['tie_half_credit'],mae=summary[split][arm]['mae'],
                seven=seven['models'][arm]['adjacent'],seven_full=seven['models'][arm]['full'],alternate=alternate['models'][arm]['adjacent'],cross=cross['models'][arm]['adjacent'],
                cash_down=seven['models'][arm].get('cash_down_resource_change'),seven_scale8=seven['models'][arm]['scale']['8'],cross_scale24=cross['models'][arm]['scale']['24'])
        result[split]['paired_winner']=summary[split]['paired_vs_direct']; result[split]['paired_seven']=seven['paired_vs_direct']; result[split]['paired_cross']=cross['paired_vs_direct']
        result[split]['phases']={k:{arm:dict(winner=v[arm]['winner_tie_half_credit']['mean'],mae=v[arm]['mae']) for arm in ARMS} for k,v in diag[split].items() if k.startswith('Day')}
        result[split]['days']={str(d):{arm:dict(winner=real[arm]['days'][str(d)]['tie_half_credit']['mean'],
            seven=seven['days'][str(d)][arm]['adjacent']['mean'],alternate=alternate['days'][str(d)][arm]['adjacent']['mean'],
            cross=cross['days'].get(str(d),{}).get(arm,{}).get('adjacent',{}).get('mean')) for arm in ARMS} for d in (1,3,7,10,15,20,24,27,28,29,30)}
        result[split]['late_h0']={str(d):{arm:real[arm]['turns'][(d-1)*24] for arm in ARMS} for d in (27,28,29,30)}
        result[split]['seed_seven']=diag[split]['seven_seed_consistency']; result[split]['strata']=diag[split]['strata']
    result['local']=read(folder/'LOCAL_CONTROLS.json'); result['training']=read(RUN/'TRAIN_DONE.json'); result['aggregation']=read(RUN/'LABEL_TIMING.json')
    write(folder/'BRIEF.json',result)
    print(json.dumps({split:{arm:{k:v for k,v in row.items() if k in ('winner','mae','seven','alternate','cross')} for arm,row in result[split].items() if arm in ARMS} for split in ('development','confirmation')},indent=2))
if __name__=='__main__': main()
