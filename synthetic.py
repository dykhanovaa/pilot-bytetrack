"""Run with .venv/bin/python synthetic.py. Fail loudly on regression."""
from pathlib import Path
import json,numpy as np
from skip_aware import boxes,stock,SkipAware,state
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)

def run(mode,k,scenario):
    tracker=stock() if mode=='stock' else SkipAware()
    log=[]
    for f in range(1,151):
        gt=[[20,20,60,60,.9,0]] # A starts with frame 1
        if f>=12 and not (45<=f<=52) and not(85<=f<=125):
            gt.append([180,20,220,60,.9,0]) # B enters, short gap, long gap
        observed=(f-1)%k==0
        # Explicitly test a real observed empty frame as well.
        if scenario=='empty_observation' and 70<=f<=80:gt=[]
        inp=boxes(gt if observed else [])
        if mode=='stock': result=tracker.update(inp)
        else: result=tracker.step(inp if observed else None,observed=observed)
        log.append(dict(frame=f,observed=observed,input_type='Boxes' if mode=='stock' or observed else 'None (not observed)',
            input_xyxy_conf_cls=inp.data.tolist() if mode=='stock' or observed else None,
            tracks=state(tracker),output=result.tolist()))
    return log

def ids(log,start,end,x):
    return {int(r[4]) for row in log if start<=row['frame']<=end
            for r in row['output'] if abs(r[0]-x)<5}

def main():
    checks=[]; summary=[]
    for scenario in ['lifecycle','empty_observation']:
        for k in range(1,11):
            logs={m:run(m,k,scenario) for m in ['stock','skipaware']}
            for mode,log in logs.items():
                (OUT/f'synthetic_{scenario}_{mode}_k{k}.json').write_text(json.dumps(log,indent=2))
                early=ids(log,1,40,20);late=ids(log,12,44,180)
                short=ids(log,61,84,180);long=ids(log,126,150,180)
                summary.append(dict(scenario=scenario,mode=mode,k=k,A=sorted(early),
                    late_B=sorted(late),after_short_gap=sorted(short),after_long_gap=sorted(long)))
                if scenario=='lifecycle':
                    assert early, (mode,k,'A missing')
                    if mode=='stock' and k>1:assert not late,(mode,k,'expected upstream bug')
                    else: assert len(late)==1,(mode,k,'late track missing')
                    if mode=='skipaware':
                        assert short==late,(k,'short reappearance changed ID')
                        assert long and long.isdisjoint(late),(k,'long gap failed expiration')
                        assert all(all(any(t['id']==int(r[4]) and t['confirmed'] and t['state']=='tracked' for t in x['tracks']) for r in x['output']) for x in log)
            if k==1:
                assert logs['stock']==logs['skipaware'], 'k1 must be byte-for-byte identical'
            checks.append(f'PASS {scenario} k={k}: late entry, states, expiry, empty interface; k1 parity where applicable')
    # An observed empty Boxes must remove an unconfirmed track, unlike a skip.
    tr=SkipAware();tr.step(boxes([[20,20,60,60,.9,0]]),observed=True)
    tr.step(boxes([[20,20,60,60,.9,0],[180,20,220,60,.9,0]]),observed=True)
    pending=[t.track_id for t in tr.tracked_stracks if not t.is_activated]
    tr.step(boxes(),observed=True)
    assert pending and all(i in [t.track_id for t in tr.removed_stracks] for i in pending)
    checks.append('PASS true observed empty Boxes removes unconfirmed track')
    (OUT/'synthetic_summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'checks.log').write_text('\n'.join(checks)+'\n')
    print('\n'.join(checks))
if __name__=='__main__':main()
