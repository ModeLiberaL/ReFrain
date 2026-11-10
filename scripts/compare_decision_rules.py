"""Report default logit thresholds and empirical calibration losses on B1."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from evaluation import load_reference,Paired,circular
from x2_score import native_parser,score
ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
r = root
out=r/'comparison'
parser=native_parser(args.upd_code_root)
d={s:load_reference(r,'qwen',s,parser,score) for s in ['CM','E']}
ev,cm=d['E'],d['CM']
base,std=ev['correct'],ev['std']
orig=np.load(out/'predictions.npz')
assert orig['sid'].tolist()==[x['sid'] for x in ev['rows']]
rep=std&~orig['R0']&base
paired=Paired(ev['rows'])
arms=[]
for head in ['published_C1','train_cv']:
 h=np.load(r/'probe'/f'{head}.npz')
 values=dict(zip(h['sid'].tolist(),h['logits'].tolist()))
 sc={s:np.array([values[x['sid']] for x in v['rows']]) for s,v in d.items()}
 pred=np.where(sc['E']>=0,base,~std)
 delta=pred.astype(int)-base.astype(int)
 sci,uci=paired.interval(delta,std),paired.interval(delta,~std)
 arms.append(dict(method='logits_'+head,threshold=0,probability_threshold=.5,
  standard=float(pred[std].mean()),upd=float(pred[~std].mean()),
  circular_dual=circular(ev['rows'],pred)['macro']['dual'],
  gross_loss=int((std&base&~pred).sum()),repairs=0,retained_repairs=int((rep&pred).sum()),repair_total=int(rep.sum()),
  calibration_loss=int(((sc['CM']<0)&cm['std']&cm['correct']).sum()),
  standard_change_ci=sci,upd_change_ci=uci,passes_original_rule=bool(sci[1]>-.01 and uci[1]>0)))
ncm,ne=int(cm['std'].sum()),int(std.sum())
loss_table=[]
main=json.loads((out/'results.json').read_text(encoding='utf-8'))
for name,v in main['calibration'].items():
 arm=next(a for a in main['arms'] if a['arm']==name)
 loss_table.append(dict(method='logits_CV',budget=v['budget'],cm_loss=v['lost_correct'],e_loss=arm['gross_loss']))
controls=json.loads((out/'cached_controls/results.json').read_text(encoding='utf-8'))
for a in controls['arms']:
 loss_table.append(dict(method=a['method'],budget=a['budget'],cm_loss=round(a['calibration_loss']*ncm),e_loss=a['gross_loss']))
clip=json.loads((out/'clip_classifier/results.json').read_text(encoding='utf-8'))
for a in clip['arms']:
 if a['budget'] is not None:
  loss_table.append(dict(method='CLIP_'+a['head'],budget=a['budget'],cm_loss=round(a['calibration_loss']*ncm),e_loss=a['gross_loss']))
for a in loss_table:
 a.update(cm_standard=ncm,e_standard=ne,cm_loss_rate=a['cm_loss']/ncm,e_loss_rate=a['e_loss']/ne)
result=dict(status='complete',default_logit_arms=arms,calibration_loss_table=loss_table)
(out/'decision_rule_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
