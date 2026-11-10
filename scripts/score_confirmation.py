"""Score a single frozen confirmation set without fitting or threshold selection."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

from evaluation import Paired, circular
from x2_score import native_parser, score

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--upd-code-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    ROOT = root
    frozen = json.loads((ROOT/'frozen_models_thresholds.json').read_text(encoding='utf-8'))
    rows = read_rows(ROOT/'confirm.jsonl')
    std = np.array([r['type']=='standard' for r in rows])
    parser = native_parser(args.upd_code_root)
    predictions, raw_answers, invalid = {}, {}, {}
    for name in ('B1','EmbLoRA','ordinary_SFT'):
        folder = ROOT/name
        assert json.loads((folder/'status.json').read_text(encoding='utf-8'))['state'] == 'complete'
        answers = read_rows(folder/'answers.jsonl')
        assert [r['sid'] for r in answers] == [r['sid'] for r in rows]
        parsed = [score(r,a,parser,'option') for r,a in zip(rows,answers)]
        predictions[name] = np.array([bool(a['correct']) if s else bool(a['abstain']) for a,s in zip(parsed,std)])
        raw_answers[name] = answers
        invalid[name] = sum(a['invalid'] for a in parsed)
    weights = np.load(ROOT/'frozen_probe.npz')
    logits = np.load(ROOT/'B1/logits.npy', mmap_mode='r')
    assert len(logits)==len(rows)
    scores = np.empty(len(rows))
    for start in range(0,len(rows),128):
        scores[start:start+128] = np.asarray(logits[start:start+128],dtype=np.float64) @ weights['coef'] + float(weights['intercept'])
    assert np.isfinite(scores).all()
    for key,name in [('R2_loss0','CV_0'),('R2_loss0.005','CV_0.5')]:
        tau = frozen['calibration'][key]['threshold']
        keep = scores >= tau
        predictions[name] = np.where(keep,predictions['B1'],~std)
        routed = [dict(sid=r['sid'],response=a['response'] if kept else r['nota'], route='answer' if kept else 'nota')
                  for r,a,kept in zip(rows,raw_answers['B1'],keep)]
        (ROOT/(name+'_answers.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in routed))
    before = read_rows(ROOT/'B0_standard/answers.jsonl')
    srows = [r for r,s in zip(rows,std) if s]
    assert [r['sid'] for r in before] == [r['sid'] for r in srows]
    b0 = np.zeros(len(rows), dtype=bool)
    b0[std] = [bool(score(r,a,parser,'option')['correct']) for r,a in zip(srows,before)]
    base = predictions['B1']
    newly_repaired = std & ~b0 & base
    stable = std & b0 & base
    paired = Paired(rows)
    def interval(v,mask):
        return paired.interval(np.asarray(v,dtype=float),mask)
    def retention(pred):
        count = int(newly_repaired.sum())
        k = len({r['cluster'] for r,yes in zip(rows,newly_repaired) if yes})
        result = dict(retained=int((newly_repaired & pred).sum()),total=count,image_clusters=k,ci95=None)
        if k >= 10:
            n = np.bincount(paired.ix,weights=newly_repaired & pred,minlength=paired.k)
            d = np.bincount(paired.ix,weights=newly_repaired,minlength=paired.k)
            sn,sd = paired.w@n,paired.w@d
            result['ci95'] = np.quantile(sn[sd>0]/sd[sd>0],[.025,.975]).tolist()
        return result
    tasks = sorted({r['task'] for r in rows})
    task_masks = {task:np.array([r['task']==task for r in rows]) for task in tasks}
    arms = []
    for name in ('B1','CV_0','CV_0.5','EmbLoRA','ordinary_SFT'):
        pred = predictions[name]
        delta = pred.astype(int)-base.astype(int)
        sci,uci = interval(delta,std),interval(delta,~std)
        task_results = {}
        for task,m in task_masks.items():
            task_results[task] = dict(standard=float(pred[m&std].mean()),upd=float(pred[m&~std].mean()),
                standard_count=int((m&std).sum()),upd_count=int((m&~std).sum()),
                lost=int((m&std&base&~pred).sum()),repaired=int((m&std&~base&pred).sum()),
                standard_delta_ci=interval(delta,m&std),upd_delta_ci=interval(delta,m&~std))
        arms.append(dict(method=name,standard=float(pred[std].mean()),upd=float(pred[~std].mean()),
            macro_upd=float(np.mean([v['upd'] for v in task_results.values()])),
            circular=circular(rows,pred),gross_loss=int((std&base&~pred).sum()),
            gross_loss_rate=float((base&~pred)[std].mean()),repairs=int((std&~base&pred).sum()),
            new_repair_loss=int((newly_repaired&~pred).sum()),old_correct_loss=int((stable&~pred).sum()),
            new_repair_retention=retention(pred),standard_delta_ci=sci,upd_delta_ci=uci,
            passes_original_rule=None if name=='B1' else bool(sci[1]>-.01 and uci[1]>0),
            invalid=invalid.get(name),tasks=task_results))
    contrasts={}
    for ours in ('CV_0','CV_0.5'):
        for baseline in ('EmbLoRA','ordinary_SFT'):
            p,q=predictions[ours],predictions[baseline]
            delta=p.astype(int)-q.astype(int)
            loss_reduction=(base&~q).astype(int)-(base&~p).astype(int)
            contrasts[ours+'_minus_'+baseline] = dict(standard_ci=interval(delta,std),
                upd_ci=interval(delta,~std),gross_loss_reduction_ci=interval(loss_reduction,std),
                new_repair_retention_ci=interval(delta,newly_repaired) if len({r['cluster'] for r,take in zip(rows,newly_repaired) if take})>=10 else None)
    result=dict(state='complete',counts=dict(rows=len(rows),standard=int(std.sum()),upd=int((~std).sum()),image_clusters=paired.k),
        calibration=frozen['calibration'],arms=arms,contrasts=contrasts,
        answering_sft=dict(new_repairs=int(newly_repaired.sum()),losses=int((std&b0&~base).sum()),
            base_standard=float(b0[std].mean()),B1_standard=float(base[std].mean())),
        protocol=dict(frozen_models_thresholds='frozen_models_thresholds.json',
            independent_data='data_summary.json',bootstrap_draws=2000,bootstrap_unit='image cluster',bootstrap_seed=0,
            pass_standard_lower_bound=-.01,pass_upd_lower_bound=0.,
            scope='Single image-disjoint AI2D-derived confirmation; no fitting or recalibration'))
    (ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    np.savez_compressed(ROOT/'predictions.npz',sid=[r['sid'] for r in rows],scores=scores,B0_standard=b0,**predictions)
    lines=['# 冻结设置的独立确认结果','',
           '| 方法 | Standard | UPD | Circular Dual | 损失/修复 | 新修复保持 | 原判据 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for a in arms:
        keep=a['new_repair_retention'];passed=a['passes_original_rule']
        lines.append(f"| {a['method']} | {100*a['standard']:.4f}% | {100*a['upd']:.4f}% | {100*a['circular']['macro']['dual']:.4f}% | {a['gross_loss']}/{a['repairs']} | {keep['retained']}/{keep['total']} | {'参照' if passed is None else '通过' if passed else '未通过'} |")
    (ROOT/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(state='complete',counts=result['counts'],arms=arms,contrasts=contrasts),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
