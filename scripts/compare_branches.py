"""Matched post-SFT abstention comparison and retention of newly repaired answers."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation import Paired, circular, load_reference
from x2_score import native_parser, score
from selector import select_threshold, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--upd-code-root', type=Path, required=True)
    ap.add_argument('--clipup', type=Path)
    a = ap.parse_args()
    args = a
    parser = native_parser(args.upd_code_root)
    base = load_reference(a.reference, 'qwen', 'E', parser, score)
    new = {split: load_reference(a.root, 'qwen', split, parser, score) for split in ('T', 'CM', 'E')}
    ev, cm = new['E'], new['CM']
    assert base['rows'] == ev['rows'], 'R0/R1 evaluation inputs differ'
    rows, std, ref = ev['rows'], ev['std'], ev['correct']
    paired = Paired(rows)
    fit = np.load(a.root / 'probe' / 'train_cv.npz')
    scores = dict(zip(fit['sid'].tolist(), fit['logits'].tolist()))
    ss = {split: np.array([scores[r['sid']] for r in d['rows']]) for split, d in new.items()}
    assert all(np.isfinite(x).all() for x in ss.values())
    predictions = {'R0': base['correct'], 'R1': ref}
    calibration, routed_answers = {}, {}
    raw = [json.loads(s) for s in (a.root / 'qwen_base/E/answers.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    for budget, name in [(0., 'R2_loss0'), (.005, 'R2_loss0.005')]:
        tau, _ = select_threshold(ss['T'], ss['CM'], cm['std'], cm['correct'], budget)
        keep = ss['E'] >= tau
        predictions[name] = np.where(keep, ref, ~std)
        calibration[name] = dict(budget=budget, threshold=tau if np.isfinite(tau) else str(tau),
            standard_count=int(cm['std'].sum()),
            lost_correct=int(((ss['CM'] < tau) & cm['std'] & cm['correct']).sum()))
        routed_answers[name] = route(rows, raw, ss['E'], tau)
    if a.clipup:
        answers = {x['sid']: x for x in
            (json.loads(s) for s in a.clipup.read_text(encoding='utf-8').splitlines() if s.strip())}
        assert set(answers) == {r['sid'] for r in rows}
        parsed = [score(r, answers[r['sid']], parser, 'option') for r in rows]
        predictions['R3'] = np.array([bool(p['correct']) if is_std else bool(p['abstain'])
                                     for p, is_std in zip(parsed, std)])

    rep = std & ~base['correct'] & ref
    old = std & base['correct'] & ref
    rep_clusters = len({r['cluster'] for r, take in zip(rows, rep) if take})
    arms = []
    for name, pred in predictions.items():
        delta = pred.astype(int) - ref.astype(int)
        delta0 = pred.astype(int) - base['correct'].astype(int)
        lost_new = int((rep & ~pred).sum())
        lost_old = int((old & ~pred).sum())
        gross = int((std & ref & ~pred).sum())
        repairs = int((std & ~ref & pred).sum())
        assert gross == lost_new + lost_old
        sci, uci = paired.interval(delta, std), paired.interval(delta, ~std)
        retained = int((rep & pred).sum())
        retention = dict(retained=retained, denominator=int(rep.sum()), repair_clusters=rep_clusters,
                         rate=retained / int(rep.sum()) if rep.any() else None, ci95=None)
        if rep_clusters >= 10:
            numerator = np.bincount(paired.ix, weights=(rep & pred), minlength=paired.k)
            denominator = np.bincount(paired.ix, weights=rep, minlength=paired.k)
            sampled_n, sampled_d = paired.w @ numerator, paired.w @ denominator
            valid = sampled_d > 0
            retention['ci95'] = np.quantile(sampled_n[valid] / sampled_d[valid], [.025, .975]).tolist()
            retention['bootstrap_nonempty_draws'] = int(valid.sum())
        arms.append(dict(arm=name, standard=float(pred[std].mean()), upd=float(pred[~std].mean()),
            circular=circular(rows, pred), gross_loss=gross, repairs=repairs,
            gross_loss_pp=100 * gross / int(std.sum()),
            new_repair_loss=lost_new, new_repair_loss_pp=100 * lost_new / int(std.sum()),
            old_correct_loss=lost_old, old_correct_loss_pp=100 * lost_old / int(std.sum()),
            new_repair_retention=retention, std_change_ci=sci, upd_change_ci=uci,
            passes_original_rule=bool(sci[1] > -.01 and uci[1] > 0) if name.startswith(('R2', 'R3')) else None,
            end_to_end_std_change_ci=paired.interval(delta0, std),
            end_to_end_upd_change_ci=paired.interval(delta0, ~std)))
    r0 = base['correct']
    result = dict(state='complete',
        counts=dict(rows=len(rows), standard=int(std.sum()), upd=int((~std).sum())),
        sft=dict(repairs=int(rep.sum()), loss=int((std & r0 & ~ref).sum()),
            net_correct_change=int((ref[std].astype(int) - r0[std].astype(int)).sum()),
            repair_clusters=rep_clusters),
        calibration=calibration, arms=arms,
        protocol=dict(reference='R1, the shared answering-SFT checkpoint',
            threshold_grid='T scores', threshold_constraint='CM gross loss relative to R1',
            bootstrap_draws=2000, bootstrap_seed=0, bootstrap_unit='image cluster',
            pass_standard_lower_bound=-.01, pass_upd_lower_bound=0.,
            evaluation='historically exposed E', training_seed=42))
    if 'R3' in predictions:
        diff = predictions['R2_loss0'].astype(int) - predictions['R3'].astype(int)
        result['R2_vs_R3'] = dict(standard=paired.interval(diff, std), upd=paired.interval(diff, ~std))
    out = a.root / 'comparison'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out / 'predictions.npz', sid=[r['sid'] for r in rows], **predictions)
    for name, answers in routed_answers.items():
        (out / f'{name}_answers.jsonl').write_text(
            ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in answers), encoding='utf-8')
    lines = ['# 作答 SFT 后的拒答比较', '',
        f"E：Standard {std.sum()} 条，UPD {(~std).sum()} 条。前置 SFT 修复 {rep.sum()} 条，"
        f"损失 {result['sft']['loss']} 条。", '',
        '| 分支 | Standard % | UPD % | Circular Dual % | 损失 / 修复（相对 R1） | 新修复保留 | 原标准 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arm in arms:
        retention = arm['new_repair_retention']
        retained = f"{retention['retained']}/{retention['denominator']}" if retention['denominator'] else 'N/A'
        passed = arm['passes_original_rule']
        lines.append(f"| {arm['arm']} | {100*arm['standard']:.4f} | {100*arm['upd']:.4f} | "
            f"{100*arm['circular']['macro']['dual']:.4f} | {arm['gross_loss']} / {arm['repairs']} | "
            f"{retained} | {'—' if passed is None else ('通过' if passed else '未通过')} |")
    lines += ['', '判据：相对 R1 的 ΔStandard 配对 95% CI 下界 > −1 pp，ΔUPD 下界 > 0。',
        '计数、完整区间、端到端变化和新修复损失拆分见 results.json。', '',
        '## 局限', '', '单种子、小规模任务适配；区间描述评价数据不确定性。E 为历史暴露总体。',
        '新修复集合不足 10 个图像簇时仅报告计数和点估计。']
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'state': result['state'], 'sft': result['sft'], 'arms': arms}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
