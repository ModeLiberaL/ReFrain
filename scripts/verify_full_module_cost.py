"""Recompute paired online cost and intervals from the released observations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from verify_scenario_records import close


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/full_module_cost'))
    args=parser.parse_args()
    expected=json.loads((args.root/'results.json').read_text())
    records=[json.loads(line) for line in (args.root/'measurements.jsonl').read_text().splitlines()]
    by=defaultdict(list)
    for record in records:by[record['sid']].append(record)
    assert len(by)==500 and all(sorted(r['repetition'] for r in g)==[0,1,2] for g in by.values())
    grouped=list(by.values())
    times={arm:np.array([np.mean([r[arm]['seconds'] for r in group]) for group in grouped]) for arm in ['B1','Full']}
    for arm,t in times.items():
        close([t.mean(),np.median(t),np.quantile(t,.95)],
              [expected['arms'][arm][k] for k in ['mean_seconds','median_seconds','p95_seconds']],arm)
    delta=times['Full']-times['B1']
    close(delta.mean(),expected['paired_increment_seconds'],'mean increment')
    close(delta.mean()/times['B1'].mean(),expected['relative_increment'],'relative increment')
    _,ix=np.unique(np.array([g[0]['group'] for g in grouped]),return_inverse=True)
    k=int(ix.max())+1
    draws=[];rng=np.random.default_rng(20261116)
    for _ in range(2000):
        w=rng.multinomial(k,np.full(k,1/k))[ix]
        draws.append([w @ delta/w.sum(),w @ delta/(w @ times['B1'])])
    ci=np.quantile(draws,[.025,.975],axis=0)
    close(ci[:,0],expected['increment_ci95_seconds'],'increment interval')
    close(ci[:,1],expected['relative_ci95'],'relative interval')
    assert all(r['check']['veto_equal_offline'] and r['check']['tokens_equal_research_cache'] for r in records)
    assert max(abs(r['check']['score_difference']) for r in records)<=1e-3
    print('PASS: 500 inputs, 1500 pairs; all mean/median/P95/increment estimates and paired intervals reproduced.')


if __name__=='__main__':main()
