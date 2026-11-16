"""Recompute matched-data SFT frontiers, calibration, ledgers and paired intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_scenario_records import load, order, curve, point, common, circular, close

ARMS=['R0','R1b','R2b','TA_gate']
CAPS=[0.,.0025,.005,.01]
NATIVE=['T-only SFT','Joint SFT 510','Joint SFT 1527','TA SFT']


def keep(root, split, reference):
    raw=load(root/f'{split}_letters.npz')
    lookup={s:i for i,s in enumerate(raw['sid'].tolist())}
    rows=json.loads((reference/f'{split}_questions.json').read_text())
    values=[]
    for row in rows:
        z=raw['logits'][lookup[row['sid']]].astype(np.float64)
        other=[ord(letter)-65 for letter,_ in row['display'] if letter!=row['nota']]
        values.append(z[other].max()-z[ord(row['nota'])-65])
    return np.array(values)


def ledger(std, base_hit, base_ref, hit, refused):
    original=std & base_hit
    wrong=std & ~base_hit & ~base_ref
    lost=original & ~hit
    return dict(original_correct=int(original.sum()),original_wrong_answers=int(wrong.sum()),
        loss=int(lost.sum()),repairs=int((std & ~base_hit & hit).sum()),
        loss_to_abstention=int((lost & refused).sum()),loss_to_wrong_answer=int((lost & ~refused).sum()),
        wrong_answers_withheld=int((wrong & refused).sum()),
        previously_refused_now_answered=int((~std & base_ref & ~refused).sum()),
        new_abstentions=int((~std & ~base_ref & refused).sum()),gain_pool=int((~std & ~base_ref).sum()),
        row_standard=float(hit[std].mean()),row_upd=float(hit[~std].mean()))


def verify(root, reference, expected, split, draws):
    d=load(root/f'{split}_records.npz')
    meta=load(reference/f'{split}_records.npz')
    assert np.array_equal(d['sid'],meta['sid'])
    d['correct']=d['s_plus'];d['TA_gate']=d['TA_gate_keep']
    close(keep(root,split,reference),d['TA_gate'],(split,'raw letter score'))
    n,pool=int(d['std'].sum()),int(d['gain'].sum())
    one=np.ones(len(d['sid']),dtype=int)
    orders={name:order(d[name]) for name in ARMS}
    curves={name:curve(d,orders[name],one) for name in ARMS}
    front={name:{str(cap):point(d,curves[name],one,cap) for cap in CAPS} for name in ARMS}
    masks={name+'/'+cap:d[name]<float(p['threshold']) for name in ARMS
           for cap,p in expected['frozen'][name].items()}
    predictions={'B1':d['base_hit']}
    refs={'B1':d['base_refused']}
    for name,mask in masks.items():
        public_name=name.replace('TA_gate','TA gate')
        predictions[public_name]=np.where(mask,~d['std'],d['base_hit'])
        refs[public_name]=d['base_refused']|mask
    for name in NATIVE:
        key=name.replace(' ','_').replace('+','')
        predictions[name]=d[key+'_hit'];refs[name]=d[key+'_refused']
    values,circ_samples,_=circular(meta,predictions,draws)
    for name,ref in expected['circular'].items():
        close(values[name],[ref[k] for k in ['standard','upd','dual']],(split,name,'circular'))
        if draws==2000:
            close(np.quantile(circ_samples[name][:,2],[.025,.975]),ref['circular_dual_ci95'],(split,name,'Dual CI'))
    for name,ref in expected['ledgers'].items():
        actual=ledger(d['std'],d['base_hit'],d['base_refused'],predictions[name],refs[name])
        for key,value in actual.items():close(value,ref[key],(split,name,key))
    for name in ARMS:
        for cap in CAPS:
            p,e=front[name][str(cap)],expected['frontier'][name][str(cap)]
            close([p['gain'],p['loss'],p['conversion_rate']],[e['gain'],e['loss'],e['rate']],(split,name,cap))
        for cap,e in expected['frozen'][name].items():
            mask=masks[name+'/'+cap]
            close([(mask & d['correct']).sum(),(mask & d['gain']).sum()],[e['loss'],e['gain']],(split,name,cap,'frozen'))
    pairs=[('R1b','TA_gate'),('R2b','TA_gate'),('R0','TA_gate')]
    samples={}
    for a,b in pairs:
        for cap in CAPS:samples[a,b,'realized_loss_ceiling',cap,'conversion_rate']=[]
        for protocol,metric in [('exact_common_loss','conversion_rate'),('frozen_deployment','loss_rate'),
                                ('frozen_deployment','conversion_rate')]:
            samples[a,b,protocol,.005,metric]=[]
    groups,ix=np.unique(d['groups'],return_inverse=True)
    weights=np.random.default_rng(20261116).multinomial(len(groups),np.full(len(groups),1/len(groups)),draws)
    for gw in weights:
        w=gw[ix];wn,wp=w @ d['std'],w @ d['gain']
        c={name:curve(d,orders[name],w) for name in ARMS}
        p={name:{cap:point(d,c[name],w,cap) for cap in CAPS} for name in ARMS}
        for a,b in pairs:
            for cap in CAPS:
                samples[a,b,'realized_loss_ceiling',cap,'conversion_rate'].append(p[a][cap]['conversion_rate']-p[b][cap]['conversion_rate'])
            samples[a,b,'exact_common_loss',.005,'conversion_rate'].append(common(c,a,b,np.floor(.005*wn+1e-12))['gain_difference']/wp)
            for field,metric,denom in [('correct','loss_rate',wn),('gain','conversion_rate',wp)]:
                samples[a,b,'frozen_deployment',.005,metric].append(
                    (w @ (d[field] & masks[a+'/0.005'])-w @ (d[field] & masks[b+'/0.005']))/denom)
    for p in expected['paired']:
        a,b,protocol,cap,metric=[p[k] for k in ['left','right','protocol','budget','metric']]
        if protocol=='circular':
            j=['standard','upd','dual'].index(metric.removeprefix('circular_'))
            actual=values[a][j]-values[b][j]
            sample=circ_samples[a][:,j]-circ_samples[b][:,j]
        else:
            sample=samples[a,b,protocol,cap,metric]
            if protocol=='realized_loss_ceiling':
                actual=front[a][str(cap)]['conversion_rate']-front[b][str(cap)]['conversion_rate']
            elif protocol=='exact_common_loss':
                value=common(curves,a,b,np.floor(cap*n+1e-12))
                close([value[k] for k in ['loss','left_gain','right_gain']],
                      [p[k] for k in ['loss','left_gain','right_gain']],(split,a,b,'exact counts'))
                actual=value['gain_difference']/pool
            else:
                field,denom=('correct',n) if metric=='loss_rate' else ('gain',pool)
                actual=((d[field] & masks[a+'/0.005']).sum()-(d[field] & masks[b+'/0.005']).sum())/denom
        close(actual,p['difference'],(split,a,b,protocol,metric))
        if draws==2000:close(np.quantile(sample,[.025,.975]),p['ci95'],(split,a,b,protocol,metric,'CI'))
    print('PASS',split,': raw scores, native ledgers, Circular, frontiers, frozen points and paired comparisons',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/matched_sft'))
    parser.add_argument('--reference',type=Path,default=Path('results/two_block/qwen'))
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    result=json.loads((args.root/'results.json').read_text())
    a,b=[keep(args.root,s,args.reference) for s in ['A','B']]
    labels=load(args.reference/'B_records.npz')
    candidates=np.r_[-np.inf,np.unique(a),np.inf]
    losses=np.searchsorted(np.sort(b[labels['correct']]),candidates,side='left')
    for cap in [0.,.005]:
        j=np.flatnonzero(losses<=cap*labels['std'].sum()+1e-12)[-1]
        p=result['gate_calibration'][str(cap)]
        close([candidates[j],losses[j]],[p['threshold'],p['loss']],('calibration',cap))
    for split in ['E','I']:verify(args.root,args.reference,result['splits'][split],split,args.bootstrap)


if __name__=='__main__':main()
