"""Score native answers with the official MM-UPD option parser."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from mmupd_adapter import _native


def parse(rows, answers, native):
    by={answer['sid']:answer for answer in answers}
    assert len(by)==len(answers)==len(rows) and set(by)=={r['sid'] for r in rows}
    gt={(r['task'],r['family']):dict(r['display'])[r['gold']] for r in rows if r['type']=='standard'}
    correct,refused,routes=[],[],Counter()
    for row in rows:
        text=str(by[row['sid']]['response']).strip()
        choices={k:v for k,v in row['display'] if k in 'ABCDE'}
        pred=native['can_infer_option'](text,choices,question_type='option')
        route='official_direct'
        if pred:
            hit,ref=pred==row['gold'],pred==row['nota']
        else:
            task=row['task'].rsplit('_',1)[-1]
            _,valid,accepted=native['build_option_str_w_gt'](
                [choices[k] for k in 'ABCDE' if k in choices],gt[row['task'],row['family']],
                'standard' if row['type']=='standard' else task,'option',task)
            pred=native['can_infer_option'](text,choices,valid_option=valid)
            if pred:
                hit=pred==row['gold'] or (row['type']=='upd' and pred in accepted)
                ref,route=pred==row['nota'],'official_auxiliary_option'
            elif re.fullmatch(r'I cannot answer[.!]?\s*',text,flags=re.I):
                hit,ref,route=row['type']=='upd',True,'explicit_refusal_normalization'
            else:
                raise RuntimeError(f'Unresolved official extraction: {row["sid"]}: {text[:200]}')
        correct.append(bool(hit));refused.append(bool(ref));routes[route]+=1
    return np.array(correct),np.array(refused),dict(routes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--answers',type=Path,required=True)
    parser.add_argument('--upd-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    answers=[json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
    hit,refused,routes=parse(rows,answers,_native(str(args.upd_root)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,sid=np.array([r['sid'] for r in rows]),correct=hit,refused=refused)
    print(json.dumps(dict(rows=len(rows),routes=routes)))


if __name__=='__main__':main()
