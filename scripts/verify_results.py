"""Recompute the main and frozen-confirmation results from released records."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'results'


def main():
    thresholds = json.loads((ROOT / 'frozen_thresholds.json').read_text(encoding='utf-8'))['calibration']
    main_results = json.loads((ROOT / 'shared_b1.json').read_text(encoding='utf-8'))
    ordinary = json.loads((ROOT / 'ordinary_sft.json').read_text(encoding='utf-8'))['arm']
    confirmation = json.loads((ROOT / 'independent_confirmation.json').read_text(encoding='utf-8'))
    mapping = {'R1':'B1','R2_loss0':'ReFrain_0','R2_loss0.005':'ReFrain_0.5','R3':'EmbLoRA',
               'CV_0':'ReFrain_0','CV_0.5':'ReFrain_0.5','ordinary_refusal_SFT':'ordinary_SFT'}
    for split in ('E','I'):
        with (ROOT / f'predictions_{split}.csv').open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
        std = np.array([r['type']=='standard' for r in rows])
        pred = {name:np.array([bool(int(r[name])) for r in rows])
                for name in ('B0','B1','ReFrain_0','ReFrain_0.5','EmbLoRA','ordinary_SFT')}
        gains = std & ~pred['B0'] & pred['B1']
        arms = [a for a in main_results['arms'] if a['arm'] != 'R0'] + [ordinary] if split=='E' else confirmation['arms']
        for arm in arms:
            key = arm.get('arm',arm.get('method'))
            name = mapping.get(key,key)
            value = pred[name]
            assert np.isclose(value[std].mean(),arm['standard'],rtol=0,atol=1e-12), (split,name,'Standard')
            assert np.isclose(value[~std].mean(),arm['upd'],rtol=0,atol=1e-12), (split,name,'UPD')
            assert int((std & pred['B1'] & ~value).sum()) == arm['gross_loss'], (split,name,'loss')
            assert int((std & ~pred['B1'] & value).sum()) == arm['repairs'], (split,name,'repair')
            expected_retention = arm['retained_repairs'] if name=='ordinary_SFT' and split=='E' else arm['new_repair_retention']['retained']
            assert int((gains & value).sum()) == expected_retention, (split,name,'retention')
        if split == 'I':
            scores = np.array([float(r['score']) for r in rows])
            for key,name in [('R2_loss0','ReFrain_0'),('R2_loss0.005','ReFrain_0.5')]:
                expected = np.where(scores >= float(thresholds[key]['threshold']),pred['B1'],~std)
                assert np.array_equal(expected,pred[name]), (split,name,'frozen routing')
        print(f'{split}: {len(rows)} examples; accuracy, loss, repair and gain retention match.')
    print('Independent confirmation: frozen routing matches on every example.')


if __name__ == '__main__':
    main()
