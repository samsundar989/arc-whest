"""Preserve earliest operational difference and its complete consumed operands."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def get(x):return np.asarray(x['values'],dtype=np.float64)


def main():
    out=ROOT/'experiments'/'e14_slice_a';rows=[]
    for label in ('b3','b4'):
        a=json.loads((out/f'{label}_f21_forced_control.json').read_text())
        b=json.loads((out/f'{label}_f21_forced_mixed.json').read_text())
        first=None
        for ar,br in zip(a['rows'],b['rows']):
            for key in ('(3,)','(2, 1)'):
                if key not in ar['trace']['before_pk']['S']:continue
                av=get(ar['trace']['before_pk']['S'][key]);bv=get(br['trace']['before_pk']['S'][key])
                dif=np.abs(av-bv)
                if np.any(dif):
                    ij=np.unravel_index(np.argmax(dif),dif.shape)
                    first={'layer':ar['layer'],'operand':'S'+key,'index':[int(x) for x in ij],
                           'reference_value':float(av[ij]),'mixed_value':float(bv[ij]),
                           'max_abs_difference':float(np.max(dif)),
                           'reference_full':av.tolist(),'mixed_full':bv.tolist()}
                    break
            if first:break
        am=np.asarray(a['final_mu']);bm=np.asarray(b['final_mu'])
        rows.append({'fixture':label,'first_consumed_difference':first,
                     'final_prediction_drift_energy':float(np.mean((am-bm)**2)),
                     'no_basis_exact_output':json.loads((out/'summary.json').read_text())[0 if label=='b3' else 1]['no_basis_exact_output']})
    (out/'first_operands.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k!='first_consumed_difference'} for row in rows]))


if __name__=='__main__':main()
