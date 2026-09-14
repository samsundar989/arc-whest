"""Integer tariff and memory inventory for the specified mixed kernel.

Static upper model only. The active meter, peak memory and residual timing
remain separate acceptance obligations.
"""
from __future__ import annotations
import json
from pathlib import Path


def mm64(a,b,c):return 2*a*c*(2*b-1)


def sb32(a,b,c,depth=4,min_block=64):
    lo=min(a,b,c);d=0
    while d<depth and (lo>>(d+1))>=min_block:d+=1
    def rec(a,b,c,t):
        if t<=0 or a%2 or b%2 or c%2:return a*c*(2*b-1)
        return 7*rec(a//2,b//2,c//2,t-1)+3*(a*b+b*c)+4*a*c
    return rec(a,b,c,d)


def qr64(n,r):return 4*(2*n*r*r-(2*r*r*r)//3)


def inventory(n=1024,d=16,r=64,f21=928457637478):
    rows=[]
    replaced=0
    for l in range(1,d):
        k=l;m=k-1;final=l==d-1
        replaced+=2*sb32(n,n,k*n)
        if m:replaced+=sb32(n,n,m*n)
        if not final:replaced+=3*sb32(n,k*n,n)
        block={'layer':l,'all_D_transport':sb32(n,n,k*n),
               'young_A_transport':sb32(n,n,n),'old_Q_transport':mm64(n,n,r) if m else 0,
               'old_A_P_reconstruction':2*mm64(n,r,m*n) if m else 0,
               'old_D21_return':0 if final or not m else
                    2*mm64(n,m*n,r)+mm64(n,r,n)+sb32(n,m*n,n),
               'young_D21_return':0 if final else 3*sb32(n,n,n)}
        if not final:
            join=(4*mm64(n,n,r)+qr64(n,r)+2*mm64(r,n,n)+2*mm64(r,n,r))
            if m:
                join+=(2*mm64(r,n,r)+mm64(r,r,r)+mm64(n,r,r)
                       +2*mm64(r,r,m*n)+2*mm64(r,r,r))
            block['join']=join
        rows.append(block)
    backbone=f21-replaced
    products=sum(sum(v for key,v in row.items() if key!='layer') for row in rows)
    K=d*(d-1)//2
    envelope=512*(K*n*n+d*n*r+K*n*r+d*r*r)+128*d*n
    payload=4*(d-1)*n*n+4*n*n+8*n*r+16*(d-2)*n*r+8*r*r
    return {'n':n,'d':d,'r':r,'f21_counted':f21,'backbone':backbone,
            'blocks':rows,'products_qr':products,'pointwise_copy_cast_envelope':envelope,
            'total_upper':backbone+products+envelope,
            'retained_payload_bytes_peak_layer_formula':payload,
            'additional_transient_bytes':'two f64 old legs 16mn²; f64 D cast 8mn²; f32 PP 4mn²; old/new immutable D; Strassen recursive packed inputs/outputs and QR workspace must be measured'}


def main():
    out=Path(__file__).resolve().parents[1]/'experiments'/'e14_slice_a'
    out.mkdir(exist_ok=True)
    rows=[inventory(r=r) for r in (32,64,96,128)]
    (out/'static_cost_inventory.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'r':x['r'],'total_upper':x['total_upper']} for x in rows]))


if __name__=='__main__':main()
