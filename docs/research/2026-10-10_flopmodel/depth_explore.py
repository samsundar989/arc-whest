import json
from sbmodel import sb_cost
U = 2*1024**3 - 1024**2
MM = json.load(open('estimator_k3_v81d_starall2.py.mm.json'))
def cost(m,k,n,d,blk=True):
    return sb_cost(1,m,k,n,d,blk)
tot5=0; tot6=0; dops=0; gain_by=dict()
rows=[]
for x in MM:
    m,k = x['A']; n = x['B'][1]; d = x['depth']
    c0,o0 = cost(m,k,n,d,True)
    best=(c0,o0,d)
    for dd in range(d+1, 8):
        c,o = cost(m,k,n,dd,True)
        if c < best[0]: best=(c,o,dd)
    rows.append((x['caller'],x['line'],x['layer'],m,k,n,d,c0,o0)+best)
# summarize: savings if each product used its FLOP-optimal depth (block output)
s = sum(r[7]-r[9] for r in rows); o = sum(r[10]-r[8] for r in rows)
print("FLOP-optimal per-shape depth (block glue): saving %.3f u (%.3f%% of F), ops %+d" % (s/U, 100*s/387664679697, o))
from collections import defaultdict
g=defaultdict(lambda:[0,0,0])
for r in rows:
    g[(r[0],r[1])][0]+=r[7]-r[9]; g[(r[0],r[1])][1]+=r[10]-r[8]; g[(r[0],r[1])][2]+=1
for k,v in sorted(g.items(), key=lambda kv:-kv[1][0]):
    if v[0]>0: print("  %-22s %5d save %.3f u ops %+d (n=%d)" % (k[0],k[1],v[0]/U,v[1],v[2]))
# leaf sizes at optimal
print("examples:")
for r in rows:
    if r[0]=='_shared_contract_lean' and r[1]==763 and r[2] in (5,14):
        print(r[:7], "d5 cost %.4f u -> d%d %.4f u" % (r[7]/U, r[11], r[9]/U))
    if r[0]=='_advance' and r[1]==1191 and r[2] in (5,14):
        print(r[:7], "d5 cost %.4f u -> d%d %.4f u" % (r[7]/U, r[11], r[9]/U))
