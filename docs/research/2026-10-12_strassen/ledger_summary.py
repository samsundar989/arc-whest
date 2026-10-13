import pickle, sys
from collections import defaultdict
MM, L = pickle.load(open("census/v84.pkl","rb"))
U = 2*1024**3 - 1024**2
tot = sum(r[1] for r in L)+32768
print("TOTAL", tot, "u", tot/U, "F/B", tot/2**41, "ops", len(L)+1)
print("n _mm calls", len(MM), "flops in _mm", sum(m["flops"] for m in MM)/U, "leaf", sum(m["leaf"] for m in MM)/U, "ops", sum(m["ops"] for m in MM))
mm=[r for r in L if r[3][0] in ("_mm","_sb")]
print("ledger recs keyed _mm/_sb:", len(mm), sum(r[1] for r in mm)/U, " keyed _sb only:", sum(1 for r in mm if r[3][0]=="_sb"))
by=defaultdict(lambda:[0,0,0,0])
for m in MM:
    k=(m["A"],m["B"],m["depth"])
    by[k][0]+=1; by[k][1]+=m["flops"]; by[k][2]+=m["leaf"]; by[k][3]+=m["ops"]
print(len(by),"distinct (shape,depth)")
dd=defaultdict(lambda:[0,0,0,0])
for (A,B,d),v in by.items():
    for i in range(4): dd[d][i]+=v[i]
for d in sorted(dd): print("depth",d,"calls",dd[d][0],"flops u %.2f leaf %.2f glue %.2f ops %d"%(dd[d][1]/U,dd[d][2]/U,(dd[d][1]-dd[d][2])/U,dd[d][3]))
rows=sorted(by.items(), key=lambda kv:-kv[1][1])
for (A,B,d),v in rows[:45]:
    m,k=A; n=B[1]
    print("%-14s@%-14s d=%d leaf=(%g,%g,%g) calls=%3d flops=%7.3fu glue=%6.3fu (%.1f%%) frac_dense=%.4f"%(A,B,d,m/2**d,k/2**d,n/2**d,v[0],v[1]/U,(v[1]-v[2])/U,100*(v[1]-v[2])/v[1], v[1]/(v[0]*m*n*(2*k-1))))
# non-mm matmuls
other=[r for r in L if r[0]=="matmul" and r[3][0] not in ("_mm","_sb")]
print("matmul outside _mm:", len(other), sum(r[1] for r in other)/U)
print(L[0], L[100])
from collections import Counter
print(Counter(r[0] for r in L).most_common(30))
