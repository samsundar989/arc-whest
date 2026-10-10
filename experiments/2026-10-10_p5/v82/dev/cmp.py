import numpy as np, glob, json
D='experiments/2026-10-10_p5/v82/dev/'
a0=np.load('C:/Users/samsu/AppData/Local/Temp/claude/c--Users-samsu-Documents-arc-whest/9a8b0ce7-0ba5-4d27-a5c2-15764de6e45e/scratchpad/v82a_dev03.npz'); a1=np.load(D+'v82a_3_10.npz')
base=np.concatenate([a0['preds'],a1['preds']]); t=np.concatenate([a0['truth'],a1['truth']])
mb=((base-t)**2).mean(axis=1)
for f in sorted(glob.glob(D+'it*_0_10.npz')):
    p=np.load(f)['preds']; m=((p-t)**2).mean(axis=1)
    bad=[e['i'] for e in json.loads(open(f[:-4]+'.out').read().strip().splitlines()[-1])['errors']]
    ok=np.array([i for i in range(10) if i not in bad])
    print(f.split('/')[-1][-15:], 'nets %d  vs v82a %+.3f%%  worse %d/%d'%(len(ok), 100*(m[ok].mean()/mb[ok].mean()-1), (m[ok]>mb[ok]).sum(), len(ok)))
