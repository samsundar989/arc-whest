import numpy as np, zipfile, json, math, os
ROOT=os.path.expanduser('~/mnt/arc-whest')
LAB=ROOT+'/cache/twin_labels'
N=1024; DEPTH=16
def _member_offsets(path):
    out={}
    with zipfile.ZipFile(path) as z:
        for i in z.infolist():
            assert i.compress_type==0
            with open(path,'rb') as f:
                f.seek(i.header_offset)
                h=f.read(30)
                nlen=int.from_bytes(h[26:28],'little'); elen=int.from_bytes(h[28:30],'little')
                dstart=i.header_offset+30+nlen+elen
                f.seek(dstart)
                magic=f.read(6); ver=f.read(2)
                assert magic==b'\x93NUMPY'
                if ver[0]==1:
                    hl=int.from_bytes(f.read(2),'little'); hoff=10
                else:
                    hl=int.from_bytes(f.read(4),'little'); hoff=12
                hdr=f.read(hl).decode('latin1')
            out[i.filename[:-4]]=(dstart+hoff+hl,hdr)
    return out
class Labels:
    def __init__(self,g):
        self.path=f'{LAB}/fit_{g:03d}.npz'
        self.off=_member_offsets(self.path)
    def vec(self,k):
        o,h=self.off[k]
        return np.fromfile(self.path,dtype=np.float32,count=DEPTH*N,offset=o).reshape(DEPTH,N).astype(np.float64)
    def mat(self,k,l):
        o,h=self.off[k]
        return np.fromfile(self.path,dtype=np.float32,count=N*N,offset=o+l*N*N*4).reshape(N,N).astype(np.float64)
def weights(g,l):
    per=N*N
    return np.fromfile(ROOT+'/cache/twin/full_w.f32',dtype=np.float32,count=per,offset=(g*DEPTH+l)*per*4).reshape(N,N).astype(np.float64)
def truth(g):
    t=np.load(ROOT+'/cache/twin/full_truth.npy',mmap_mode='r')
    return np.array(t[g],dtype=np.float64)
_erf=np.vectorize(math.erf)
def Phi(x): return 0.5*(1+_erf(np.asarray(x)/math.sqrt(2)))
def phi(x): return np.exp(-0.5*np.asarray(x)**2)/math.sqrt(2*math.pi)
