import zipfile, struct, numpy as np
from numpy.lib import format as F
def member_memmap(path, name):
    z = zipfile.ZipFile(path)
    info = z.getinfo(name)
    assert info.compress_type == 0
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        h = f.read(30)
        nlen, elen = struct.unpack('<HH', h[26:30])
        start = info.header_offset + 30 + nlen + elen
        f.seek(start)
        v = F.read_magic(f)
        shape, fo, dt = F._read_array_header(f, v)
        off = f.tell()
    return np.memmap(path, dtype=dt, mode='r', offset=off, shape=shape, order='F' if fo else 'C')
