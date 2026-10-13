"""Verify lossless transport against local extraction payload hashes."""
import json,zipfile,lzma,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
def job(r):
    with zipfile.ZipFile(HERE/'corpus/transport.zip') as z: raw=lzma.decompress(z.read(r['name']+'.xz'))
    assert len(raw)==r['raw'] and hashlib.sha256(raw).hexdigest()==r['sha']
    return r['name']
def main():
    start=time.monotonic(); records=json.loads((HERE/'corpus/TRANSPORT.json').read_text())['records']
    with ProcessPoolExecutor(max_workers=12) as pool: names=list(pool.map(job,records))
    assert len(names)==841
    (HERE/'corpus/TRANSPORT_VERIFIED.json').write_text(json.dumps(dict(episodes=len(names),raw_payload_hashes_match=True,seconds=time.monotonic()-start)))
    print('TRANSPORT_VERIFIED',len(names),flush=True)
if __name__=='__main__': main()
