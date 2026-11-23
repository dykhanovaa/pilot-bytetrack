"""Download pinned public UA-DETRAC mirror annotations and three technical clips."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,json,hashlib,time
ROOT=Path(__file__).resolve().parent
REPO='sunnycloudhust/TrafficVision'
BASE='data/ua_detrac/raw/DETRAC'

def get(url):
 for n in range(4):
  try:
   r=requests.get(url,timeout=90);r.raise_for_status();return r
  except requests.RequestException:
   if n==3: raise
   time.sleep(n+1)

def main():
 rev='47b78997538030f06a3013d2a119bdfda025982f'
 train=get(f'https://huggingface.co/api/datasets/{REPO}/tree/{rev}/{BASE}/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML').json()
 test=get(f'https://huggingface.co/api/datasets/{REPO}/tree/{rev}/{BASE}/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML').json()
 pilot=['MVI_39031','MVI_39051','MVI_39211']
 jobs=[]
 for row in train+test:
  name=Path(row['path']).stem
  if row in train or name in pilot:
   jobs.append((row['path'], ROOT/'data'/'annotations'/(name+'.xml')))
 for seq in pilot:
  for t in range(1,121):
   name=f'img{t:05}.jpg'
   jobs.append((f'{BASE}/DETRAC-Images/DETRAC-Images/{seq}/{name}',ROOT/'data'/'images'/seq/name))
 def fetch(job):
  src,dst=job;url=f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{src}'
  dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():dst.write_bytes(get(url).content)
  return {'file':str(dst.relative_to(ROOT)), 'url':url,'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
 rows=[]
 with ThreadPoolExecutor(max_workers=10) as ex:
  for n,row in enumerate(ex.map(fetch,jobs),1):
   rows.append(row)
   if n%50==0: print('Downloaded',n,'/',len(jobs),flush=True)
 (ROOT/'data'/'sources.json').write_text(json.dumps({'repo':REPO,'revision':rev,'pilot_sequences':pilot,'pilot_split':'test','pilot_frames':[1,120],'train_sequences':len(train),'files':rows},indent=2))
 print('DONE',len(rows),rev)
if __name__=='__main__':main()
