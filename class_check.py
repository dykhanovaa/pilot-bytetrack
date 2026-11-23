"""Additional isolated TEST frames to cover van/others absent in short clips."""
import json,requests,xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np,torch,cv2
from pilot import ROOT,OUT,annotation,valid,clip,save
from ultralytics import YOLO
from fetch_data import get,REPO,BASE

def main():
 sources=json.loads((ROOT/'data'/'sources.json').read_text());rev=sources['revision']
 listing=get(f'https://huggingface.co/api/datasets/{REPO}/tree/{rev}/{BASE}/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML').json()
 def fetch(row):
  dst=ROOT/'data'/'annotations'/Path(row['path']).name
  if not dst.exists():dst.write_bytes(get(f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{row["path"]}').content)
  return dst
 with ThreadPoolExecutor(max_workers=8) as ex:paths=list(ex.map(fetch,listing))
 candidates={t:[] for t in ['car','bus','van','others']}
 for p in paths:
  reg,fr,_,_=annotation(p)
  for f,rows in fr.items():
   for ident,b,typ in rows:
    if b[0]>5 and b[1]>5 and b[2]<955 and b[3]<535:
     area=(b[2]-b[0])*(b[3]-b[1]);candidates[typ].append((area,p.stem,f,ident,b.tolist()))
 samples=[]
 for typ,rows in candidates.items():
  ids=set()
  for area,seq,f,ident,b in sorted(rows,reverse=True):
   if (seq,ident) in ids:continue
   ids.add((seq,ident));samples.append(dict(type=typ,sequence=seq,frame=f,GT_id=ident,GT_xyxy=b))
   if len(ids)==3:break
 torch.set_num_threads(4);model=YOLO(str(ROOT/'yolov8n.pt'));vis=OUT/'class_frames';vis.mkdir(exist_ok=True)
 for row in samples:
  seq,f=row['sequence'],row['frame'];p=ROOT/'data'/'images'/seq/f'img{f:05}.jpg';p.parent.mkdir(parents=True,exist_ok=True)
  url=f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{BASE}/DETRAC-Images/DETRAC-Images/{seq}/{p.name}'
  if not p.exists():p.write_bytes(get(url).content)
  im=cv2.imread(str(p));assert im is not None
  res=model.predict(im,imgsz=640,rect=False,conf=.1,iou=.7,agnostic_nms=True,max_det=300,device='cpu',verbose=False)[0].boxes.cpu().numpy().data
  b=np.array(row['GT_xyxy']);best=(0,None,None)
  for r in res:
   q=r[:4];inter=np.maximum(0,np.minimum(b[2:],q[2:])-np.maximum(b[:2],q[:2])).prod();iou=float(inter/((b[2:]-b[:2]).prod()+(q[2:]-q[:2]).prod()-inter))
   if iou>best[0]:best=(iou,int(r[5]),r.tolist())
  row.update(best_iou=best[0],COCO_class=best[1],COCO_name=model.names[best[1]] if best[1] is not None else None,
      selected=best[1] in [2,5,7],image_url=url)
  x,y,u,v=map(int,b);cv2.rectangle(im,(x,y),(u,v),(0,200,0),3)
  cv2.putText(im,f'GT {row["type"]} id {row["GT_id"]}',(x,max(25,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,180,0),2)
  if best[2] is not None:
   a,c,d,e=map(int,best[2][:4]);cv2.rectangle(im,(a,c),(d,e),(0,0,255),2)
   cv2.putText(im,f'COCO {row["COCO_name"]} IoU {best[0]:.2f}',(a,min(530,e+18)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)
  cv2.imwrite(str(vis/f'{row["type"]}_{seq}_{f}.jpg'),im)
 save('class_samples.json',samples)
 print(json.dumps(samples,indent=2))
if __name__=='__main__':main()
