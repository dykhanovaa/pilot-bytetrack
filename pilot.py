"""Technical pilot only; does not test research hypotheses or benchmark speed.
Run .venv/bin/python pilot.py after fetch_data.py and synthetic.py.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,subprocess,sys,platform,time,csv,re,xml.etree.ElementTree as ET
import numpy as np,cv2,torch
from ultralytics import YOLO
from ultralytics.utils import ops
from skip_aware import SkipAware,boxes,stock,state,PARAMS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'TrackEval'))
import trackeval
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
GT_TYPES={'car','bus','van','others'}
COCO_TYPES={2,5,7} # motorcycle excluded: outside UA-DETRAC road motor-vehicle scope

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
def clip(b):
    a=np.asarray(b,dtype=float).copy();a[[0,2]]=a[[0,2]].clip(0,960);a[[1,3]]=a[[1,3]].clip(0,540);return a

def ignore_fraction(b,regions):
    # Exact rectangle union area; overlapping ignore regions counted once.
    x1,y1,x2,y2=b;area=max(0,x2-x1)*max(0,y2-y1)
    if not area:return 1.
    rects=[]
    for a,c,d,e in regions:
        l,r=max(x1,a),min(x2,d);t,u=max(y1,c),min(y2,e)
        if r>l and u>t:rects.append((l,t,r,u))
    xs=sorted({v for r in rects for v in (r[0],r[2])});union=0.
    for l,r in zip(xs,xs[1:]):
        spans=sorted((b,d) for a,b,c,d in rects if a<r and c>l)
        total=0.;end=-float('inf')
        for a,b in spans:
            total+=max(0,b-max(a,end));end=max(end,b)
        union+=(r-l)*total
    return union/area

def valid(b,regions):return b[2]>b[0] and b[3]>b[1] and ignore_fraction(b,regions)<=.5

def annotation(path):
    root=ET.parse(path).getroot();regions=[];frames={};counts=Counter()
    for x in root.findall('./ignored_region/box'):
        b=x.attrib;l,t,w,h=[float(b[k]) for k in ('left','top','width','height')];regions.append(clip([l,t,l+w,t+h]))
    for frame in root.findall('frame'):
        rows=[]
        for target in frame.findall('./target_list/target'):
            typ=target.find('attribute').get('vehicle_type');assert typ in GT_TYPES,typ
            a=target.find('box').attrib;l,t,w,h=[float(a[k]) for k in ('left','top','width','height')];b=clip([l,t,l+w,t+h])
            if valid(b,regions):rows.append((int(target.get('id')),b,typ));counts[typ]+=1
        frames[int(frame.get('num'))]=rows
    assert len(frames)==len(root.findall('frame')), 'duplicate frame numbers'
    assert all(len({r[0] for r in rows})==len(rows) for rows in frames.values())
    return regions,frames,counts,root.find('sequence_attribute').attrib

def write_mot(path,frames,gt=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for t,rows in frames.items():
            for row in rows:
                if gt: ident,b,_=row;score=1
                else:b=np.asarray(row[:4]);ident=int(row[4]);score=float(row[5])
                x,y,x2,y2=b
                # Class id 1 is a single-class interchange token, NOT pedestrian semantics.
                f.write(f'{t},{ident},{x:.5f},{y:.5f},{x2-x:.5f},{y2-y:.5f},{score:.5f},1,1\n')

def evaluate(gtfolder,trackerfolder,seqlens,tracker_names):
    config=dict(GT_FOLDER=str(gtfolder),TRACKERS_FOLDER=str(trackerfolder),
        SKIP_SPLIT_FOL=True,SEQ_INFO=seqlens,TRACKERS_TO_EVAL=tracker_names,
        DO_PREPROC=False,PRINT_CONFIG=False,BENCHMARK='MOT17',SPLIT_TO_EVAL='train')
    ds=trackeval.datasets.MotChallenge2DBox(config)
    met=[trackeval.metrics.HOTA({'PRINT_CONFIG':False}),
         trackeval.metrics.Identity({'THRESHOLD':.5,'PRINT_CONFIG':False}),
         trackeval.metrics.CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False})]
    results=[]
    for tr in tracker_names:
        for seq in seqlens:
            raw=ds.get_raw_seq_data(tr,seq)
            data=ds.get_preprocessed_seq_data(raw,'pedestrian')
            row={'tracker':tr,'sequence':seq,'GT_dets':data['num_gt_dets'],'pred_dets':data['num_tracker_dets']}
            for m in met:
                r=m.eval_sequence(data)
                for key in ['HOTA','DetA','AssA','DetRe','IDF1','IDSW']:
                    if key in r:row[key]=int(r[key]) if key=='IDSW' else float(np.mean(r[key])*100)
            results.append(row)
    return results

def checks(model):
    log=[]
    # 960x540 -> fixed square 640: ratio 2/3, vertical padding 140.
    src=torch.tensor([[96.,54.,192.,162.]])
    mapped=src.clone()*(2/3);mapped[:,[1,3]]+=140
    restored=ops.scale_boxes((640,640),mapped.clone(),(540,960))
    assert torch.allclose(src,restored,atol=1e-4)
    log.append('PASS letterbox scale_boxes roundtrip; absolute XYXY, no extra -1 offset')
    b=boxes([[96,54,192,162,.9,0]])
    assert np.allclose(b.xywh[0],[144,108,96,108])
    assert boxes().data.shape==(0,6)
    log.append('PASS Boxes XYXY->XYWH and empty shape (0,6) float32')
    assert abs(ignore_fraction([0,0,10,10],[[0,0,6,10],[4,0,8,10]])-.8)<1e-9
    assert valid([0,0,10,10],[[0,0,5,10]])
    assert not valid([0,0,10,10],[[0,0,5.1,10]])
    assert np.allclose(clip([-5,-7,970,550]),[0,0,960,540])
    log.append('PASS ignored-region UNION, >0.5 boundary, clipping, zero area')
    assert not valid([0,0,0,5],[])
    assert {i:model.names[i] for i in COCO_TYPES}=={2:'car',5:'bus',7:'truck'}
    assert 3 not in COCO_TYPES
    log.append('PASS class whitelist COCO car/bus/truck; motorcycle excluded')
    blank=model.predict(np.zeros((540,960,3),np.uint8),imgsz=640,rect=False,conf=.1,classes=[2,5,7],device='cpu',verbose=False)[0].boxes
    assert len(blank)==0
    tr=SkipAware();out=tr.step(blank.cpu().numpy(),observed=True);assert len(out)==0
    log.append('PASS actual detector on blank image -> empty Boxes -> tracker -> empty output')
    return log

def main_manifest():
    sources=json.loads((ROOT/'data'/'sources.json').read_text())
    train=[x for x in sources['files'] if 'Train-Annotations' in x['url']]
    assert len(train)==60
    rows=[]
    for item in train:
        p=ROOT/item['file'];regions,fr,counts,attrs=annotation(p)
        rows.append(dict(sequence=p.stem,frames=len(fr),GT_dets=sum(counts.values()),
            density=sum(counts.values())/len(fr),class_counts=dict(counts),attributes=attrs,
            last_annotated_frame=max(fr),missing_XML_frame_numbers=sorted(set(range(1,max(fr)+1))-set(fr)),
            density_denominator='annotated frames only; absent XML frames are unknown, NOT empty GT',
            annotation_sha256=digest(p),images_status='not_downloaded_for_main_series'))
    rows.sort(key=lambda x:(x['density'],x['sequence']))
    for i,r in enumerate(rows):r['density_group']=['low','medium','high'][i//20]
    save('main_manifest.json',{'source_revision':sources['revision'],'rows':rows,
        'rank_boundaries':[[rows[19]['density'],rows[20]['density']],[rows[39]['density'],rows[40]['density']]],
        'boundary_midpoints':[(rows[19]['density']+rows[20]['density'])/2,(rows[39]['density']+rows[40]['density'])/2]})
    return rows

def main():
    torch.set_num_threads(4)
    main_manifest() # before any quality evaluation
    model=YOLO(str(ROOT/'yolov8n.pt'))
    testlog=checks(model)
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements.lock').write_text(lock)
    def shell(*args):return subprocess.check_output(args,text=True).strip()
    passport=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        machine=platform.machine(),os=platform.platform(),python=sys.version,
        hardware='\n'.join(x for x in shell('system_profiler','SPHardwareDataType').splitlines() if not any(k in x for k in ['Serial Number','UUID','UDID'])),
        graphics=shell('system_profiler','SPDisplaysDataType'),
        memory_bytes=int(shell('sysctl','-n','hw.memsize')),power=shell('pmset','-g','custom'),
        backend='CPU',torch_threads=4,batch=1,precision='FP32',
        weights_sha256=digest(ROOT/'yolov8n.pt'),
        trackeval_commit=json.loads((ROOT/'vendor'/'TrackEval'/'UPSTREAM.json').read_text())['commit'],
        ultralytics='8.3.0',torch=torch.__version__,tracker_parameters=PARAMS,
        protocol_sha256=digest(ROOT/'protocol.json'),trackeval_numpy_patch_sha256=digest(ROOT/'trackeval_numpy.patch'),
        source_hashes={f:digest(ROOT/f) for f in ['skip_aware.py','synthetic.py','pilot.py','fetch_data.py']})
    save('passport.json',passport)
    seqs=json.loads((ROOT/'data'/'sources.json').read_text())['pilot_sequences']
    gtfolder=OUT/'mot'/'gt';trfolder=OUT/'mot'/'trackers';seqlens={};all_gt={};all_pred={}
    pilot_manifest=[];audit=[];detection_log=[]
    names=[f'{m}_k{k}' for m in ['stock','adapted','observed_only'] for k in range(1,11)]
    # Warmup technical only, excluded from inference durations. No speed claims.
    blank=np.zeros((540,960,3),np.uint8)
    kwargs=dict(imgsz=640,rect=False,conf=.1,iou=.7,max_det=300,agnostic_nms=True,
                classes=sorted(COCO_TYPES),augment=False,device='cpu',half=False,verbose=False)
    for _ in range(3):model.predict(blank,**kwargs)
    for seq in seqs:
        regions,frames,counts,attrs=annotation(ROOT/'data'/'annotations'/(seq+'.xml'))
        gt={t:frames[t] for t in range(1,121)};all_gt[seq]=gt;seqlens[seq]=120
        write_mot(gtfolder/seq/'gt'/'gt.txt',gt,True)
        (gtfolder/seq/'seqinfo.ini').write_text(f'[Sequence]\nname={seq}\nseqLength=120\nimWidth=960\nimHeight=540\nframeRate=25\n')
        cache={};cachefile=OUT/f'detections_{seq}.json'
        old=json.loads(cachefile.read_text()) if cachefile.exists() else None
        for t in range(1,121):
            imagepath=ROOT/'data'/'images'/seq/f'img{t:05}.jpg'
            im=cv2.imread(str(imagepath));assert im is not None and im.shape[:2]==(540,960)
            if old is None:
                begin=time.perf_counter();res=model.predict(im,**kwargs)[0]
                duration=time.perf_counter()-begin
                raw=res.boxes.cpu().numpy().data
                detection_log.append(dict(sequence=seq,frame=t,detector_seconds=duration))
            else:raw=np.asarray(old[str(t)],dtype=np.float32).reshape(-1,6)
            # Filter ignored areas before tracker; keep original class for audit.
            kept=[]
            for r in raw:
                b=clip(r[:4])
                if int(r[5]) in COCO_TYPES and valid(b,regions):kept.append([*b,float(r[4]),int(r[5])])
            cache[t]=kept
            # Best geometric counterpart per GT, independent of class label.
            for ident,b,typ in gt[t]:
                best=max([(float(trackeval.datasets._base_dataset._BaseDataset._calculate_box_ious(np.array([b]),np.array([r[:4]]),box_format='x0y0x1y1')[0,0]),int(r[5])) for r in kept],default=(0.,-1))
                audit.append(dict(sequence=seq,frame=t,GT_id=ident,GT_type=typ,GT_xyxy=b.tolist(),best_iou=best[0],COCO_class=best[1]))
        if old is None:cachefile.write_text(json.dumps(cache))
        pilot_manifest.append(dict(sequence=seq,split='test',frames=[1,120],GT_dets=sum(len(x) for x in gt.values()),
            empty_GT_frames=[t for t,x in gt.items() if not x],empty_detection_frames=[t for t,x in cache.items() if not x],
            ignored_regions=[r.tolist() for r in regions],attributes=attrs))
        for mode in ['stock','adapted','observed_only']:
            for k in range(1,11):
                tr=stock() if mode=='stock' else SkipAware();pred={};trace=[]
                for t in range(1,121):
                    obs=(t-1)%k==0
                    rows=[[*r[:5],0] for r in cache[t]] if obs else []
                    inp=boxes(rows)
                    result=tr.update(inp) if mode=='stock' else tr.step(inp if obs else None,observed=obs)
                    if mode=='observed_only' and not obs:result=np.empty((0,8))
                    # Forecasts may enter ignored areas; symmetric output filtering.
                    results=[]
                    for r in result:
                        r=r.copy();r[:4]=clip(r[:4])
                        if valid(r[:4],regions):results.append(r.tolist())
                    pred[t]=results
                    trace.append(dict(frame=t,observed=obs,output_source='measurement_update' if obs else ('none' if mode!='adapted' else 'causal_KF_forecast'),
                        detections=rows if obs else None,tracks=state(tr),output=results))
                name=f'{mode}_k{k}';all_pred[seq,name]=pred
                write_mot(trfolder/name/'data'/(seq+'.txt'),pred)
                save(f'trace_{seq}_{name}.json',trace)
        print('TRACKED',seq,flush=True)
    # Pin pilot sources and counts before reading metric values.
    save('pilot_manifest.json',pilot_manifest);save('class_audit.json',audit)
    
    if detection_log:save('detector_durations.json',detection_log)
    results=evaluate(gtfolder,trfolder,seqlens,names)
    save('metrics_full_frames.json',results)
    with (OUT/'metrics_full_frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    # Paired evaluation on S_k; timestamps compressed ONLY in separate evaluation view.
    sampled=[]
    for k in range(1,11):
        sample_gt=OUT/'sampled'/f'k{k}'/'gt';sample_tr=OUT/'sampled'/f'k{k}'/'trackers';lens={}
        for seq in seqs:
            idx=list(range(1,121,k));lens[seq]=len(idx)
            write_mot(sample_gt/seq/'gt'/'gt.txt',{j:all_gt[seq][t] for j,t in enumerate(idx,1)},True)
            for name in [f'adapted_k{k}','adapted_k1',f'stock_k{k}']:
                write_mot(sample_tr/name/'data'/(seq+'.txt'),{j:all_pred[seq,name][t] for j,t in enumerate(idx,1)})
        sampled+=evaluate(sample_gt,sample_tr,lens,list(dict.fromkeys([f'adapted_k{k}','adapted_k1',f'stock_k{k}'])))
    save('metrics_sampled_frames.json',sampled)
    # End-to-end metric sanity on an all-empty tracker and an all-empty GT sequence.
    mini_gt=OUT/'metric_checks'/'gt';mini_tr=OUT/'metric_checks'/'trackers'
    known={1:[(1,np.array([10,10,50,50]),'car')],2:[],3:[(1,np.array([10,10,50,50]),'car')]}
    perfect={1:[[10,10,50,50,1,.9]],2:[],3:[[10,10,50,50,1,.9]]}
    write_mot(mini_gt/'known'/'gt'/'gt.txt',known,True)
    for name,fr in [('perfect',perfect),('empty',{1:[],2:[],3:[]}),('switch',{1:perfect[1],2:[],3:[[10,10,50,50,2,.9]]})]:
        write_mot(mini_tr/name/'data'/'known.txt',fr)
    sanity=evaluate(mini_gt,mini_tr,{'known':3},['perfect','empty','switch'])
    assert sanity[0]['HOTA']==100 and sanity[0]['IDF1']==100
    assert sanity[1]['HOTA']==0 and sanity[1]['IDF1']==0
    assert sanity[2]['IDSW']==1
    save('metric_sanity.json',sanity);testlog.append('PASS TrackEval perfect=100, empty=0, ID switch after empty GT frame=1')
    for seq in seqs:
        a=all_pred[seq,'adapted_k1'];b=all_pred[seq,'stock_k1'];assert a==b
    testlog.append('PASS real k1 exact stock/adapted output parity on all three clips')
    with (OUT/'checks.log').open('a') as f:f.write('\n'.join(testlog)+'\n')
    # Cases for visual review: each sequence, late entry and sampled/forecast frames.
    vis=OUT/'frames';vis.mkdir(exist_ok=True)
    for seq in seqs:
        first={}
        for t,rows in all_gt[seq].items():
            for ident,b,typ in rows:first.setdefault(ident,t)
        late=next((t for t in sorted(first.values()) if t>1),21)
        chosen=sorted({1, min(late+5,120), min(late+6,120),61})
        for t in chosen:
            im=cv2.imread(str(ROOT/'data'/'images'/seq/f'img{t:05}.jpg'))
            for reg in pilot_manifest[seqs.index(seq)]['ignored_regions']:
                x,y,u,v=map(int,reg);cv2.rectangle(im,(x,y),(u,v),(128,128,128),1)
            for ident,b,typ in all_gt[seq][t]:
                x,y,u,v=map(int,b);cv2.rectangle(im,(x,y),(u,v),(0,200,0),2)
                cv2.putText(im,f'GT {ident} {typ}',(x,max(12,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,0),1)
            for r in all_pred[seq,'adapted_k5'][t]:
                x,y,u,v=map(int,r[:4]);cv2.rectangle(im,(x,y),(u,v),(0,0,255),2)
                cv2.putText(im,f'ID {int(r[4])}',(x,min(530,v+13)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
            cv2.putText(im,f'{seq} frame {t} k5 '+('OBSERVED' if (t-1)%5==0 else 'KF FORECAST'),(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
            cv2.imwrite(str(vis/f'{seq}_{t:03}.jpg'),im)
    print('\n'.join(testlog));print('PILOT COMPLETE',len(results),'full-frame rows')
if __name__=='__main__':main()
