
#!/usr/bin/env python3
"""
events_h5_to_video.py

Render events.h5 (datasets: x,y,t,p) to MP4.
Example:
python events_h5_to_video.py --input events.h5 --output preview.mp4 --fps 120 --accumulation-ms 8.333
"""
import argparse,h5py,cv2
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True)
ap.add_argument("--output",required=True)
ap.add_argument("--fps",type=float,default=120.0)
ap.add_argument("--accumulation-ms",type=float,default=8.333)
ap.add_argument("--width",type=int,default=None)
ap.add_argument("--height",type=int,default=None)
args=ap.parse_args()

with h5py.File(args.input,"r") as f:
    x=f["x"][:]; y=f["y"][:]; t=f["t"][:]; p=f["p"][:]
    W=args.width or int(f.attrs.get("width", x.max()+1))
    H=args.height or int(f.attrs.get("height", y.max()+1))

acc_us=int(args.accumulation_ms*1000)
t0=int(t.min()); t1=int(t.max())
edges=np.arange(t0,t1+acc_us,acc_us,dtype=np.uint64)
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
vw=cv2.VideoWriter(args.output,fourcc,args.fps,(W,H),True)
idx=0
for start,end in zip(edges[:-1],edges[1:]):
    # 3-channel BGR image, gray background
    img=np.full((H,W,3),128,np.uint8)

    while idx<len(t) and t[idx]<end:
        if t[idx]>=start:
            if p[idx]:
                # ON = RED
                img[y[idx],x[idx]] = (0,0,255)
            else:
                # OFF = BLUE
                img[y[idx],x[idx]] = (255,0,0)
        idx+=1

    vw.write(img)
vw.release()
print("Done:",args.output)
