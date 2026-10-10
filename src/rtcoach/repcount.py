"""Online, camera-agnostic rep counter on pose keypoints.
Idea: whatever joints move periodically dominate the first principal component of the recent pose window,
whatever the camera angle. Project the newest frame onto that component, track the signal with
hysteresis relative to the window's own amplitude, and count one rep per full down-up cycle.
"""
import numpy as np
JOINTS=[11,12,13,14,15,16,23,24,25,26,27,28]  # shoulders, elbows, wrists, hips, knees, ankles (MediaPipe)
class RepCounter:
    def __init__(self, fps=10, win_s=6.0, hi=0.7, lo=0.3, min_amp=0.25, min_period_s=0.6, smooth=3, use_world=False, hp_s=0.0, chain=False, gap_gate=0.0, backfill=False, chain_ratio=2.0, max_gap_s=6.0):
        self.chain=chain; self.cr=chain_ratio; self.maxgap=int(max_gap_s*fps); self.raw=[]
        self.hp=int(hp_s*fps)
        self.fps=fps; self.W=int(win_s*fps); self.hi=hi; self.lo=lo; self.min_amp=min_amp
        self.minp=int(min_period_s*fps); self.smooth=smooth; self.use_world=use_world
        self.buf=[]; self.state=None; self.count=0; self.last_rep_i=-999; self.i=-1; self.pc=None; self.rest=None
        self.events=[]; self.in_chain=False; self.gap_gate=gap_gate; self.backfill=backfill; self.rejected=[]; self.pending=None
    def feat(self, img, world):
        if self.use_world: x=world[JOINTS,:3].reshape(-1); return x
        p=img[JOINTS,:2].astype(np.float32)
        sh=(img[11,:2]+img[12,:2])/2; hp=(img[23,:2]+img[24,:2])/2
        torso=np.linalg.norm(sh-hp)+1e-3
        return (p/torso).reshape(-1)
    def update(self, img, world=None, ok=True):
        """img: (33,>=2) landmarks or None. Returns count if a rep completed on this frame else None."""
        self.i+=1
        if not ok or img is None:
            self.buf.append(self.buf[-1] if self.buf else None); 
        else: self.buf.append(self.feat(np.asarray(img,np.float32), None if world is None else np.asarray(world,np.float32)))
        self.buf=self.buf[-self.W:]
        X=np.array([b for b in self.buf if b is not None])
        if len(X)<max(10,self.W//3): return None
        # smoothing (causal moving average)
        k=self.smooth
        Xs=np.array([X[max(0,i-k+1):i+1].mean(0) for i in range(len(X))])
        if self.hp>1:
            cs=np.cumsum(np.vstack([np.zeros((1,Xs.shape[1])),Xs]),0)
            lo_i=np.maximum(0,np.arange(len(Xs))-self.hp+1)
            Xs=Xs-(cs[np.arange(len(Xs))+1]-cs[lo_i])/(np.arange(len(Xs))+1-lo_i)[:,None]
        mu=Xs.mean(0); Z=Xs-mu
        _,S,Vt=np.linalg.svd(Z,full_matrices=False); v=Vt[0]
        if self.pc is not None and v@self.pc<0: v=-v
        self.pc=v
        s=Z@v; lo,hi=np.percentile(s,5),np.percentile(s,95); amp=hi-lo
        # amplitude in torso units (image mode) gates out standing still / fidgeting
        if amp<self.min_amp: self.state=None; return None
        x=(s[-1]-lo)/amp
        if self.state is None and self.backfill:
            # activation: replay the buffered window so reps already under way are not lost
            xs=(s-lo)/amp
            first=next((v for v in xs if v>self.hi or v<self.lo),None)
            if first is None: return None
            self.rest='hi' if first>self.hi else 'lo'
            st0=self.rest; n=0; last=-999
            for j,v in enumerate(xs):
                stj='hi' if v>self.hi else ('lo' if v<self.lo else None)
                if stj and stj!=st0:
                    st0=stj
                    if stj==self.rest and j-last>=self.minp: n+=1; last=j
            self.state=st0
            if n:
                self.count+=n; self.last_rep_i=self.i-(len(xs)-1-last); self.events.append(self.i); return self.count
            return None
        # rest side: where the window started (people start a set at rest)
        if self.rest is None: self.rest='lo' if x<0.5 else 'hi'
        st='hi' if x>self.hi else ('lo' if x<self.lo else None)
        if st is None: return None
        if self.state is None: self.state=st; return None
        if st!=self.state:
            self.state=st
            back_to_rest=(st==self.rest)
            if back_to_rest and self.i-self.last_rep_i>=self.minp:
                self.last_rep_i=self.i
                if not self.chain:
                    ev=self.events
                    if self.gap_gate and len(ev)>=2:
                        med=float(np.median(np.diff(ev[-5:])))
                        if self.i-ev[-1]>self.gap_gate*med:
                            self.rejected.append(self.i); self.events_reset_at=self.i
                            # a long gap: if the next interval looks like a rep, a new chain starts there
                            self.pending=self.i; return None
                    if self.pending is not None and self.i-self.pending<=6*self.fps:
                        pass
                    self.pending=None
                    self.count+=1; self.events.append(self.i); return self.count
                self.raw.append(self.i)
                r=self.raw
                if len(r)>=2:
                    g=r[-1]-r[-2]
                    ok=g<=self.maxgap
                    if ok and len(r)>=3 and self.in_chain:
                        g0=r[-2]-r[-3]; ok=(g0/self.cr<=g<=g0*self.cr)
                    if ok:
                        if not self.in_chain: self.count+=2; self.in_chain=True   # credit the chain's first rep now
                        else: self.count+=1
                        self.events.append(self.i); return self.count
                    else: self.in_chain=False; self.raw=[self.i]
                return None
        return None
