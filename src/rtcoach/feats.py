"""Per-frame feature construction shared by training (Modal) and live inference. Keep it dependency-light (numpy)."""
import numpy as np
L=[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]; R=[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
MIRROR=np.arange(33); MIRROR[L]=R; MIRROR[R]=L
POSE_DIM=33*2+2+33+33*3+1
def pose_frames(img, world, ok, mirror=False, rot=0.0, scale=1.0):
    """img (T,33,5) image landmarks, world (T,33,3), ok (T,) -> (T, POSE_DIM) float32."""
    img=np.asarray(img,np.float32); world=np.asarray(world,np.float32); ok=np.asarray(ok,bool)
    T=len(img); xy=img[:,:,:2].copy(); vis=img[:,:,3].copy(); w=world.copy()
    if mirror:
        xy=xy[:,MIRROR]; xy[:,:,0]=1-xy[:,:,0]; vis=vis[:,MIRROR]; w=w[:,MIRROR]; w[:,:,0]*=-1
    hip=(xy[:,23]+xy[:,24])/2; sh=(xy[:,11]+xy[:,12])/2
    torso=np.linalg.norm(sh-hip,axis=1)
    s=np.median(torso[ok]) if ok.any() else 1.0
    s=max(float(s),1e-3)
    rel=(xy-hip[:,None])/s
    if rot or scale!=1.0:
        c,si=np.cos(rot),np.sin(rot); Rm=np.array([[c,-si],[si,c]],np.float32); rel=rel@Rm.T*scale
    hmed=np.median(hip[ok],0) if ok.any() else np.zeros(2,np.float32)
    traj=(hip-hmed)/s
    f=np.concatenate([rel.reshape(T,-1),traj,vis,w.reshape(T,-1),ok[:,None].astype(np.float32)],1)
    f[~ok]=0
    return f.astype(np.float32)
def resample(x, speed):
    """Temporal speed change by nearest-index resampling (speed>1 = faster movement)."""
    T=len(x); n=max(2,int(round(T/speed))); idx=np.clip(np.round(np.arange(n)*speed).astype(int),0,T-1); return x[idx]
