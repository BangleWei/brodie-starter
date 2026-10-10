"""Inference wrapper for the trained clip classifier (exercise + fine-grained labels) on a sliding window."""
import numpy as np, torch, torch.nn as nn
from .feats import pose_frames
class Net(nn.Module):
    def __init__(s, D, d, layers, n_ex, n_lab, win):
        super().__init__(); s.inp=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,d)); s.pos=nn.Parameter(torch.zeros(1,win+1,d)); s.cls=nn.Parameter(torch.zeros(1,1,d))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*4,0.1,batch_first=True,norm_first=True),layers)
        s.ho=nn.LayerNorm(d); s.hex=nn.Linear(d,n_ex); s.hlab=nn.Linear(d,n_lab)
    def forward(s,x,m):
        h=torch.cat([s.cls.expand(len(x),-1,-1),s.inp(x)],1)+s.pos
        pad=torch.cat([torch.zeros(len(x),1,dtype=torch.bool,device=x.device),~m],1)
        h=s.ho(s.enc(h,src_key_padding_mask=pad)[:,0]); return s.hex(h), s.hlab(h)
class Classifier:
    def __init__(self, path, device='cpu'):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        self.exs=ck['exs']; self.labs=ck['labs']; self.lab_ex=np.array(ck['lab_ex']); self.inp=ck['inp']; self.win=ck['win']
        self.net=Net(ck['D'],ck['d'],ck['layers'],len(self.exs),len(self.labs),self.win); self.net.load_state_dict(ck['state']); self.net.eval().to(device); self.dev=device
        torch.set_num_threads(2)
    def window_feats(self, img, world, ok, dino=None):
        parts=[]
        if self.inp in ('pose','both'): parts.append(pose_frames(img,world,ok))
        if self.inp in ('dino','both'): parts.append(np.asarray(dino,np.float32))
        T=min(len(p) for p in parts); return np.concatenate([p[-T:] for p in parts],1)
    @torch.no_grad()
    def predict_many(self, X):
        """X list of (T<=win, D) windows -> (ex_probs, lab_probs)."""
        W=self.win; xs=[];ms=[]
        for x in X:
            x=x[-W:]; m=np.zeros(W,bool); m[:len(x)]=True
            if len(x)<W: x=np.concatenate([x,np.zeros((W-len(x),x.shape[1]),np.float32)])
            xs.append(x); ms.append(m)
        a,b=self.net(torch.tensor(np.stack(xs)).to(self.dev),torch.tensor(np.stack(ms)).to(self.dev))
        return a.softmax(-1).cpu().numpy(), b.sigmoid().cpu().numpy()
