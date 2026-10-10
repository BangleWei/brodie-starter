"""E7: pure pose-geometry side rules for the one-sided stretches (quad stretch left/right, deltoid stretch left/right arm).
The classifier has (almost) no cue vocabulary for these exercises and the coaches' corrections are mostly "use the
other leg/arm" or "pull the foot back / pull the arm across", so two per-frame rules cover them from MediaPipe landmarks:
  wrong side held for >= dwell_wrong s            -> "<exercise> - wrong side"  (Use your other leg / arm)
  not stretching (no leg raised, or raised but the foot not held / arm not across) for >= dwell_nograb s after
  start_delay s                                   -> "<exercise> - no grab"     (Grab your foot and pull it back / Pull your arm across your chest)
  a switch between sides                          -> nothing for switch_quiet s
Geometry (image landmarks img (33,5) normalised x,y + world (33,3) metres, both per frame):
  quad: a leg is raised behind when its ankle is above its knee (image y) by raise_thr torso lengths; optionally its ankle
        is deeper than the hips (world z, behind_thr). The foot is held when the nearest wrist is within grab_thr torso
        lengths of that ankle (image space; world z of an occluded foot is unreliable).
  arm:  an arm crosses the chest when its wrist is past the opposite shoulder (mirror invariant: (x_wrist - x_opp_shoulder)
        has the sign opposite to (x_own_shoulder - x_opp_shoulder), scaled by torso^2 and below cross_thr) at chest height
        (|y_wrist - y_shoulders| < wy_thr torso lengths).
Numpy only. Usage from the Coach (trigger-mode branch, every frame, before the trigger):
    st=init_side_state(exercise)                       # None when the exercise has no rule
    c=side_cue(st, img, world, ok, t)                  # (text, label) or None; t = seconds since the exercise started
    ...and when the trigger fires at t:  c=override(st, t)   # (text, label) to say instead of the trigger's cue, or None
The Coach applies its own gap when it speaks a cue. Call note_said(st, t) if it speaks anything else, so cooldowns hold."""
import json, os, collections, numpy as np
CUES=json.load(open(os.path.join(os.path.dirname(__file__),'cues.json')))
LS,RS,LW,RW,LH,RH,LK,RK,LA,RA=11,12,15,16,23,24,25,26,27,28
RULES={'quad stretch (left)':('quad','L'),'quad stretch (right)':('quad','R'),'deltoid stretch (left arm)':('arm','L'),'deltoid stretch (right arm)':('arm','R')}
DEFAULT=dict(dwell_wrong=2.0, dwell_nograb=3.0, start_delay=5.0, cooldown=10.0, switch_quiet=2.0, smooth=5, max_per_rule=3,
             raise_thr=0.0, behind_thr=-9.0, grab_thr=9.0, cross_thr=-0.05, wy_thr=0.5, scale_win=30)
def labels_for(exercise): return (f'{exercise} - wrong side', f'{exercise} - no grab')
def init_side_state(exercise, fps=10, params=None):
    if exercise not in RULES: return None
    kind,side=RULES[exercise]; p=dict(DEFAULT,**(params or {}))
    return dict(ex=exercise, kind=kind, side=side, p=p, fps=fps, torso=collections.deque(maxlen=p['scale_win']), hist=collections.deque(maxlen=p['smooth']),
                cur=None, since=0.0, switch_t=-1e9, last_cue=-1e9, last_rule={}, n_rule={})
def signals(st, img, world, ok):
    """per-frame geometry for the state's exercise -> dict of scalars, or None when the pose is not detected."""
    if not ok: return None
    xy=np.asarray(img,np.float32)[:,:2]; wo=np.asarray(world,np.float32)
    hip=(xy[LH]+xy[RH])/2; sh=(xy[LS]+xy[RS])/2; tl=float(np.linalg.norm(sh-hip)); st['torso'].append(tl); s=max(float(np.median(st['torso'])),1e-3)
    d={}
    if st['kind']=='quad':
        for name,(K,A) in {'L':(LK,LA),'R':(RK,RA)}.items():
            d['raise'+name]=float(xy[K,1]-xy[A,1])/s; d['behind'+name]=float(wo[A,2]-(wo[LH,2]+wo[RH,2])/2)
            d['grab'+name]=float(min(np.linalg.norm(xy[LW]-xy[A]),np.linalg.norm(xy[RW]-xy[A])))/s
    else:
        d['crossL']=float((xy[LW,0]-xy[RS,0])*(xy[LS,0]-xy[RS,0]))/s**2; d['crossR']=float((xy[RW,0]-xy[LS,0])*(xy[RS,0]-xy[LS,0]))/s**2
        d['wyL']=float(xy[LW,1]-sh[1])/s; d['wyR']=float(xy[RW,1]-sh[1])/s
    return d
def classify(st, d):
    """-> 'correct' | 'wrong' | 'no' (not stretching: nobody raised / not across / raised but not held) | None (no pose)."""
    if d is None: return None
    p=st['p']; me=st['side']; ot='R' if me=='L' else 'L'
    if st['kind']=='quad':
        up={k:(d['raise'+k]>p['raise_thr'] and d['behind'+k]>p['behind_thr']) for k in 'LR'}
        if up[me]: return 'no' if d['grab'+me]>p['grab_thr'] else 'correct'
        if up[ot]: return 'wrong'
        return 'no'
    ac={k:(d['cross'+k]<p['cross_thr'] and abs(d['wy'+k])<p['wy_thr']) for k in 'LR'}
    if ac[me]: return 'correct'
    if ac[ot]: return 'wrong'
    return 'no'
def observe(st, cls, t):
    """feed one frame's class at time t (s since the exercise started); updates the smoothed state and timers."""
    p=st['p']; st['hist'].append(cls)
    votes=collections.Counter(c for c in st['hist'] if c is not None)
    if not votes or sum(votes.values())*2<len(st['hist']): return
    m=votes.most_common(1)[0][0]
    if m!=st['cur']:
        if st['cur'] in ('correct','wrong') and m in ('correct','wrong'): st['switch_t']=t
        st['cur']=m; st['since']=t
def active_rule(st, t):
    """the rule whose condition holds now (dwell satisfied, no side switch in the last switch_quiet s), ignoring cooldowns
    -> label or None. Lets the Coach override the trigger's cue text when the trigger fires."""
    p=st['p']; m=st['cur']
    if m is None or t-st['switch_t']<p['switch_quiet']: return None
    wrong_l,nograb_l=labels_for(st['ex'])
    if m=='wrong' and t-st['since']>=p['dwell_wrong']: return wrong_l
    if m=='no' and t>=p['start_delay'] and t-st['since']>=p['dwell_nograb']: return nograb_l
    return None
def speak(st, rule, t):
    """apply the cooldowns / per-episode caps for saying `rule` at t -> (text, label) or None (and book it when spoken)."""
    p=st['p']
    if rule is None or t-st['last_cue']<p['cooldown'] or t-st['last_rule'].get(rule,-1e9)<p['cooldown'] or st['n_rule'].get(rule,0)>=p['max_per_rule']: return None
    st['last_cue']=t; st['last_rule'][rule]=t; st['n_rule'][rule]=st['n_rule'].get(rule,0)+1; st['since']=t
    return CUES[rule]['cue'], rule
def update(st, cls, t):
    """one frame: observe, then speak the active rule if its cooldowns allow -> (text, label) or None."""
    observe(st,cls,t); return speak(st,active_rule(st,t),t)
def override(st, t):
    """the trigger fired at t: (text, label) of the active rule if one holds and its cooldowns allow, else None."""
    if st is None: return None
    return speak(st,active_rule(st,t),t)
def side_cue(st, img, world, ok, t):
    """one frame: landmarks (img (33,5), world (33,3)), ok flag, t seconds since the exercise started -> (text, label) or None."""
    if st is None: return None
    return update(st, classify(st, signals(st, img, world, ok)), t)
def note_said(st, t):
    """the coach spoke something else at t: the rules wait a full cooldown before speaking."""
    if st is not None: st['last_cue']=max(st['last_cue'],t)
