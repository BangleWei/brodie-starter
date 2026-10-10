"""E8: cadence rules for the exercises whose coach corrections are mostly "faster" (quick feet, high knees, buttkickers)
and "slow down" (shoulder gators). numpy only, same interface as side_rules.py:
    st=init_cadence_state(exercise, fps, params)      # None when the exercise has no rule
    observe(st, img, world, ok, t)                      # every frame (t = seconds since the exercise started)
    rule=active_rule(st, t); c=speak(st, rule, t)       # (text, label) or None; or just cadence_cue(st, img, world, ok, t)
    c=override(st, t)                                   # when the trigger fires: the active rule's cue instead of the trigger's
    note_said(st, t)                                    # the coach said something else
Signal (torso-normalised image landmarks, conventions of side_rules.py): legs: 'ankdiff' = (y_left_ankle - y_right_ankle)/torso
(alternating with the steps) or 'ankmean' = mean ankle height; arms (shoulder gators): 'elbowsep' = |x_left_elbow - x_right_elbow|/torso.
Causal cadence over the last `win` s: 'zc' = sign transitions (hysteresis h) of the window-mean-removed signal per second;
'acf' = 10/lag of the first autocorrelation peak (lags 2..20), 0 when the window amplitude (std) is below amp_min (not moving).
Rules: cadence below tau_slow for >= dwell s (after start_delay s) -> "<exercise> - too slow" (Pick up the pace; the
existing label 'buttkickers - stepping' is reused); above tau_fast for >= dwell s -> "<exercise> - too fast" (Slow it down,
existing label for shoulder gators). Thresholds were learned on train (scripts/e8_cadence_eval.py)."""
import json, os, collections, numpy as np
CUES=json.load(open(os.path.join(os.path.dirname(__file__),'cues.json')))
LS,RS,LE,RE,LW,RW,LH,RH,LA,RA=11,12,13,14,15,16,23,24,27,28
RULES={'quick feet':dict(kind='legs',slow='quick feet - too slow',fast=None),
       'high knees':dict(kind='legs',slow='high knees - too slow',fast=None),
       'buttkickers':dict(kind='legs',slow='buttkickers - stepping',fast=None),
       'shoulder gators':dict(kind='arms',slow=None,fast='shoulder gators - too fast')}
DEFAULT=dict(win=3.0, method='acf', signal='ankdiff', h=0.02, amp_min=0.02, tau_slow=3.3, tau_fast=0.7, dwell=1.0, start_delay=3.0, cooldown=6.0,
             max_per_rule=5, smooth=5, scale_win=30)
PER_EXERCISE={'quick feet':dict(method='acf',signal='ankdiff',amp_min=0.02,tau_slow=3.3),
              'high knees':dict(method='acf',signal='ankmean',amp_min=0.03,tau_slow=3.3),
              'buttkickers':dict(method='acf',signal='ankmean',amp_min=0.03,tau_slow=2.5),
              'shoulder gators':dict(method='zc',signal='elbowsep',h=0.05,tau_fast=0.7)}
def init_cadence_state(exercise, fps=10, params=None):
    if exercise not in RULES: return None
    p=dict(DEFAULT); p.update(PER_EXERCISE.get(exercise,{})); p.update(params or {})
    if 'per_exercise' in p: p.update(p['per_exercise'].get(exercise,{}))
    return dict(ex=exercise, rule=RULES[exercise], p=p, fps=fps, torso=collections.deque(maxlen=p['scale_win']), buf=collections.deque(maxlen=int(round(p['win']*fps))),
                hist=collections.deque(maxlen=p['smooth']), cad=0.0, cur=None, since=0.0, last_cue=-1e9, last_rule={}, n_rule={})
def signal(st, img, world, ok):
    """the per-frame raw signal (torso units) or None when the pose is not detected."""
    if not ok: return None
    xy=np.asarray(img,np.float32)[:,:2]; hip=(xy[LH]+xy[RH])/2; sh=(xy[LS]+xy[RS])/2; st['torso'].append(float(np.linalg.norm(sh-hip))); s=max(float(np.median(st['torso'])),1e-3)
    sig=st['p']['signal']
    if sig=='ankdiff': return float(xy[LA,1]-xy[RA,1])/s
    if sig=='ankmean': return -float(xy[LA,1]+xy[RA,1])/2/s
    if sig=='elbowsep': return abs(float(xy[LE,0]-xy[RE,0]))/s
    if sig=='wristsep': return abs(float(xy[LW,0]-xy[RW,0]))/s
    raise ValueError(sig)
def cadence_of(buf, p, fps=10):
    """causal cadence estimate from the window buffer (list of floats)."""
    if len(buf)<max(6,int(round(p['win']*fps))//2): return 0.0
    b=np.asarray(buf,np.float64); b=b-b.mean()
    if p['method']=='zc':
        h=p['h']; s=0; tr=0
        for v in b:
            if v>h and s<=0: tr+=(s!=0); s=1
            elif v<-h and s>=0: tr+=(s!=0); s=-1
        return tr/(len(b)/fps)
    if b.std()<p['amp_min']: return 0.0
    r=np.correlate(b,b,'full')[len(b)-1:]; r=r/max(r[0],1e-9); lags=np.arange(2,min(21,len(b)-1)); seg=r[lags]; k=int(np.argmax(seg))
    return fps/lags[k] if seg[k]>0.2 else 0.0
def observe(st, img, world, ok, t):
    """feed one frame; updates the cadence estimate and the smoothed slow/fast/ok state."""
    if st is None: return
    v=signal(st,img,world,ok); p=st['p']
    if v is None: v=st['buf'][-1] if st['buf'] else 0.0
    st['buf'].append(v); observe_cadence(st, cadence_of(list(st['buf']),p,st['fps']), t, full=len(st['buf'])>=int(round(p['win']*st['fps'])))
def observe_cadence(st, cad, t, full=True):
    """the state logic given a cadence estimate (lets the tuner replay precomputed cadence series)."""
    p=st['p']; st['cad']=cad
    cls=None if not full else ('slow' if (st['rule']['slow'] and st['cad']<p['tau_slow']) else ('fast' if (st['rule']['fast'] and st['cad']>p['tau_fast']) else 'ok'))
    st['hist'].append(cls); votes=collections.Counter(c for c in st['hist'] if c is not None)
    if not votes or sum(votes.values())*2<len(st['hist']): return
    m=votes.most_common(1)[0][0]
    if m!=st['cur']: st['cur']=m; st['since']=t
def active_rule(st, t):
    """label of the rule whose condition holds now (dwell satisfied), ignoring cooldowns, or None."""
    if st is None: return None
    p=st['p']; m=st['cur']
    if m is None or t<p['start_delay'] or t-st['since']<p['dwell']: return None
    if m=='slow': return st['rule']['slow']
    if m=='fast': return st['rule']['fast']
    return None
def speak(st, rule, t):
    p=st['p']
    if rule is None or t-st['last_cue']<p['cooldown'] or t-st['last_rule'].get(rule,-1e9)<p['cooldown'] or st['n_rule'].get(rule,0)>=p['max_per_rule']: return None
    st['last_cue']=t; st['last_rule'][rule]=t; st['n_rule'][rule]=st['n_rule'].get(rule,0)+1; st['since']=t
    return CUES[rule]['cue'], rule
def cadence_cue(st, img, world, ok, t):
    """one frame: observe, then speak the active rule if cooldowns allow -> (text, label) or None."""
    if st is None: return None
    observe(st,img,world,ok,t); return speak(st,active_rule(st,t),t)
def override(st, t):
    """the trigger fired at t: (text, label) of the active rule if one holds and cooldowns allow, else None."""
    if st is None: return None
    return speak(st,active_rule(st,t),t)
def note_said(st, t):
    if st is not None: st['last_cue']=max(st['last_cue'],t)
