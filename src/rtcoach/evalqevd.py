"""QEVD-FIT-COACH long-range evaluation, reimplementing FitCoach/src/evaluators.py (see reports/LITERATURE_MAP.md 1.4).
Episodes: segment starts at a transition feedback, ends halfway through the next transition utterance (or session end).
Transition feedbacks are excluded from GT. Times are real seconds relative to segment start."""
import json, re, os, numpy as np
from .config import DATA
def load_episodes(tag, root=None):
    eps=[]
    for m in json.load(open(os.path.join(root or os.path.join(DATA,'meta'), f'lr_{tag}.json'))):
        ev=m['events']; tr=[i for i,e in enumerate(ev) if e['transition']]
        for k,i in enumerate(tr):
            s=ev[i]['t']
            if k+1<len(tr):
                n=ev[tr[k+1]]; end=n['t']+(n.get('end',n['t'])-n['t'])/2
            else: end=m['duration']
            mm=re.search(r'(?:First up (?:are|is)|Moving on to|It\'s time for|Let\'s do(?: some)?)\s+(.*?)[!.]?$', ev[i]['text'])
            exn=mm.group(1).strip().lower() if mm else None
            if exn is None or 'end of the session' in ev[i]['text'].lower(): continue
            gt=[dict(t=e['t']-s, text=e['text']) for e in ev if not e['transition'] and s<=e['t']<end]
            if not gt: continue
            eps.append(dict(session=m['session'], seg=k, start=s, end=end, exercise=exn, gt=gt, trans_text=ev[i]['text']))
    return eps
def _align(A_t, B_t, B_txt, tol=3.0):
    rows, cols, last = [], [], -1
    if len(B_t)==0: return rows, cols
    B=np.asarray(B_t,float)
    for i,x in enumerate(A_t):
        j=int(np.argmin((B-x)**2))
        if abs(x-B[j])<tol/2 and j>last and j not in cols and B_txt[j]!='':
            rows.append(i); cols.append(j); last=j
    return rows, cols
def score(eps, preds, tol=3.0, text=True, bert=False):
    """preds: dict key (session,seg) -> list of dict(t, text). returns metrics dict."""
    mg=mp=ng=npred=0; pairs=[]; dts=[]; dur=0.
    for e in eps:
        p=sorted(preds.get((e['session'],e['seg']),[]), key=lambda x:x['t'])
        gt_t=[g['t'] for g in e['gt']]; pr_t=[x['t'] for x in p]
        r,c=_align(gt_t, pr_t, [x['text'] for x in p], tol)
        r2,_=_align(pr_t, gt_t, [g['text'] for g in e['gt']], tol)
        mg+=len(r); mp+=len(r2); ng+=len(gt_t); npred+=len(pr_t); dur+=e['end']-e['start']
        for i,j in zip(r,c): pairs.append((e['gt'][i]['text'], p[j]['text'])); dts.append(p[j]['t']-gt_t[i])
    P=mp/(npred+1e-12); R=mg/(ng+1e-12); F=2*P*R/(P+R+1e-12)
    out=dict(P=round(P,4),R=round(R,4),TF=round(F,4),n_gt=ng,n_pred=npred,minutes=round(dur/60,1),
             unmatched_pred_per_min=round((npred-mp)/(dur/60),2), pred_per_min=round(npred/(dur/60),2),
             dt_mean=round(float(np.mean(dts)),3) if dts else None, dt_abs_mean=round(float(np.mean(np.abs(dts))),3) if dts else None)
    if text and pairs:
        from rouge_score import rouge_scorer
        from nltk.translate.meteor_score import meteor_score
        rs=rouge_scorer.RougeScorer(['rougeL'])
        out['ROUGE_L']=round(float(np.mean([rs.score(g,h)['rougeL'].fmeasure for g,h in pairs])),4)
        try: out['METEOR']=round(float(np.mean([meteor_score([g.split()],h.split()) for g,h in pairs])),4)
        except LookupError: pass
        if bert:
            from bert_score import score as bs
            _,_,f=bs([h for g,h in pairs],[g for g,h in pairs],lang='en',verbose=False); out['BERTScore']=round(float(f.mean()),4)
    out['_pairs']=pairs
    return out

def intent_class(intent):
    return {'correction':'correction','count':'count'}.get(intent,'positive')
def score_by_class(eps, preds, gt_cls, pred_cls, classes=('correction','count','positive'), tol=3.0):
    """Intent-aware T-F: run the official matcher separately inside each class (GT of class c vs predictions of
    class c) and micro-sum. Also returns per-class P/R/F. gt_cls(text)->class, pred_cls(pred dict)->class."""
    tot={c:[0,0,0,0] for c in classes}  # mg, mp, ng, np
    for e in eps:
        p=sorted(preds.get((e['session'],e['seg']),[]), key=lambda x:x['t'])
        for c in classes:
            g=[x for x in e['gt'] if gt_cls(x['text'])==c]; q=[x for x in p if pred_cls(x)==c]
            gt_t=[x['t'] for x in g]; pr_t=[x['t'] for x in q]
            r,_=_align(gt_t, pr_t, ['x']*len(q), tol); r2,_=_align(pr_t, gt_t, ['x']*len(g), tol)
            t=tot[c]; t[0]+=len(r); t[1]+=len(r2); t[2]+=len(g); t[3]+=len(q)
    def prf(t):
        P=t[1]/(t[3]+1e-12); R=t[0]/(t[2]+1e-12); return dict(P=round(P,3),R=round(R,3),F=round(2*P*R/(P+R+1e-12),3),n_gt=t[2],n_pred=t[3])
    allt=[sum(t[i] for t in tot.values()) for i in range(4)]
    return dict(intent_TF=prf(allt)['F'], **{c:prf(t) for c,t in tot.items()})
