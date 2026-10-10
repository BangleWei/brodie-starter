"""Demo-day visibility guard: is the whole body in frame? (facetime/pipeline.py runs it between the pose and the Coach.)

Per frame it gets the MediaPipe image landmarks ((33,5) x,y,z,visibility,presence, normalised to the frame) or None.
- no pose for `no_pose_s` (3 s), or a pose whose head (nose) or ankles are outside the frame (x or y < 0 or > 1) or whose
  ankles are both below `vis_thr` visibility (0.3) while near the bottom edge (y > 0.9), for `partial_s` (3 s)  ->  not visible: the pipeline does not step the
  Coach (counting and corrections pause, the counter keeps its state) and the session clock freezes.
- entering "not visible" says "Step back so I can see your whole body" (at most once per `cue_every_s`, 20 s);
  when the whole body is back for `return_frames` consecutive frames after a cue, "Great, I can see you" once.
- `no_pose_s_total` counts the current run of frames with no pose at all, for the plan-mode "I will wait here" line.
Everything is frame-based at `fps` so replay and live agree.
"""
NOSE, L_ANKLE, R_ANKLE = 0, 27, 28
STEP_BACK = 'Step back so I can see your whole body'
SEE_YOU = 'Great, I can see you'
WAIT_HERE = 'I will wait here, step into view when you are ready'

def full_body_in_frame(lm, vis_thr=0.3, edge=0.9):
    """lm (33,5) -> True when the nose and both ankles are inside the frame and the ankles are not both near-invisible
    while sitting near the bottom edge (y > `edge`), which is what a cut-off foot looks like when MediaPipe still places
    it. Elsewhere in the frame low ankle visibility is normal (pushups, planks: feet flat on the floor, seen head-on)."""
    if lm is None: return False
    for j in (NOSE, L_ANKLE, R_ANKLE):
        if not (0.0 <= lm[j][1] <= 1.0) or not (0.0 <= lm[j][0] <= 1.0): return False
    low = lm[L_ANKLE][3] < vis_thr and lm[R_ANKLE][3] < vis_thr
    near_edge = max(lm[L_ANKLE][1], lm[R_ANKLE][1]) > edge
    return not (low and near_edge)

class Visibility:
    def __init__(self, fps=10, no_pose_s=3.0, partial_s=3.0, cue_every_s=20.0, return_frames=5, vis_thr=0.3):   # no_pose_s 3 s: a person walking off to rest should not be told to step back at once
        self.fps = fps; self.no_pose_n = int(no_pose_s * fps); self.partial_n = int(partial_s * fps)
        self.cue_every_n = int(cue_every_s * fps); self.return_n = return_frames; self.vis_thr = vis_thr
        self.i = 0; self.no_pose_run = 0; self.partial_run = 0; self.good_run = 0
        self.visible = True; self.last_cue_i = -10**9; self.awaiting_return = False; self.n_cues = 0

    @property
    def no_pose_s(self):
        return self.no_pose_run / self.fps

    def update(self, lm, ok):
        """One frame. Returns dict(visible, cues=[texts to say now], full=bool, no_pose_s)."""
        self.i += 1; cues = []
        full = ok and full_body_in_frame(lm, self.vis_thr)
        if not ok: self.no_pose_run += 1; self.partial_run = 0
        elif not full: self.partial_run += 1; self.no_pose_run = 0
        else: self.no_pose_run = 0; self.partial_run = 0
        self.good_run = self.good_run + 1 if full else 0
        lost = self.no_pose_run >= self.no_pose_n or self.partial_run >= self.partial_n
        if lost:
            self.visible = False
            if self.i - self.last_cue_i >= self.cue_every_n:
                cues.append(STEP_BACK); self.last_cue_i = self.i; self.awaiting_return = True; self.n_cues += 1
        elif full and self.good_run >= self.return_n:
            if not self.visible or self.awaiting_return:
                if self.awaiting_return: cues.append(SEE_YOU)
                self.visible = True; self.awaiting_return = False
        return dict(visible=self.visible, cues=cues, full=full, no_pose_s=self.no_pose_s)
