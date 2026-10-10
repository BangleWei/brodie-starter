"""Latency probe gesture: both wrists above the head (image y of wrists 15/16 below the nose y) after having been below
for at least `below_s` (1 s). Returns True on that rising edge only (one beep per raise). Frame-based at `fps`.

TPose (gesture command, opt-in with --gesture-next): both arms straight out to the sides at shoulder height, held for
`hold_s` (1.5 s = 15 frames at 10 fps, one miss allowed), fires once and then not again for `refractory_s` (5 s).
The landmarks are MediaPipe's (33,5) normalised image coordinates [x, y, z, vis, pres], the same array the probe reads.
"""
NOSE, L_WRIST, R_WRIST = 0, 15, 16
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 11, 12, 13, 14

class ArmRaise:
    def __init__(self, fps=10, below_s=1.0, confirm=2):   # two consecutive seen-up frames (false beeps on flaky pose windows); callers time from the FIRST up frame
        self.need = int(below_s * fps); self.below_run = 0; self.up = False; self.confirm = confirm; self.up_run = 0

    def update(self, lm, ok):
        if not ok or lm is None:
            self.below_run = 0; self.up_run = 0; return False
        up = (lm[L_WRIST][1] < lm[NOSE][1] and lm[R_WRIST][1] < lm[NOSE][1]
              and lm[L_WRIST][3] > 0.3 and lm[R_WRIST][3] > 0.3)          # both wrists seen, not guessed
        if not up:
            self.below_run += 1; self.up = False; self.up_run = 0; return False
        self.up_run += 1
        if self.up_run < self.confirm:                                     # optional extra frames of confirmation
            return False
        fire = (not self.up) and self.below_run >= self.need
        self.up = True; self.below_run = 0
        return fire

def is_tpose(lm, y_tol=0.12, reach=0.9, min_shoulder_w=0.05):
    """One frame: both wrists within `y_tol` of the shoulder height (normalised y), both elbows roughly between shoulder
    and wrist in y, each wrist at least `reach` x the shoulder width away from its own shoulder horizontally and on the
    outer side (arms extended out to the sides, not crossed). A shoulder width under `min_shoulder_w` (side-on or far
    away) is never a T-pose. Jumping jacks pass through this for a frame or two per rep; the hold filters them."""
    ls, rs, le, re, lw, rw = lm[L_SHOULDER], lm[R_SHOULDER], lm[L_ELBOW], lm[R_ELBOW], lm[L_WRIST], lm[R_WRIST]
    sh_y = 0.5 * (ls[1] + rs[1]); sh_w = abs(ls[0] - rs[0])
    if sh_w < min_shoulder_w:
        return False
    if abs(lw[1] - sh_y) > y_tol or abs(rw[1] - sh_y) > y_tol:
        return False
    slack = 0.5 * y_tol
    for s, e, w in ((ls, le, lw), (rs, re, rw)):
        if not (min(s[1], w[1]) - slack <= e[1] <= max(s[1], w[1]) + slack):
            return False
    for s, w, out in ((ls, lw, ls[0] - rs[0]), (rs, rw, rs[0] - ls[0])):     # out: the direction away from the other shoulder
        d = w[0] - s[0]
        if d * out <= 0 or abs(d) < reach * sh_w:
            return False
    return True

class TPose:
    """observe(lm, ok) -> True on the frame the hold completes. Consecutive T-pose frames are counted; a frame without
    the pose (or without a pose at all) is a miss, `misses` of them in a row are tolerated inside the hold and one more
    resets it. After a fire the next one is refused for `refractory_s` (frame-timed, like the probe)."""
    def __init__(self, fps=10, hold_s=1.5, refractory_s=5.0, misses=1, y_tol=0.12, reach=0.9):
        self.need = int(round(hold_s * fps)); self.refractory = int(round(refractory_s * fps)); self.max_miss = misses
        self.y_tol = y_tol; self.reach = reach
        self.run = 0; self.miss = 0; self.cooldown = 0; self.n_fired = 0; self.pose = False

    def observe(self, lm, ok=True):
        self.pose = bool(ok and lm is not None and is_tpose(lm, self.y_tol, self.reach))
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.pose:
            self.run += 1; self.miss = 0
        elif self.run > 0:
            self.miss += 1
            if self.miss > self.max_miss:
                self.run = 0; self.miss = 0
        if self.run >= self.need and self.cooldown == 0:
            self.cooldown = self.refractory; self.run = 0; self.miss = 0; self.n_fired += 1
            return True
        return False
