"""Brodie's face on the call: a puppet drawn from the owner's render, never a generated head.

Port of the spike's src/agent/avatar.mjs (brodie-facetime) to PIL for the Python coach, trimmed to what the demo needs:
the head render with its features painted out (assets/avatar/head-clean.png), the glasses matted out (glasses.png) and the
landmarks of every feature (landmarks.json) are scaled onto a 360x640 plate; eyes, brows and a mouth are drawn back on per
frame in his style. The mouth is one dark rounded shape, always: no teeth, lips or tongue. While a cue plays the mouth
cycles through the open shapes; at rest it is the smile the render had. He blinks every few seconds and breathes (three
plates, the head a pixel or two higher or lower).

    av = Avatar()                      # loads and scales the parts once (about 50 ms)
    rgb = av.frame(speaking=True)      # bytes, W*H*3, RGB24, for rtc.VideoFrame
    av.W, av.H                         # 360, 640

The coach publishes this as a video track ("coach-video") with --avatar (facetime/coach_agent.py); Chert's bridge shows
the agent's published video to the FaceTime caller (docs: facetime/bring-your-own-room).
"""
import json, math, os, random, time
from PIL import Image, ImageDraw
from facetime import ROOT

PARTS = os.environ.get('RTCOACH_AVATAR_DIR', os.path.join(ROOT, 'assets', 'avatar'))
W, H = 360, 640
BACKDROP = (13, 27, 34)
INK = (34, 28, 23)
MOUTHS_SPEAKING = ['small', 'medium', 'o', 'small', 'wide', 'medium', 'o', 'small']
BOB_STEPS = 3; BREATH_S = 3.6
BLINK_EVERY = (2.8, 5.5); BLINK_FRAMES = 1


def _q(p0, p1, p2, n=12):
    """Points along a quadratic Bezier."""
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]


class Avatar:
    def __init__(self, width=W, height=H, parts=PARTS):
        self.W, self.H = width, height
        lm = json.load(open(os.path.join(parts, 'landmarks.json')))
        head = Image.open(os.path.join(parts, 'head-clean.png')).convert('RGBA')
        glasses = Image.open(os.path.join(parts, 'glasses.png')).convert('RGBA')
        # layoutOf() from avatar.mjs: the glasses span 94 % of the width, the eyes sit at 42 % of the height
        span = lm['glasses']['x1'] - lm['glasses']['x0'] + 1
        self.scale = s = width * 0.94 / span
        midline = (lm['eyes'][0]['cx'] + lm['eyes'][1]['cx']) / 2
        self.left = width / 2 - midline * s
        self.top = height * 0.42 - lm['eyes'][0]['cy'] * s
        size = (round(lm['width'] * s), round(lm['height'] * s))
        head_s = head.resize(size, Image.LANCZOS); glasses_s = glasses.resize(size, Image.LANCZOS)
        self.eyes = [dict(cx=self.left + e['cx'] * s, cy=self.top + e['cy'] * s, rx=e['rx'] * s, ry=e['ry'] * s) for e in lm['eyes']]
        self.brows = [dict(x0=self.left + b['x0'] * s, y0=self.top + b['y0'] * s, x1=self.left + (b['x1'] + 1) * s, y1=self.top + (b['y1'] + 1) * s)
                      for b in lm['brows']]
        m = lm['mouth']
        self.mouth = dict(cx=self.left + (m['x0'] + m['x1'] + 1) / 2 * s, top=self.top + m['y0'] * s, w=(m['x1'] - m['x0'] + 1) * s, h=(m['y1'] - m['y0'] + 1) * s)
        # the glasses crop, pasted over the eyes after they are drawn (eyes sit under the lenses)
        g = lm['glasses']; pad = 6
        box = (max(0, round(g['x0'] * s) - pad), max(0, round(g['y0'] * s) - pad), min(size[0], round((g['x1'] + 1) * s) + pad), min(size[1], round((g['y1'] + 1) * s) + pad))
        self.glasses_crop = glasses_s.crop(box); self.glasses_at = (round(self.left) + box[0], round(self.top) + box[1])
        # three plates: backdrop + head, the head a step higher per plate (breathing)
        self.plates = []
        for k in range(BOB_STEPS):
            plate = Image.new('RGB', (width, height), BACKDROP)
            plate.paste(head_s, (round(self.left), round(self.top) - k), head_s)
            self.plates.append(plate)
        self._mouth_i = 0; self._mouth_t = 0.0; self._next_blink = time.monotonic() + random.uniform(*BLINK_EVERY); self._blink_left = 0
        self.t0 = time.monotonic(); self.n = 0

    # ---- features ----
    def _eyes(self, d, dy, open_frac):
        for e in self.eyes:
            ry = max(1.0, e['ry'] * open_frac)
            d.ellipse([e['cx'] - e['rx'], e['cy'] + dy - ry, e['cx'] + e['rx'], e['cy'] + dy + ry], fill=INK)
            if open_frac > 0.5:   # a glint, upper left of the pupil
                gr = max(1.5, e['rx'] * 0.22)
                d.ellipse([e['cx'] - e['rx'] * 0.35 - gr, e['cy'] + dy - ry * 0.45 - gr, e['cx'] - e['rx'] * 0.35 + gr, e['cy'] + dy - ry * 0.45 + gr], fill=(236, 250, 247))

    def _brows(self, d, dy):
        for b in self.brows:
            mid = (b['y0'] + b['y1']) / 2 + dy; arch = (b['y1'] - b['y0']) * 0.9
            pts = _q((b['x0'], mid + arch / 2), ((b['x0'] + b['x1']) / 2, mid - arch), (b['x1'], mid + arch / 2))
            d.line(pts, fill=INK, width=max(2, round((b['y1'] - b['y0']) * 0.55)), joint='curve')

    def _mouth(self, d, dy, shape):
        m = self.mouth; cx, top, w, h = m['cx'], m['top'] + dy + m['h'] * 0.25, m['w'], m['h']
        half = w / 2; stroke = max(2, round(h * 0.42)); corner = max(2, round(h * 0.5))
        def line(span, rise, weight):
            d.line(_q((cx - span, top), (cx, top + rise * 2), (cx + span, top)), fill=INK, width=weight, joint='curve')
        def blob(span, rise, depth):
            up = _q((cx - span, top), (cx, top + rise * 2), (cx + span, top)); down = _q((cx + span, top), (cx, top + depth * 2), (cx - span, top))
            d.polygon(up + down, fill=INK); d.line(up + down + [up[0]], fill=INK, width=corner, joint='curve')
        def grin(span, depth):
            reach = span; belly = top + depth * 1.34
            pts = [(cx - reach, top), (cx + reach, top)] + [((1 - t) ** 3 * (cx + reach) + 3 * (1 - t) ** 2 * t * (cx + reach) + 3 * (1 - t) * t * t * (cx - reach) + t ** 3 * (cx - reach),
                                                           (1 - t) ** 3 * top + 3 * (1 - t) ** 2 * t * belly + 3 * (1 - t) * t * t * belly + t ** 3 * top) for t in (i / 14 for i in range(15))]
            d.polygon(pts, fill=INK); d.line(pts + [pts[0]], fill=INK, width=corner, joint='curve')
        if shape == 'rest': line(half, h * 0.82, stroke)
        elif shape == 'mm': line(half * 0.66, h * 0.16, max(2, round(stroke * 0.92)))
        elif shape == 'small': blob(half * 0.6, h * 0.34, h * 0.8)
        elif shape == 'medium': grin(half * 0.86, h * 1.35)
        elif shape == 'o': d.ellipse([cx - half * 0.44, top + h * 1.05 - h * 0.9, cx + half * 0.44, top + h * 1.05 + h * 0.9], fill=INK)
        elif shape == 'wide': grin(half * 1.04, h * 2.5)
        else: line(half, h * 0.82, stroke)

    # ---- one frame ----
    def frame(self, speaking=False, mouth=None, now=None):
        now = time.monotonic() if now is None else now
        k = int((math.sin((now - self.t0) * 2 * math.pi / BREATH_S) + 1) / 2 * (BOB_STEPS - 0.001)); dy = -k
        if self._blink_left > 0: open_frac, self._blink_left = 0.08, self._blink_left - 1
        elif now >= self._next_blink: open_frac, self._blink_left, self._next_blink = 0.08, BLINK_FRAMES - 1, now + random.uniform(*BLINK_EVERY)
        else: open_frac = 1.0
        if mouth is None:
            if speaking:
                if now - self._mouth_t > 0.11: self._mouth_i = (self._mouth_i + random.choice((1, 1, 2))) % len(MOUTHS_SPEAKING); self._mouth_t = now
                mouth = MOUTHS_SPEAKING[self._mouth_i]
            else: mouth = 'rest'
        img = self.plates[k].copy(); d = ImageDraw.Draw(img)
        self._eyes(d, dy, open_frac); self._brows(d, dy)
        img.paste(self.glasses_crop, (self.glasses_at[0], self.glasses_at[1] + dy), self.glasses_crop)
        self._mouth(d, dy, mouth)
        self.n += 1
        return img

    def frame_bytes(self, **kw):
        return self.frame(**kw).tobytes()


if __name__ == '__main__':   # python -m facetime.avatar [out.png]: a contact sheet of the mouths and a blink, for the eye
    import sys
    av = Avatar(); out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'tmp', 'avatar_sheet.png')
    shapes = ['rest', 'mm', 'small', 'medium', 'o', 'wide']
    sheet = Image.new('RGB', (W * 7, H), BACKDROP)
    for i, s in enumerate(shapes): sheet.paste(av.frame(mouth=s), (i * W, 0))
    av._blink_left = 1; sheet.paste(av.frame(mouth='rest'), (6 * W, 0))
    os.makedirs(os.path.dirname(out), exist_ok=True); sheet.save(out); print('wrote', out, 'frames so far', av.n)
    t = time.perf_counter(); [av.frame(speaking=True) for _ in range(50)]; print(f'{(time.perf_counter() - t) / 50 * 1000:.1f} ms per frame')
