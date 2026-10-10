"""MediaPipe pose wrapper: frames (RGB uint8) -> (33,5) image landmarks [x,y,z,vis,pres] + (33,3) world landmarks."""
import numpy as np, mediapipe as mp, os
from mediapipe.tasks.python import vision, BaseOptions
MODELS=os.path.join(os.path.dirname(__file__),'..','..','models')
class Pose:
    def __init__(self, variant='full', video=True):
        opts=vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=f'{MODELS}/pose_landmarker_{variant}.task'),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
        self.lm=vision.PoseLandmarker.create_from_options(opts); self.video=video
    def __call__(self, rgb, t_ms):
        img=mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        r=self.lm.detect_for_video(img, int(t_ms)) if self.video else self.lm.detect(img)
        if not r.pose_landmarks: return None, None
        a=np.array([[p.x,p.y,p.z,p.visibility or 0,p.presence or 0] for p in r.pose_landmarks[0]],np.float32)
        w=np.array([[p.x,p.y,p.z] for p in r.pose_world_landmarks[0]],np.float32)
        return a,w
    def close(self): self.lm.close()
