import subprocess, numpy as np
def read_frames(path, w=None, h=None):
    if w is None:
        o=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','csv=p=0',path],capture_output=True,text=True).stdout.strip().split(',')
        w,h=int(o[0]),int(o[1])
    raw=subprocess.run(['ffmpeg','-v','error','-i',path,'-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    return np.frombuffer(raw,np.uint8).reshape(-1,h,w,3)
