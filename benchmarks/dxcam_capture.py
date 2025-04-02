import time
import dxcam

TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[DXcam] Capture benchmark"

fps = 0
screencapture = dxcam.create(output_idx=0)
screencapture.start(target_fps=60)
for i in range(1000):
    image = screencapture.get_latest_frame()
screencapture.stop()
del screencapture