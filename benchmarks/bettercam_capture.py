import time
import bettercam

TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[bettercam] Capture benchmark"

fps = 0
screencapture = bettercam.create(output_idx=0, output_color="BGRA")
screencapture.start(target_fps=60, video_mode=True)
for i in range(1000):
    image = screencapture.get_latest_frame()
screencapture.stop()
del screencapture