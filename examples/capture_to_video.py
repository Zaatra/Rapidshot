import rapidshot

# install OpenCV using `pip install rapidshot[cv2]` command.
import cv2

TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[Rapidshot] Capture benchmark"

target_fps = 30
screencapture = rapidshot.create(output_idx=0, output_color="BGR")
screencapture.start(target_fps=target_fps, video_mode=True)
writer = cv2.VideoWriter(
    "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (1920, 1080)
)
for i in range(600):
    writer.write(screencapture.get_latest_frame())
screencapture.stop()
writer.release()