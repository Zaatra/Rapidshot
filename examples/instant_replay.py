"""
A minimal example of creating an instant replay with hotkeys using pyav,
rapidshot, and pynput.

Hotkeys:
    - Ctrl+Alt+H: Save a replay of the last N seconds.
    - Ctrl+Alt+I: Stop the recording.

Press the hotkeys during execution to trigger the respective actions.
"""

import logging
import time
from collections import deque
from threading import Event, Lock

import rapidshot  # Updated library name
import av
from pynput import keyboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class InstantReplayRecorder:
    def __init__(
        self,
        target_fps: int = 120,
        replay_duration_sec: int = 10,
        resolution: tuple = (1920, 1080),
        output_format: str = "mpeg4",
        output_pix_fmt: str = "yuv420p",
        bitrate: int = 8_000_000,
    ):
        self.target_fps = target_fps
        self.replay_duration_sec = replay_duration_sec
        self.resolution = resolution
        self.output_format = output_format
        self.output_pix_fmt = output_pix_fmt
        self.bitrate = bitrate

        # Buffer holds encoded AV packets for the last replay_duration_sec seconds
        self.buffer = deque(maxlen=self.target_fps * self.replay_duration_sec)
        self.buffer_lock = Lock()
        self.stop_event = Event()
        self.replay_count = 0

        # Create the initial AV container and stream for writing replay video
        self.container, self.stream = self._create_container()

        # Initialize screen capture via rapidshot
        self.screencapture = rapidshot.create(output_color="RGB")
        self.screencapture.start(target_fps=self.target_fps, video_mode=True)

        logging.info("InstantReplayRecorder initialized. Press Ctrl+Alt+H to save replay, Ctrl+Alt+I to stop.")

    def _create_container(self):
        """Creates a new AV container and stream for recording the replay."""
        filename = f"replay{self.replay_count}.mp4"
        container = av.open(filename, mode="w")
        stream = container.add_stream(self.output_format, rate=self.target_fps)
        stream.pix_fmt = self.output_pix_fmt
        stream.width, stream.height = self.resolution
        stream.bit_rate = self.bitrate
        logging.info(f"Created new container: {filename}")
        return container, stream

    def save_replay(self):
        """Saves the current buffer as a replay video."""
        logging.info("Saving Instant Replay for the last {} seconds...".format(self.replay_duration_sec))
        with self.buffer_lock:
            for idx, packet in enumerate(self.buffer):
                packet.pts = packet.dts = idx
                self.container.mux(packet)
        # Flush any remaining frames
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        logging.info(f"Replay saved as replay{self.replay_count}.mp4")
        self.replay_count += 1
        self.container, self.stream = self._create_container()

    def stop_record(self):
        """Stops the recording loop."""
        logging.info("Stopping recording.")
        self.stop_event.set()

    def run(self):
        """Runs the main loop for capturing frames and handling hotkeys."""
        hotkey_listener = keyboard.GlobalHotKeys({
            "<ctrl>+<alt>+h": self.save_replay,
            "<ctrl>+<alt>+i": self.stop_record
        })
        hotkey_listener.start()
        logging.info("Hotkey listener started. Awaiting input...")

        try:
            while not self.stop_event.is_set():
                frame_data = self.screencapture.get_latest_frame()
                if frame_data is None:
                    continue
                try:
                    # Convert the captured frame to an AV video frame for encoding
                    frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    with self.buffer_lock:
                        for packet in self.stream.encode(frame):
                            self.buffer.append(packet)
                except Exception as e:
                    logging.error("Error encoding frame: %s", e)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Exiting...")
        finally:
            hotkey_listener.stop()
            hotkey_listener.join()
            self.screencapture.stop()
            self.container.close()
            logging.info("Recording stopped and resources released.")


def main():
    recorder = InstantReplayRecorder()
    recorder.run()


if __name__ == "__main__":
    main()