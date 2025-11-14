import sounddevice as sd
import numpy as np
import threading


class AudioPlayer(threading.Thread):
    """Threaded audio player.

    Usage:
    ```
        player = AudioPlayer(wav, sr)
        player.start()
        player.stop() # optional
    ```
    The player will stop automatically when the buffer finishes playing or
    when ``stop()`` is called.
    """

    def __init__(self, wav: np.ndarray, sr: int, finished_callback=None):
        super().__init__(daemon=True)
        self.wav = wav
        self.sr = sr
        self.finished_callback = finished_callback
        self._event = threading.Event()
        self._current_frame = 0
        self.stream = None

    def _callback(self, outdata: np.ndarray, frames: int, time, status):
        """sounddevice callback that reads from the provided wav buffer."""
        chunksize = min(len(self.wav) - self._current_frame, frames)
        outdata[:chunksize] = self.wav[self._current_frame:self._current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            # tell sounddevice to stop the stream when data is exhausted
            raise sd.CallbackStop()
        self._current_frame += chunksize

    def run(self):
        channels = self.wav.shape[1] if self.wav.ndim > 1 else 1
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            dtype=self.wav.dtype,
            channels=channels,
            finished_callback=self.finished_callback,
            callback=self._callback,
        )

        # Open the stream and block here until playback finishes or stop() is called.
        with self.stream:
            while self.stream.active and not self._event.is_set():
                # wait in small increments to react to stop() promptly
                self._event.wait(0.1)

    def stop(self):
        """Request playback stop; thread will exit shortly."""
        self._event.set()