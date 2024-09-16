# TerminalVideoPlayer, a program using command line interface to play videos.
# Copyright (C) 2024  Roman Lisov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# https://www.gnu.org/licenses/gpl-3.0.html

import time


class FramerateKeeper:
    """Keeps framerate at targeted (desired) value of frames per second.

    You should pass a targeted (desired) framerate to init function.
    start() should be called just before first frame is going to be processsed.
    After any frame has been processed or skipped due to any reason,
        on_frame_processed() should be called to notify about it.
    on_frame_processed() will optionally sleep some time,
        if frames are being processed faster than targeted framerate.
    You can call should_drop_frame() to return True
        if frames are being processsed slower than framerate.
        In this case, you might need to address this by yourself.
        If you decide to actually drop frame,
            you should call on_frame_processed() anyway.

    """

    def __init__(self, targeted_FPS: int) -> None:
        self.targeted_FPS = targeted_FPS
        self._frames_passed = 0
        self._targeted_frame_time = 1 / self.targeted_FPS

    def start(self) -> None:
        self._started_timestamp = time.time()

    def should_drop_frame(self) -> bool:
        time_passed = time.time() - self._started_timestamp
        targeted_frames_count = time_passed * self.targeted_FPS
        return targeted_frames_count - self._frames_passed > 1

    def on_frame_processed(self, sleep_if_overprocessing: bool) -> None:
        self._frames_passed += 1
        if sleep_if_overprocessing:
            time_passed = time.time() - self._started_timestamp
            targeted_frames_count = time_passed * self.targeted_FPS
            if self._frames_passed > targeted_frames_count:
                difference = self._frames_passed - targeted_frames_count
                delay = self._targeted_frame_time * difference
                time.sleep(delay)

    def get_frames_processed_count(self) -> int:
        return self._frames_passed
