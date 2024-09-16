#!/usr/bin/env python3

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

import sys

import cv2

from src.python.configuration import Configuration, get_config_location
from src.python.image_processing import ImageRenderer
from src.python.framerate_keeper import FramerateKeeper
from src.python.terminal_utils import (print_error, print_warn,
                                       get_terminal_size, clear_terminal)
from src.python.media_types import MediaTypes, recognize_media_type


def watch_audio(target_file_path: str, config: Configuration) -> None:
    print_error("Audio support is not yet implemented")
    sys.exit(-1)


def watch_image(target_file_path: str, config: Configuration) -> None:
    """Reads image, renders it and prints to stdout."""
    terminal_rows, terminal_columns = get_terminal_size()
    image = cv2.imread(target_file_path)
    if image is None:
        print_error("Failed to load an image: " + target_file_path)
        sys.exit(-1)
    image_renderer = ImageRenderer(config)
    rendered_image = image_renderer.render(image,
                                           terminal_rows, terminal_columns)
    print(rendered_image)


def watch_video(target_file_path: str, config: Configuration) -> None:
    """Reads video, renders it frame-by-frame and prints to stdout."""
    image_renderer = ImageRenderer(config)
    cap = cv2.VideoCapture(target_file_path)
    terminal_rows, terminal_columns = get_terminal_size()
    terminal_rows -= 2
    clear_terminal()
    framerate_keeper = FramerateKeeper(25)
    framerate_keeper.start()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if not framerate_keeper.should_drop_frame():
            if framerate_keeper.get_frames_processed_count() % 30 == 0:
                # Because fetching terminal size is expensive,
                # it's done only once in 30 frames.
                terminal_rows_now, terminal_columns_now = get_terminal_size()
                terminal_rows_now -= 2
                # Clearing every frame introduces flickering.
                # So clearing is done only if size has changed.
                # Otherwise, garbage will might appear.
                if terminal_columns_now != terminal_columns or \
                   terminal_rows_now != terminal_rows:
                    terminal_rows = terminal_rows_now
                    terminal_columns = terminal_columns_now
                    clear_terminal()
            rendered_frame = image_renderer.render(frame, terminal_rows,
                                                   terminal_columns)
            print("\033[H" + rendered_frame)
        framerate_keeper.on_frame_processed(True)


if __name__ == "__main__":
    config = Configuration()
    config_file_location = get_config_location()
    if not config.try_apply_json(config_file_location):
        print_error("Failed to apply config file at " + config_file_location)
        print_error("Will fallback to default values")
        config.load_default_values()
        print_warn("There will be attempt to regenerate config file at " +
                   config_file_location)
        config.save_to_json(config_file_location)

    if len(sys.argv) < 2:
        print("No input file path was provided")
        print("Provide path to input file as a first command line argument")
        exit(-1)
    target_file_path = sys.argv[1]
    print("Target media path:", target_file_path)

    media_type = recognize_media_type(target_file_path)
    if media_type == MediaTypes.Unknown:
        print_error("Failed to recognize file type:", target_file_path)
        sys.exit(-1)
    elif media_type == MediaTypes.Image:
        watch_image(target_file_path, config)
    elif media_type == MediaTypes.Audio:
        watch_audio(target_file_path, config)
    elif media_type == MediaTypes.Video:
        watch_video(target_file_path, config)
