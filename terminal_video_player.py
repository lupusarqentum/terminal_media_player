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

from src.configuration import Configuration
from src.image_processing import ImageRenderer
from src.utils import (print_error, print_warn,
                       get_terminal_size, MediaTypes, recognize_media_type)


def watch_image(target_file_path: str, config: Configuration,
                terminal_rows: int, terminal_columns: int) -> None:
    """Reads image, renders it and prints to stdout."""
    image = cv2.imread(target_file_path)
    if image is None:
        print_error("Failed to load an image: " + target_file_path)
        sys.exit(-1)
    image_renderer = ImageRenderer(config)
    rendered_image = image_renderer.render(image,
                                           terminal_rows, terminal_columns)
    print(rendered_image)


def watch_video(target_file_path: str, config: Configuration,
                terminal_rows: int, terminal_columns: int) -> None:
    """Reads video, renders it frame-by-frame and prints to stdout."""
    image_renderer = ImageRenderer(config)
    #TODO: do not trust user
    cap = cv2.VideoCapture(target_file_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            terminal_rows, terminal_columns = get_terminal_size()
            rendered_frame = image_renderer.render(frame, terminal_rows,
                                                   terminal_columns)
            print(rendered_frame)
        else:
            break


def watch_audio(target_file_path: str, config: Configuration) -> None:
    print_error("Audio support is not yet implemented")
    sys.exit(-1)


if __name__ == "__main__":
    CONFIG_LOCATION_PREFIX = "./"
    config = Configuration()
    config_path = CONFIG_LOCATION_PREFIX + "config.json"
    if not config.read_and_apply_JSON_config(config_path):
        print_error("An error occurred when trying to apply config. \
Fallback to default config instead")
        default_config_path = CONFIG_LOCATION_PREFIX + "default_config.json"
        if not config.read_and_apply_JSON_config(default_config_path):
            print_error("An error occurred when trying to apply \
default config. Can't operate")
            sys.exit(-1)

    if len(sys.argv) < 2:
        target_file_path = "examples/input.png"
        print_warn("No input file path was provided. \
Assuming" + target_file_path)
    else:
        target_file_path = sys.argv[1]
        print("Target media path: \"" + target_file_path + "\"")

    terminal_rows, terminal_columns = get_terminal_size()

    media_type = recognize_media_type(target_file_path)
    if media_type == MediaTypes.Unknown:
        print_error("Failed to recognize file type: " + target_file_path)
        sys.exit(-1)
    elif media_type == MediaTypes.Image:
        watch_image(target_file_path, config, terminal_rows, terminal_columns)
    elif media_type == MediaTypes.Audio:
        watch_audio(target_file_path, config)
    elif media_type == MediaTypes.Video:
        watch_video(target_file_path, config, terminal_rows, terminal_columns)
