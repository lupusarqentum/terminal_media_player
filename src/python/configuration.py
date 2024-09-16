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

import os
import json
from string import printable

from src.python.terminal_utils import print_error


class Configuration:
    """Storage for all config values."""

    def load_default_values(self):
        """Assigns default config values."""
        self._character_aspect_ratio = 2.25
        self._ascii_characters_grayscale = " .'`^\",:;Il!i><~+_-?][}{1)(" + \
                                           "|\\/tfjrxnuvczXYUJCLQ0OZmwqp" + \
                                           "dbkhao*#MW&8%B@$"
        self._paint_background = False
        self._paint_foreground = True
        self._use_all_rgb_colors = True
        self._audio_enabled = True
        self._boldify = True

    def save_to_json(self, config_file_path: str) -> None:
        """Saves all config values to JSON file."""
        data = {"character_aspect_ratio": self._character_aspect_ratio,
                "ascii_characters_grayscale": self._ascii_characters_grayscale,
                "paint_background": self._paint_background,
                "paint_foreground": self._paint_foreground,
                "use_all_rgb_colors": self._use_all_rgb_colors,
                "enable_audio": self._audio_enabled,
                "boldify": self._boldify}
        try:
            os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
            with open(config_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except OSError:
            print_error("Failed to save config file at " + config_file_path)

    def try_apply_json(self, config_file_path: str) -> bool:
        """Parses JSON-formatted config file and applies it.

        If config file is invalid, or any field is missing,
            or any value is inappropriate, config applying is failed
            and no actual change happens.

        Returns:
            True, if config was successfully applied, False otherwise.
        """
        try:
            file = open(config_file_path, "r")
        except FileNotFoundError:
            print_error(f"Failed to found config: {config_file_path}")
            return False
        except OSError as e:
            print_error(f"Failed to read config: {config_file_path}\n{e}")
            return False
        with file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print_error(f"Failed to parse config:\n{e}")
                return False

        try:
            character_aspect_ratio = data["character_aspect_ratio"]
            ascii_characters_grayscale = data["ascii_characters_grayscale"]
            paint_background = data["paint_background"]
            paint_foreground = data["paint_foreground"]
            use_all_rgb_colors = data["use_all_rgb_colors"]
            audio_enabled = data["enable_audio"]
            boldify = data["boldify"]
        except KeyError:
            print_error("Failed to apply config because some value is missing")
            return False

        if not ((self._is_character_aspect_ratio(character_aspect_ratio)) and
                (self._is_ascii_grayscale(ascii_characters_grayscale)) and
                (type(paint_background) is bool) and
                (type(paint_foreground) is bool) and
                (type(use_all_rgb_colors) is bool) and
                (type(audio_enabled) is bool) and
                (type(boldify) is bool)):
            print_error("Failed to apply config because of invalid values")
            return False

        self._character_aspect_ratio = character_aspect_ratio
        self._ascii_characters_grayscale = ascii_characters_grayscale
        self._paint_background = paint_background
        self._paint_foreground = paint_foreground
        self._use_all_rgb_colors = use_all_rgb_colors
        self._audio_enabled = audio_enabled
        self._boldify = boldify

        return True

    def _is_character_aspect_ratio(self, value) -> bool:
        return type(value) is float and value > 0

    def _is_ascii_grayscale(self, value) -> bool:
        return type(value) is str and all(char in printable for char in value)

    def get_character_aspect_ratio(self) -> float:
        """Returns character aspect ratio,
               calculated as height / width of terminal characters.
        """
        return self._character_aspect_ratio

    def get_ascii_characters_grayscale(self) -> str:
        """Returns a string of printable ASCII characters
            ordered from lightest to darkest."""
        return self._ascii_characters_grayscale

    def should_paint_background(self) -> bool:
        """Returns True if background should be painted."""
        return self._paint_background

    def should_paint_foreground(self) -> bool:
        """Returns True if characters should be painted."""
        return self._paint_foreground

    def use_all_rgb_colors(self) -> bool:
        """Returns True if 24-bit colors should be used."""
        return self._use_all_rgb_colors

    def audio_enabled(self) -> bool:
        """Returns True if audio should be played."""
        return self._audio_enabled

    def should_boldify(self) -> bool:
        """Returns True if characters should be printed bold."""
        return self._boldify


def get_config_location() -> str:
    user_name = os.getenv("USER")
    home_folder = os.getenv("HOME", "/home/" + user_name)
    config_folder = os.getenv("XDG_CONFIG_HOME",
                              default=home_folder + "/.config")
    result = config_folder + "/lupusarqentum/terminal_media_player.json"
    return result
