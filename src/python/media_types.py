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


class MediaTypes:
    Unknown = 0
    Image = 1
    Video = 2
    Audio = 3


def recognize_media_type(target_file_path: str) -> int:
    """Recognizes media type of file at target_file_path.

    Returns:
        MediaTypes.Unknown or MediaTypes.Image
            or MediaTypes.Video or MediaTypes.Audio.
    """
    return MediaTypes.Video
