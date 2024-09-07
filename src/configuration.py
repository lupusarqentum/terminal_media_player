import json
import string

from src.utils import print_error


class Configuration:
    """
    """
    
    def __init__(self) -> None:
        self._character_aspect_ratio = 2.25
        self._ascii_characters_grayscale = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        #self._ascii_characters_grayscale =  ".-:=!)[itCZ#3hX8%DRW"
        self._polarization_level = 1.0
        self._colorful_background_enabled = False
        self._colorful_charset_enabled = True
        self._audio_enabled = True
        self._background_color_offset = 0
        self._charset_color_offset = 0
    
    def read_and_apply_JSON_config(self, config_file_path: str) -> None:
        try:
            file = open(config_file_path, "r")
        except FileNotFoundError:
            print_error(f"Failed to found a config file: {config_file_path}")
            return False
        except OSError as e:
            print_error(f"Failed to read a config file: {config_file_path}\n{e}")
            return False
        with file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print_error(f"Failed to parse a config file:\n{e}")
                return False

        try:
            character_aspect_ratio = data["character_aspect_ratio"]
            ascii_characters_grayscale = data["ascii_characters_grayscale"]
            polarization_level = data["polarization_level"]
            colorful_background_enabled = data["colorful_background_enabled"]
            colorful_charset_enabled = data["colorful_charset_enabled"]
            audio_enabled = data["audio_enabled"]
            background_color_offset = data["background_color_offset"]
            charset_color_offset = data["charset_color_offset"]
        except KeyError:
            print_error("Failed to apply config because some of the values are missing")
            return False

        try:
            assert type(character_aspect_ratio) == float and character_aspect_ratio > 0
            assert type(ascii_characters_grayscale) == str and all(char in string.printable for char in ascii_characters_grayscale)
            assert type(polarization_level) == float and -0.001 <= polarization_level <= 1.001
            assert type(colorful_background_enabled) == bool
            assert type(colorful_charset_enabled) == bool
            assert type(audio_enabled) == bool
            assert type(background_color_offset) == int and 0 <= background_color_offset <= 255
            assert type(charset_color_offset) == int and 0 <= charset_color_offset <= 255
        except AssertionError:
            print_error("Failed to apply config because some of the fields contain invalud values")
            return False
    
        self._character_aspect_ratio = character_aspect_ratio
        self._ascii_characters_grayscale = ascii_characters_grayscale
        self._polarization_level = polarization_level
        self._colorful_background_enabled == colorful_background_enabled
        self._colorful_charset_enabled = colorful_charset_enabled;
        self._audio_enabled = audio_enabled
        self._background_color_offset = background_color_offset
        self._charset_color_offset = charset_color_offset

        return True

    def get_character_aspect_ratio(self) -> float:
        return self._character_aspect_ratio
    
    def get_ascii_characters_grayscale(self) -> str:
        return self._ascii_characters_grayscale
    
    def get_polarization_level(self) -> float:
        return self._polarization_level
    
    def get_colorful_background_enabled(self) -> bool:
        return self._colorful_background_enabled
    
    def get_colorful_charset_enabled(self) -> bool:
        return self._colorful_charset_enabled
    
    def get_audio_enabled(self) -> bool:
        return self_audio_enabled
    
    def get_background_color_offset(self) -> int:
        return self._background_color_offset
    
    def get_charset_color_offset(self) -> int:
        return self._charset_color_offset


