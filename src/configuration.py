import json
import string

from src.utils import print_error


class Configuration:
    """Storage for all config values.

    Methods defined here:

    read_and_apply_JSON_config(self, config_file_path)
        Reads config file located at config_file_path, parses and validates it,
        and applies it.

    get_character_aspect_ratio(self)
        Returns character aspect ratio.
        It should be calculated as height / width of the terminal characters.
        Calculated value should be supplied within the config file.

    get_ascii_characters_grayscale(self)
        Returns a string containing printable ASCII characters sorted from lightest to darkest.

    get_polarization_level(self)
        Returns intensity polarization level.

    get_colorful_background_enabled(self)
        Returns True if background of characters should be painted.

    get_colorful_charset_enabled(self)
        Returns True if characters should be painted.

    get_audio_enabled(self)
        Returns True if audio should be played.
    
    get_boldify(self)
        Returns True if characters should be printed bold.

    get_background_color_offset(self)
        Returns characters' backgrounds' color channels offset.

    get_charset_color_offset(self)
        Returns characters' color channels offset.
    """

    def read_and_apply_JSON_config(self, config_file_path: str) -> bool:
        """Reads JSON-formatted config file and applies it.

        Reads JSON-formatted config file, applies it.
        If config file is invalid, or any field is missing,
        or any value is inappropriate, False is returned and no actual change happens.
        True is returned if everything OK.

        Parameters:
            config_file_path: A config file path.

        Returns:
            True, if config file was successfully read and applied, and false otherwise.
        """
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
            boldify = data["boldify"]
            background_color_offset = data["background_color_offset"]
            charset_color_offset = data["charset_color_offset"]
        except KeyError:
            print_error("Failed to apply config because some of the values are missing")
            return False

        if not ((type(character_aspect_ratio) == float and character_aspect_ratio > 0) and
            (type(ascii_characters_grayscale) == str and all(char in string.printable for char in ascii_characters_grayscale)) and
            (type(polarization_level) == float and -0.001 <= polarization_level <= 1.001) and
            (type(colorful_background_enabled) == bool) and
            (type(colorful_charset_enabled) == bool) and
            (type(audio_enabled) == bool) and
            (type(boldify) == bool) and
            (type(background_color_offset) == int and 0 <= background_color_offset <= 255) and
            (type(charset_color_offset) == int and 0 <= charset_color_offset <= 255)):
            print_error("Failed to apply config because some of the fields contain invalud values")
            return False

        self._character_aspect_ratio = character_aspect_ratio
        self._ascii_characters_grayscale = ascii_characters_grayscale
        self._polarization_level = polarization_level
        self._colorful_background_enabled = colorful_background_enabled
        self._colorful_charset_enabled = colorful_charset_enabled;
        self._audio_enabled = audio_enabled
        self._boldify = boldify
        self._background_color_offset = background_color_offset
        self._charset_color_offset = charset_color_offset

        return True

    def get_character_aspect_ratio(self) -> float:
        """Returns character aspect ratio.

        It should be calculated as height / width of the terminal characters.
        Calculated value should be supplied within the config file.
        """
        return self._character_aspect_ratio
    
    def get_ascii_characters_grayscale(self) -> str:
        """Returns a string containing printable ASCII characters sorted from lightest to darkest."""
        return self._ascii_characters_grayscale
    
    def get_polarization_level(self) -> float:
        """Returns intensity polarization level."""
        return self._polarization_level
    
    def get_colorful_background_enabled(self) -> bool:
        """Returns True if background of characters should be painted."""
        return self._colorful_background_enabled
    
    def get_colorful_charset_enabled(self) -> bool:
        """Returns True if characters should be painted."""
        return self._colorful_charset_enabled
    
    def get_audio_enabled(self) -> bool:
        """Returns True if audio should be played."""
        return self_audio_enabled
    
    def get_boldify(self) -> bool:
        """Returns True if characters should be printed bold."""
        return self._boldify

    def get_background_color_offset(self) -> int:
        """Returns characters' backgrounds' color channels offset."""
        return self._background_color_offset
    
    def get_charset_color_offset(self) -> int:
        """Returns characters' color channels offset."""
        return self._charset_color_offset


