import json
from string import printable

from src.utils import print_error


class Configuration:
    """Storage for all config values."""

    def read_and_apply_JSON_config(self, config_file_path: str) -> bool:
        """Reads JSON-formatted config file and applies it.

        If config file is invalid, or any field is missing,
            or any value is inappropriate, config applying is failed
            and no actual change happens.

        Returns:
            True, if config was successfully read and applied, False otherwise.
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
            polarization_level = data["polarization_level"]
            colorful_background_enabled = data["colorful_background_enabled"]
            colorful_charset_enabled = data["colorful_charset_enabled"]
            audio_enabled = data["audio_enabled"]
            boldify = data["boldify"]
            background_color_offset = data["background_color_offset"]
            charset_color_offset = data["charset_color_offset"]
        except KeyError:
            print_error("Failed to apply config because some value is missing")
            return False

        if not ((self._is_character_aspect_ratio(character_aspect_ratio)) and
                (self._is_ascii_grayscale(ascii_characters_grayscale)) and
                (self._is_polarization_level(polarization_level)) and
                (type(colorful_background_enabled) is bool) and
                (type(colorful_charset_enabled) is bool) and
                (type(audio_enabled) is bool) and
                (type(boldify) is bool) and
                (background_color_offset in range(256)) and
                (charset_color_offset in range(256))):
            print_error("Failed to apply config because of invalid values")
            return False

        self._character_aspect_ratio = character_aspect_ratio
        self._ascii_characters_grayscale = ascii_characters_grayscale
        self._polarization_level = polarization_level
        self._colorful_background_enabled = colorful_background_enabled
        self._colorful_charset_enabled = colorful_charset_enabled
        self._audio_enabled = audio_enabled
        self._boldify = boldify
        self._background_color_offset = background_color_offset
        self._charset_color_offset = charset_color_offset

        return True

    def _is_character_aspect_ratio(self, value) -> bool:
        return type(value) is float and value > 0

    def _is_polarization_value(self, value) -> bool:
        return type(value) is float and -0.001 <= value <= 1.001

    def _is_ascii_grayscale(self, value) -> bool:
        return type(value) is str and all(char in printable for char in value)

    def get_character_aspect_ratio(self) -> float:
        """Returns character aspect ratio.

        It should be calculated as height / width of the terminal characters.
        Calculated value should be supplied within the config file.
        """
        return self._character_aspect_ratio

    def get_ascii_characters_grayscale(self) -> str:
        """Returns a string of printable ASCII characters
            ordered from lightest to darkest."""
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
        return self._audio_enabled

    def get_boldify(self) -> bool:
        """Returns True if characters should be printed bold."""
        return self._boldify

    def get_background_color_offset(self) -> int:
        """Returns characters' backgrounds' color channels offset."""
        return self._background_color_offset

    def get_charset_color_offset(self) -> int:
        """Returns characters' color channels offset."""
        return self._charset_color_offset
