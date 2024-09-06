import string


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
        self._target_file_path = "examples/input.png"
        # self.target_file_path = None
    
    def get_character_aspect_ratio(self) -> float:
        return self._character_aspect_ratio
    
    def set_character_aspect_ratio(self, character_aspect_ratio: float) -> None:
        assert type(character_aspect_ratio) == float and character_aspect_ratio > 0
        self._character_aspect_ratio = character_aspect_ratio
    
    def get_ascii_characters_grayscale(self) -> str:
        return self._ascii_characters_grayscale
    
    def set_ascii_characters_grayscale(self, ascii_characters_grayscale: str) -> None:
        assert type(ascii_characters_grayscale) == str and all(char in string.printable for char in ascii_characters_grayscale)
        self._ascii_characters_grayscale = ascii_characters_grayscale
    
    def get_polarization_level(self) -> float:
        return self._polarization_level
    
    def set_polarization_level(self, value: float) -> None:
        assert type(character_aspect_ratio) == float and -0.00001 <= character_aspect_ratio <= 1.00001
        self._polarization_level = value
    
    def get_colorful_background_enabled(self) -> bool:
        return self._colorful_background_enabled
    
    def set_colorful_background_enabled(self, value: bool) -> None:
        assert type(value) == bool
        self._colorful_background_enabled = value
    
    def get_colorful_charset_enabled(self) -> bool:
        return self._colorful_charset_enabled
    
    def set_colorful_charset_enabled(self, value: bool) -> None:
        assert type(value) == bool
        self._colorful_charset_enabled = value
    
    def get_audio_enabled(self) -> bool:
        return self._audio_enabled
    
    def set_audio_enabled(self, value: bool) -> None:
        assert type(value) == bool
        self._audio_enabled = value
    
    def get_background_color_offset(self) -> int:
        return self._background_color_offset
    
    def set_background_color_offset(self, value: int) -> None:
        assert type(value) == int
        self._background_color_offset = value
    
    def get_charset_color_offset(self) -> int:
        return self._charset_color_offset
    
    def set_charset_color_offset(self, value: int) -> None:
        assert type(value) == int
        self._charset_color_offset = value
    
    def get_target_file_path(self) -> str:
        return self._target_file_path
    
    def set_target_file_path(self, file_path: str) -> None:
        assert file_path is str
        self._target_file_path = file_path


