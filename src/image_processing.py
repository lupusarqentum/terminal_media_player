import numpy
import cv2

from src.configuration import Configuration


class ImageProcessor:
    """
    """
    
    def __init__(self, config: Configuration) -> None:
        self._config = config
        self._hasLoaded = False
        self._hasRendered = False
    
    def load(self, target_file_path: str) -> None:
        self._source_image = cv2.imread(target_file_path)
        if self._source_image is None:
            raise FileNotFoundError("Can't find an image to load: " + target_file_path)
        self._hasLoaded = True
        self._hasRendered = False
    
    def render(self, terminal_rows: int, terminal_columns: int) -> None:
        assert self._hasLoaded
        character_aspect_ratio = self._config.get_character_aspect_ratio()
        source_shape = self._source_image.shape
        new_size = find_ASCII_image_size(source_shape[0], source_shape[1], terminal_rows, terminal_columns, character_aspect_ratio)
        resized_image = cv2.resize(self._source_image, new_size)
        grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        polarized_grayscale = polarize_grayscale(grayscale, self._config.get_polarization_level())
        ascii_image = asciify_grayscale(polarized_grayscale, self._config.get_ascii_characters_grayscale())
        if self._config.get_colorful_charset_enabled():
            ascii_image = colorize_ascii_image_characters(ascii_image, resized_image, self._config.get_charset_color_offset())
        if self._config.get_colorful_background_enabled():
            ascii_image = colorize_ascii_image_background(ascii_image, resized_image, self._config.get_background_color_offset())
        self._rendered_image = ascii_image
        self._terminal_columns = terminal_columns
        self._hasRendered = True
    
    def display(self) -> None:
        assert self._hasRendered
        display_ascii_image(self._rendered_image, self._terminal_columns)


def find_ASCII_image_size(image_height: int, image_width: int, terminal_rows: int, terminal_columns: int, character_aspect_ratio: float) -> tuple:
    """Finds ASCII image size that fits terminal size and keeps pixel aspect ratio.
    
    Receives pixel size of an image and finds appropriate size to display ASCII art in terminal.
    Found ASCII image size is in rows and columns. 
    Number of terminal rows and columns is used to ensure that the size will fit the terminal.
    Aspect ratio (in pixels) of ASCII image with found size will be the same as aspect ratio of source image.
    
    Parameters:
        image_height: Height of source image in pixels.
        image_width: Width of source image in pixels.
        terminal_rows: Maximum allowed number of rows.
        terminal_columns: Maximum allowed number of columns.
        character_aspect_ratio: Aspect ratio of terminal characters (height / width)
    
    Returns:
        A tuple containing rows and columns for a new size.
    """
    new_aspect_ratio = image_height / (image_width * character_aspect_ratio)
    new_columns = min(terminal_columns, terminal_rows / new_aspect_ratio)
    new_rows = new_columns * new_aspect_ratio
    new_columns = round(new_columns)
    new_rows = round(new_rows)
    return (new_columns, new_rows)


def asciify_grayscale(grayscale: numpy.ndarray, ascii_characters_grayscale: str) -> numpy.ndarray:
    """Converts a grayscale of an image to ASCII image.
    
    Parameters:
        grayscale: A matrix of integers from 0 to 255 (gray intensity).
    
    Returns:
        A matrix of ASCII characters. 
    """
    def asciify_pixel(intensity):
        pos = min(round(intensity / 255 * len(ascii_characters_grayscale)), len(ascii_characters_grayscale) - 1)
        return ascii_characters_grayscale[pos]
    vectorized = numpy.vectorize(asciify_pixel)
    return vectorized(grayscale)


def polarize_grayscale(grayscale: numpy.ndarray, polarization_level: float) -> numpy.ndarray:
    """Pushes dark areas of an image darker and light areas lighter.
    
    Parameters:
        grayscale: A matrix of integers from 0 to 255 (gray intensity).
        polarization_level: A float number from 0.0 to 1.0 controlling how aggressively
            polarization should happen. For example, 0.0 value means nothing will happen at all,
            while 1.0 value means very aggressive polarization.
    
    Returns:
        A new matrix of integers from 0 to 255 (gray intensity of polarized grayscale).
    """
    if polarization_level == 0.0:
        return grayscale
    average_intensity = numpy.average(grayscale)
    multiplier = 127 / average_intensity * polarization_level
    return cv2.addWeighted(grayscale, multiplier, grayscale, 1.0 - polarization_level, 0.0)


def convert_color_to_ansi_sequence(color: list, for_character: bool) -> str:
    """Converts RGB color into ANSI escape sequence for character or background.
    
    Finds one of 240 ANSI escape sequences that sets the best matching color.
    16 possible ANSI escape sequences for color setting are ignored because their corresponding 
    RGB values highly depend on a specific terminal.
    
    Parameters:
        color: An array-like object containing three integers from 0 to 255, i.e. BGR channels.
        for_character: True, if ANSI escape sequence for character color setting is required,
            and False if need to set color for background.
    
    Returns:
        An ANSI escape sequence.
    
    """
    is_shade_of_gray = max(color) - min(color) < 15
    color_number = None
    if is_shade_of_gray:
        color_number = 232 + numpy.clip((int(color[0]) - 8) // 10, 0, 23)
    else:
        b = max(int(color[0]) - 95, -1) // 40 + 1
        g = max(int(color[1]) - 95, -1) // 40 + 1
        r = max(int(color[2]) - 95, -1) // 40 + 1
        color_number = 16 + 36 * r + 6 * g + b
    beginning = "\033[38;5;" if for_character else "\033[48;5;"
    return beginning + str(color_number) + "m"


def colorize_ascii_image_characters(ascii_image: numpy.ndarray, source_image: numpy.ndarray, color_offset: int) -> numpy.ndarray:
    """Gives every ASCII image character a foreground color.
    
    Assigns each ASCII image character a foreground color based on source_image color.
    
    Parameters:
        ascii_image: A matrix of ASCII characters.
        source_image: An image. Must be the same shape as the ascii_image.
        color_offset: A value that will be added to every channel of every character color.
    
    Returns:
        A matrix containing python strings. Each string is a combination of escape sequence and ASCII character.
    """
    result = numpy.zeros(ascii_image.shape, dtype="object")
    for i in range(ascii_image.shape[0]):
        for j in range(ascii_image.shape[1]):
            fore_color_sequence = convert_color_to_ansi_sequence(source_image[i, j] + color_offset, True)
            result[i, j] = fore_color_sequence + ascii_image[i, j]
    result[ascii_image.shape[0] - 1, ascii_image.shape[1] - 1] += "\033[39m"
    return result


def colorize_ascii_image_background(ascii_image: numpy.ndarray, source_image: numpy.ndarray, color_offset: int) -> numpy.ndarray:
    """Gives every ASCII image character a background color.
    
    Assigns each ASCII image character a backround color based on source_image color.
    
    Parameters:
        ascii_image: A matrix of ASCII characters.
        source_image: An image. Must be the same shape as the ascii_image.
        color_offset: A value that will be added to every channel of every character background.
    
    Returns:
        A matrix containing python strings. Each string is a combination of escape sequence and ASCII character.
    """
    result = numpy.zeros(ascii_image.shape, dtype="object")
    for i in range(ascii_image.shape[0]):
        for j in range(ascii_image.shape[1]):
            back_color_sequence = convert_color_to_ansi_sequence(source_image[i, j] + color_offset, False)
            result[i, j] = back_color_sequence + ascii_image[i, j]
    for i in range(ascii_image.shape[0]):
        result[i, ascii_image.shape[1] - 1] += "\033[49m"
    return result


def display_ascii_image(ascii_image: numpy.ndarray, terminal_columns: int) -> None:
    """Prints an ASCII image to stdout centered horizontally.
    
    Parameters:
        ascii_image: An ASCII image to print. Must be a matrix of strings.
            Each string must contain exactly one printable character.
        terminal_columns: A size of the terminal in columns.
    
    Returns:
        None.
    """
    columns_used = ascii_image.shape[1]
    offset_length = (terminal_columns - columns_used) // 2
    offset = " " * offset_length
    for i in range(ascii_image.shape[0]):
        print(offset, end="")
        print("".join(ascii_image[i]))


