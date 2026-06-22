import regex as re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode so that visually similar characters are represented consistently.
    NFC is usually safer for African language text with diacritics because it preserves
    composed characters where possible.
    """
    return unicodedata.normalize("NFC", text)


def clean_line(text: str, lowercase: bool = False) -> str:
    """
    Basic cleaning without destroying morphology.
    Avoid removing apostrophes, hyphens, tone marks, or non-English characters.
    """
    text = normalize_unicode(text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove empty bracket artifacts or verse numbers if they occur alone.
    text = re.sub(r"^\d+[\.:]\s*", "", text)

    return text


def is_good_line(text: str, min_chars: int = 3) -> bool:
    """
    Filter out empty or extremely short lines.
    """
    if len(text.strip()) < min_chars:
        return False

    # Keep lines that contain at least one letter.
    return bool(re.search(r"\p{L}", text))