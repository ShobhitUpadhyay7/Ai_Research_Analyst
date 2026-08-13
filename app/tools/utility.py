import re


def clean_snippet(text: str) -> str:
    """
    Cleans search snippets and small extracted text blocks.
    """
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)

    return text.strip()