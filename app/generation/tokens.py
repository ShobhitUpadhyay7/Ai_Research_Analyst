from functools import lru_cache

from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-1.7B"


@lru_cache
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0

    tokenizer = _get_tokenizer()

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
) -> str:
    if max_tokens <= 0:
        return ""

    tokenizer = _get_tokenizer()

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(tokens) <= max_tokens:
        return text

    return tokenizer.decode(
        tokens[:max_tokens],
        skip_special_tokens=True,
    )