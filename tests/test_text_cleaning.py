from app.tools.utility import clean_snippet


def test_clean_snippet_removes_extra_whitespace():
    raw_text = """
        Hybrid search is useful.


        It combines       lexical and semantic retrieval.
    """

    cleaned = clean_snippet(raw_text)

    assert "Hybrid search is useful." in cleaned
    assert "combines lexical and semantic retrieval." in cleaned
    assert "\n\n\n" not in cleaned