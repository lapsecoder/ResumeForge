"""Tests for text normalisation."""

from app.ingestion.normalizer import normalize_text


def test_crlf_to_lf() -> None:
    assert normalize_text("hello\r\nworld") == "hello\nworld"


def test_cr_to_lf() -> None:
    assert normalize_text("hello\rworld") == "hello\nworld"


def test_excessive_blank_lines_collapsed() -> None:
    text = "a\n\n\n\n\nb"
    assert normalize_text(text) == "a\n\nb"


def test_trailing_whitespace_stripped_per_line() -> None:
    text = "hello   \nworld   "
    assert normalize_text(text) == "hello\nworld"


def test_leading_trailing_stripped() -> None:
    text = "  \nhello\n  "
    assert normalize_text(text) == "hello"


def test_unicode_normalization() -> None:
    text = "\u0041\u0301"  # A + combining acute accent
    result = normalize_text(text)
    assert result == "\u00c1"  # precomposed A-acute


def test_preserves_single_blank_lines() -> None:
    text = "section one\n\nsection two"
    assert normalize_text(text) == "section one\n\nsection two"


def test_empty_string() -> None:
    assert normalize_text("") == ""


def test_whitespace_only() -> None:
    assert normalize_text("   \n\n   ") == ""


def test_mixed_line_endings() -> None:
    text = "a\r\nb\nc\r\n"
    assert normalize_text(text) == "a\nb\nc"


def test_preserves_meaningful_structure() -> None:
    text = "Name\nContact Info\n\nExperience\n\nEducation"
    result = normalize_text(text)
    assert "Name" in result
    assert "Contact Info" in result
    assert "Experience" in result
    assert "Education" in result


def test_mojibake_cp1252_em_dash_repaired() -> None:
    # UTF-8 bytes of U+2014 mis-decoded as cp1252: "â€""
    assert normalize_text("A \u00e2\u20ac\u201d B") == "A — B"


def test_mojibake_cp1252_quotes_and_bullet_repaired() -> None:
    # UTF-8 quotes/bullet bytes mis-decoded as cp1252. NFKC renders a
    # repaired ellipsis (U+2026) as three ASCII dots downstream.
    assert normalize_text(
        "\u00e2\u20ac\u0153x\u00e2\u20ac\u02dcy\u00e2\u20ac\u00a6z\u00e2\u20ac\u00a2"
    ) == "“x‘y...z•"


def test_mojibake_greek_console_em_dash_repaired() -> None:
    # "ΓÇö" is the UTF-8 em-dash bytes surfaced via cp437/ANSI rendering.
    assert normalize_text("\u0393\u00c7\u00f6") == "—"


def test_mojibake_cp437_bullet_repaired() -> None:
    # "ΓÇó" is UTF-8 bytes of U+2022 decoded as cp437/ANSI.
    assert normalize_text("\u0393\u00c7\u00f3") == "•"


def test_mojibake_family_maps_to_original_char() -> None:
    # Every repairable mojibake triplet (cp1252 + cp437) maps back to its
    # original typographic character; the ellipsis then collapses to "..."
    # under NFKC.
    cases = {
        "\u00e2\u20ac\u201d": "—",
        "\u00e2\u20ac\u201c": "–",
        "\u00e2\u20ac\u0153": "“",
        "\u00e2\u20ac\u02dc": "‘",
        "\u00e2\u20ac\u2122": "’",
        "\u00e2\u20ac\u00a2": "•",
        "\u0393\u00c7\u00f6": "—",
        "\u0393\u00c7\u00f4": "–",
        "\u0393\u00c7\u00a3": "“",
        "\u0393\u00c7\u00a5": "”",
        "\u0393\u00c7\u00ff": "‘",
        "\u0393\u00c7\u00d6": "’",
        "\u0393\u00c7\u00aa": "...",
        "\u0393\u00c7\u00f3": "•",
    }
    for mojibake, expected in cases.items():
        assert normalize_text(mojibake) == expected, repr(mojibake)


def test_mojibake_repair_preserves_legitimate_text() -> None:
    # Spaced-apart characters that merely resemble a mojibake triplet are
    # never merged, and real typographic characters pass through untouched.
    assert normalize_text("\u0393 \u00c7 \u00f3") == "\u0393 \u00c7 \u00f3"
    assert normalize_text("aΓÇb") == "aΓÇb"
    assert normalize_text("A — B • C") == "A — B • C"


def test_del_bullet_delimiter_replaced() -> None:
    assert normalize_text("Task\x7fDone") == "Task•Done"
    assert (
        normalize_text("Communication \x7f Tele-calling")
        == "Communication • Tele-calling"
    )
