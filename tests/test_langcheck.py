from readme_rosetta.langcheck import check_language

SRC = "Translate your README into dozens of languages at once."


def test_arabic_ok_even_with_product_names():
    assert (
        check_language(
            SRC, "ترجم ملف README إلى عشرات اللغات دفعة واحدة باستخدام Ollama.", "ar"
        )
        is None
    )


def test_arabic_with_thai_is_rejected():
    out = "ترجم ملف README إلى عشرات ภาษาต่างๆ اللغات دفعة واحدة."
    assert "unexpected script" in check_language(SRC, out, "ar")


def test_english_left_in_non_latin_target_is_rejected():
    assert "not written in" in check_language(SRC, SRC, "ja")


def test_unchanged_latin_target_is_rejected():
    assert "identical" in check_language(SRC, SRC, "es")


def test_short_identical_strings_are_fine():
    assert check_language("README Rosetta", "README Rosetta", "es") is None


def test_cyrillic_in_danish_is_rejected():
    out = "Oversæt din README til десятки sprog på én gang."
    assert check_language(SRC, out, "da") is not None


def test_japanese_accepts_mixed_kana_and_kanji():
    assert (
        check_language(SRC, "README を一度に何十もの言語に翻訳します。", "ja") is None
    )
