import pytest
from readme_rosetta.sphinx_handler import SphinxHandler
from unittest.mock import MagicMock


def test_fix_rst_underlines():
    translator = MagicMock()
    handler = SphinxHandler(translator)

    text = "Short Title\n===\n\nLonger Title Here\n===\n"
    # "Longer Title Here" is 17 chars. "===" is 3 chars.
    fixed = handler.fix_rst_underlines(text)

    lines = fixed.splitlines()
    assert lines[0] == "Short Title"
    assert lines[1] == "==========="  # Length 11 matches "Short Title"
    assert lines[3] == "Longer Title Here"
    assert lines[4] == "================="  # Length 17 matches "Longer Title Here"


def test_restore_rst_with_underlines():
    translator = MagicMock()
    handler = SphinxHandler(translator)

    # Simulate a translation where placeholders are restored and then underlines fixed
    translated = "Tietoja projektista\n===\n"
    # "Tietoja projektista" is 19 chars
    fixed = handler.fix_rst_underlines(translated)

    lines = fixed.splitlines()
    assert lines[0] == "Tietoja projektista"
    assert lines[1] == "==================="  # 19 '=' chars
