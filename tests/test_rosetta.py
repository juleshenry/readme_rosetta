import pytest
from readme_rosetta.markdown_handler import MarkdownHandler
from unittest.mock import MagicMock


def test_code_block_protection():
    translator = MagicMock()
    handler = MarkdownHandler(translator)

    text = """
# Header
This is some text with `inline code`.
And a block:
```python
def hello():
    print("world")
```
More text.
"""
    protected, placeholders = handler.protect_code_blocks(text)

    assert "ROSETTA_CB_0" in protected
    assert "ROSETTA_CB_1" in protected
    assert "inline code" not in protected
    assert "def hello()" not in protected

    restored = handler.restore_code_blocks(protected, placeholders)
    assert restored == text


def test_header_clean_link():
    translator = MagicMock()
    handler = MarkdownHandler(translator)

    assert handler.clean_header_for_link("# My Header") == "#my-header"
    assert handler.clean_header_for_link("# [Title](http://link)") == "#title"
