from readme_rosetta.slug import Slugger, heading_anchors, heading_text, slugify


def test_github_slugs():
    assert slugify("🗿 README Rosetta") == "-readme-rosetta"
    assert slugify("My Header") == "my-header"
    assert slugify("What's new? (v2.0)") == "whats-new-v20"
    assert slugify("Instalación rápida") == "instalación-rápida"
    assert slugify("日本語 の 見出し") == "日本語-の-見出し"
    assert slugify("snake_case & more") == "snake_case--more"


def test_heading_text_strips_markup():
    assert heading_text("## [Python](https://pypi.org) `pip`") == "Python pip"
    assert heading_text("# **Bold** and _em_ ##") == "Bold and em"


def test_duplicates_are_numbered():
    s = Slugger()
    assert [s.slug("Usage"), s.slug("Usage"), s.slug("Usage")] == [
        "usage",
        "usage-1",
        "usage-2",
    ]


def test_heading_anchors_skip_code_fences():
    text = "# Title\n```sh\n# not a heading\n```\n## Title\n"
    assert heading_anchors(text) == [(0, "title"), (4, "title-1")]
