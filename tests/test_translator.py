from readme_rosetta.cache import Cache
from readme_rosetta.translator import Translator


def test_translates_and_caches(make_translator, tmp_path):
    translator, backend = make_translator()
    assert translator.translate("Hello world", "es") == "olleHx dlrowx"
    assert translator.translate("Hello world", "es") == "olleHx dlrowx"
    assert len(backend.requests) == 1
    translator.cache.save()

    # A new run with the same cache makes no requests at all.
    again, backend2 = make_translator()
    assert again.translate("Hello world", "es") == "olleHx dlrowx"
    assert backend2.requests == []


def test_segments_are_batched(make_translator):
    translator, backend = make_translator()
    out = translator.translate_many([f"Sentence number {i}" for i in range(5)], "es")
    assert len(backend.requests) == 1
    assert out[0] == "ecnetneSx rebmunx 0"


def test_segments_without_prose_skip_the_model(make_translator):
    translator, backend = make_translator()
    assert translator.translate("`code` https://x.io", "es") == "`code` https://x.io"
    assert backend.requests == []


def test_protected_code_is_restored(make_translator):
    translator, _ = make_translator()
    assert translator.translate("Run `make test` now", "es") == "nuRx `make test` wonx"


def test_broken_placeholder_is_retried_with_feedback(make_translator):
    # First request drops the token; the retry is fine.
    def hook(text, request_no):
        return text.replace("⟦0⟧", "") if request_no == 1 else text

    translator, backend = make_translator(hook=hook)
    assert translator.translate("Run `make` now", "es") == "nuRx `make` wonx"
    assert len(backend.requests) == 2
    assert "rejected" in backend.requests[1][-1]["content"]
    assert translator.failures == []


def test_persistent_failure_keeps_source_and_is_reported(make_translator):
    translator, _ = make_translator(hook=lambda text, n: text + " ภาษาไทย")
    assert translator.translate("Hello there friend", "es") == "Hello there friend"
    assert len(translator.failures) == 1
    assert "unexpected script" in translator.failures[0].reason
    # Failures are never cached.
    assert translator.cache.entries == {}


def test_wrong_script_is_rejected(make_translator):
    # Backend "translates" to Spanish while we asked for Japanese.
    translator, backend = make_translator()
    backend.target = "es"
    translator.translate("This sentence has plenty of letters", "ja")
    assert translator.failures and "not written in" in translator.failures[0].reason


def test_keep_terms_stay_readable_and_must_survive(make_translator):
    # The model sees the term (so the sentence makes sense) ...
    translator, backend = make_translator(keep_terms=["Rosetta"])
    translator.translate("Rosetta sets up translations", "es")
    assert "Rosetta" in backend.requests[0][0]["content"]
    # ... and a reply that translates it is rejected.
    assert translator.failures
    assert 'translated or dropped "Rosetta"' in translator.failures[0].reason


def test_segment_of_only_keep_terms_skips_the_model(make_translator):
    translator, backend = make_translator(keep_terms=["README Rosetta"])
    assert translator.translate("🗿 README Rosetta", "es") == "🗿 README Rosetta"
    assert backend.requests == []


def test_cache_key_depends_on_model_and_glossary(make_translator):
    translator, _ = make_translator()
    key = translator.cache_key("Hi", "es", "markdown")
    translator.glossary = {"es": {"pull request": "solicitud de cambios"}}
    assert translator.cache_key("Hi", "es", "markdown") != key
    translator.backend.model = "other"
    assert translator.cache_key("Hi", "es", "markdown") != key


def test_glossary_reaches_prompt(make_translator):
    translator, _ = make_translator(
        glossary={"es": {"pull request": "solicitud de cambios"}}
    )
    assert "pull request → solicitud de cambios" in translator.system_prompt("es")


def test_cache_prune_only_touches_processed_targets(tmp_path):
    cache = Cache(str(tmp_path / "c.json"))
    cache.entries = {"es:old": "x", "fr:old": "y"}
    cache.set("es:new", "z")
    assert cache.prune(["es"]) == 1
    assert set(cache.entries) == {"fr:old", "es:new"}


def test_old_cache_format_is_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"en:es:abc": "hola"}')
    assert Cache(str(path)).entries == {}


def test_backend_is_part_of_translator_id(make_translator):
    translator, _ = make_translator()
    assert isinstance(translator, Translator)
    assert translator.backend.id == "fake:test"


def test_wrapped_reply_is_unwrapped_without_retry(make_translator):
    translator, backend = make_translator(hook=lambda text, n: f"<span>{text}</span>")
    assert translator.translate("Hello world", "es") == "olleHx dlrowx"
    assert len(backend.requests) == 1


def test_invented_inline_html_is_retried(make_translator):
    translator, backend = make_translator(
        hook=lambda text, n: text.replace("dlrowx", "<b>dlrowx</b>") if n == 1 else text
    )
    assert translator.translate("Hello world again", "es") == "olleHx dlrowx niagax"
    assert "HTML tags" in backend.requests[1][0]["content"]


def test_bad_cached_entry_is_retranslated(make_translator):
    translator, backend = make_translator()
    source = "See [LICENSE](LICENSE) for details."
    key = translator.cache_key(source, "es", "markdown")
    translator.cache.set(
        key, "Siehe [LICENSE]-Datei](LICENSE)."
    )  # written by an older version
    assert (
        translator.translate(source, "es") == "eeSx [ESNECILx](LICENSE) rofx sliatedx."
    )
    assert len(backend.requests) == 1


def test_good_cached_entry_is_reused(make_translator):
    translator, backend = make_translator()
    source = "See [LICENSE](LICENSE) for details."
    translator.cache.set(
        translator.cache_key(source, "es", "markdown"), "Ver [LICENSE](LICENSE)."
    )
    assert translator.translate(source, "es") == "Ver [LICENSE](LICENSE)."
    assert backend.requests == []


def test_cache_recheck_ignores_code_when_judging_script(make_translator):
    translator, backend = make_translator("ja")
    source = "Run `readme-rosetta --backend ollama --model qwen2.5:7b` to translate"
    good = "`readme-rosetta --backend ollama --model qwen2.5:7b` を実行して翻訳します"
    translator.cache.set(translator.cache_key(source, "ja", "markdown"), good)
    assert translator.translate(source, "ja") == good
    assert backend.requests == []
