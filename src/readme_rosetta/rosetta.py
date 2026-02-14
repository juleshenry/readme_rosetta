from transformers import pipeline
import sys
import re
import os
import subprocess
from typing import Optional, List
import argparse

try:
    from .lang_codes import lang_codes
except ImportError:
    from lang_codes import lang_codes

try:
    import polib
except ImportError:
    polib = None

# Expanded NLLB mapping (common ones, but we'll try to guess for others)
# Format: {code: nllb_long_code}
NLLB_CODES = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "ara_Arab",
    "as": "asm_Beng",
    "ast": "ast_Latn",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "br": "bre_Latn",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "ceb": "ceb_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fa": "pes_Arab",
    "ff": "fuv_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "fy": "fry_Latn",
    "ga": "gle_Latn",
    "gd": "gla_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "ig": "ibo_Latn",
    "ilo": "ilo_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "lb": "ltz_Latn",
    "lg": "lug_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lav_Latn",
    "mg": "plt_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "sqi_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "wo": "wol_Latn",
    "xh": "xho_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "zh": "zho_Hans",
    "zu": "zul_Latn",
}

_translator_pipeline = None


def get_translator():
    global _translator_pipeline
    if _translator_pipeline is None:
        model_id = "facebook/nllb-200-distilled-600M"
        cache_dir = os.path.join(os.getcwd(), ".rosetta_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        print(f"Loading model {model_id}...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
        _translator_pipeline = (model, tokenizer)
    return _translator_pipeline


def translate_text(from_code: str, to_code: str, text: str) -> str:
    if not text.strip():
        return text

    src_lang = NLLB_CODES.get(from_code, "eng_Latn")
    tgt_lang = NLLB_CODES.get(to_code, "spa_Latn")

    model, tokenizer = get_translator()

    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=1000,
        )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def forge_stone(
    first_header: str,
    start_code: str,
    end_code: str,
    existing_table: Optional[str] = None,
) -> str:
    def clean_header_for_link(header: str) -> str:
        header = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", header)
        header = header.strip().lstrip("#").strip()
        return "#" + header.lower().replace(" ", "-")

    header_link = clean_header_for_link(first_header)
    lang_name = str(lang_codes.get(end_code, end_code))

    translated_lang_name = translate_text(start_code, end_code, lang_name)
    new_header_link = header_link + "-" + translated_lang_name.lower().replace(" ", "-")

    new_row = f"| {lang_name} | [Link to Head of Docs]({new_header_link}) |"

    if existing_table and "<!-- <Original README.md> -->" in existing_table:
        if f"| {lang_name} |" in existing_table:
            return existing_table

        lines = existing_table.strip().split("\n")
        last_row_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("|"):
                last_row_idx = i

        if last_row_idx != -1:
            lines.insert(last_row_idx + 1, new_row)
            return "\n".join(lines) + "\n"

    return f"""<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages](https://www.github.com/juleshenry/readme_rosetta)
| About | |
| ------ | ---- |
| English | [Link to Head of Docs]({header_link}) |
{new_row}
"""


def readme_md_translate(md_text_path: str, start_code: str, end_code: str):
    result = ""
    first_header = ""
    existing_table = ""
    in_rosetta = False

    if not os.path.exists(md_text_path):
        return "", ""

    with open(md_text_path, "r") as f:
        lines = f.readlines()

    lines_to_translate = []
    for o in lines:
        if "<!-- <Original README.md> -->" in o:
            in_rosetta = True
            existing_table += o
            continue

        if in_rosetta:
            if o.startswith("#") and "Documentation Support" not in o:
                in_rosetta = False
            else:
                existing_table += o
                continue

        if "<!-- toc -->" in o:
            break

        if o.startswith("#") and not first_header:
            first_header = o
            o_for_translation = (
                o.replace("\n", " ") + str(lang_codes.get(end_code, end_code)) + "\n"
            )
            lines_to_translate.append(o_for_translation)
        else:
            lines_to_translate.append(o)

    lnz = ""
    for i, line in enumerate(lines_to_translate):
        lnz += line
        if (i + 1) % 5 == 0:
            result += translate_text(start_code, end_code, lnz)
            lnz = ""
    if lnz:
        result += translate_text(start_code, end_code, lnz)

    rosetta = forge_stone(first_header, start_code, end_code, existing_table)
    return result, rosetta


def translate_po_file(po_path: str, from_code: str, to_code: str):
    if not polib:
        print("polib not installed, skipping PO translation")
        return

    print(f"Translating PO file: {po_path} to {to_code}")
    po = polib.pofile(po_path)
    for entry in po:
        if entry.msgid and not entry.msgstr:
            entry.msgstr = translate_text(from_code, to_code, entry.msgid)
    po.save()


def setup_sphinx(project_path: str, langs: List[str], start_code: str = "en"):
    docs_dir = os.path.join(os.getcwd(), "docs")
    source_dir = os.path.join(docs_dir, "source")

    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.getcwd(), "src"))

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        subprocess.run(
            [
                python_exe,
                "-m",
                "sphinx.cmd.quickstart",
                "-q",
                "-p",
                "Project",
                "-a",
                "Author",
                "-v",
                "0.1.0",
                docs_dir,
            ],
            env=env,
        )

    # Ensure conf.py has i18n settings and sys.path for autodoc
    conf_path = os.path.join(source_dir, "conf.py")
    if os.path.exists(conf_path):
        with open(conf_path, "r") as f:
            conf_content = f.read()

        updates = []
        if 'sys.path.insert(0, os.path.abspath("../../src"))' not in conf_content:
            updates.append(
                'import sys, os\nsys.path.insert(0, os.path.abspath("../../src"))'
            )
        if 'locale_dirs = ["locale/"]' not in conf_content:
            updates.append('locale_dirs = ["locale/"]')
        if "gettext_compact = False" not in conf_content:
            updates.append("gettext_compact = False")

        if updates:
            with open(conf_path, "a") as f:
                f.write("\n" + "\n".join(updates) + "\n")

    # Run apidoc
    subprocess.run(
        [python_exe, "-m", "sphinx.ext.apidoc", "-o", source_dir, project_path, "-f"],
        env=env,
    )

    # Generate gettext
    subprocess.run(
        [
            python_exe,
            "-m",
            "sphinx.cmd.build",
            "-M",
            "gettext",
            source_dir,
            os.path.join(docs_dir, "build"),
        ],
        env=env,
    )

    # Update and translate PO files
    for lang in langs:
        if lang == ".":
            continue
        print(f"Processing Sphinx translation for: {lang}")
        subprocess.run(
            [
                python_exe,
                "-m",
                "sphinx_intl",
                "update",
                "-p",
                os.path.join(docs_dir, "build", "gettext"),
                "-l",
                lang,
            ],
            cwd=docs_dir,
            env=env,
        )

        locale_dir = os.path.join(source_dir, "locale", lang, "LC_MESSAGES")
        if os.path.exists(locale_dir):
            for f in os.listdir(locale_dir):
                if f.endswith(".po"):
                    translate_po_file(os.path.join(locale_dir, f), start_code, lang)

    print("Sphinx setup and translation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="README Rosetta: Translate your documentation."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="README_SOURCE.md",
        help="Path to the source file or project directory.",
    )
    parser.add_argument(
        "--sphinx",
        action="store_true",
        help="Setup Sphinx documentation and translate it.",
    )
    parser.add_argument(
        "--gitbook",
        action="store_true",
        help="Generate GitBook compatible documentation (SUMMARY.md).",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to the output README file (default: README.md).",
    )
    parser.add_argument("--langs", nargs="+", help="Target languages for translation.")
    parser.add_argument(
        "--src-lang", default="en", help="Source language code (default: en)."
    )

    args = parser.parse_args()

    if args.sphinx:
        if not args.langs:
            print("Error: --langs is required with --sphinx")
            sys.exit(1)
        setup_sphinx(args.path, args.langs, args.src_lang)
    elif args.langs:
        # Bulk README translation using --langs
        src_file = args.path
        start_code = args.src_lang
        target_codes = args.langs
        output_file = args.readme

        if not os.path.exists(src_file):
            print(f"Error: {src_file} not found.")
            sys.exit(1)

        with open(src_file, "r") as f:
            source_content = f.read()

        current_rosetta = ""
        translated_sections = []

        for code in target_codes:
            print(f"Translating to {code}...")
            trans_text, rosetta_part = readme_md_translate(src_file, start_code, code)
            translated_sections.append(trans_text)
            if not current_rosetta:
                current_rosetta = rosetta_part
            else:
                current_rosetta = forge_stone("", start_code, code, current_rosetta)

        with open(output_file, "w") as out:
            out.write(current_rosetta)
            out.write("\n")
            out.write(source_content)
            for section in translated_sections:
                out.write("\n\n<!-- toc -->\n\n")
                out.write(section)

        if args.gitbook:
            with open("SUMMARY.md", "w") as summary:
                summary.write("# Summary\n\n")
                summary.write(f"* [Introduction]({args.readme})\n")
                for code in target_codes:
                    lang_name = str(lang_codes.get(code, code))
                    # GitBook works well with language subdirectories or sections
                    # Here we link to the translated sections in the same file
                    summary.write(f"* [{lang_name}]({args.readme})\n")
            print("SUMMARY.md created for GitBook.")

        print(f"Bulk translation complete! {output_file} created.")
    else:
        # Default single translation
        start_code = args.src_lang
        md_text = args.path
        end_code = "es"  # default
        print(f"Translating {md_text} from {start_code} to {end_code}...")
        translated_text, rosetta_table = readme_md_translate(
            md_text, start_code, end_code
        )
        with open(args.readme, "w") as nrm:
            nrm.write(rosetta_table)
            with open(md_text, "r") as og:
                nrm.write(og.read())
            nrm.write("\n<!-- toc -->\n")
            nrm.write(translated_text)
        print(f"Done! See {args.readme}")


if __name__ == "__main__":
    main()
