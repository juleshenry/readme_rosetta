import polib
import os
import re


def is_bad_translation(msgid, msgstr):
    if not msgstr.strip():
        return False

    # Check for hallucinated structural elements
    for char in ["#", "|", "```"]:
        if char in msgstr and char not in msgid:
            return True

    # Check for unbalanced backticks (very common cause of docutils errors)
    # Both single ` and double ``
    if msgstr.count("``") % 2 != 0:
        return True
    # Count single backticks excluding those in double backticks
    single_ticks = msgstr.replace("``", "").count("`")
    if single_ticks % 2 != 0:
        return True

    # Check for hallucinated RST headers/lines
    if "\n===" in msgstr or "\n---" in msgstr or "\n~~~" in msgstr or "\n===" in msgstr:
        if not ("===" in msgid or "---" in msgid or "~~~" in msgid):
            return True

    # Check for mismatched ROSETTA placeholders
    id_placeholders = set(re.findall(r"ROSETTA_(?:CB|RST)_\d+", msgid))
    str_placeholders = set(re.findall(r"ROSETTA_(?:CB|RST)_\d+", msgstr))
    if id_placeholders != str_placeholders:
        return True

    # If msgid is a single line but msgstr is multi-line with novel structure
    if "\n" not in msgid.strip() and msgstr.count("\n") > 2:
        return True

    # If msgid is short but msgstr is a novel, it's bad
    if len(msgid.split()) < 5 and len(msgstr.split()) > 25:
        return True

    return False


def cleanup_po(filepath):
    po = polib.pofile(filepath)
    count = 0
    fixed_style = 0
    for entry in po:
        # Fix common hallucination: bolding headers or short fields
        if entry.msgid.strip() and "**" in entry.msgstr and "**" not in entry.msgid:
            if len(entry.msgid.split()) < 10:
                entry.msgstr = entry.msgstr.replace("**", "")
                fixed_style += 1

        if is_bad_translation(entry.msgid, entry.msgstr):
            entry.msgstr = ""
            if "fuzzy" in entry.flags:
                entry.flags.remove("fuzzy")
            count += 1

    if count > 0 or fixed_style > 0:
        po.save()
        print(f"File {filepath}: Cleared {count}, Fixed style in {fixed_style}")


if __name__ == "__main__":
    for root, dirs, files in os.walk("docs/source/locale"):
        for file in files:
            if file.endswith(".po"):
                cleanup_po(os.path.join(root, file))
