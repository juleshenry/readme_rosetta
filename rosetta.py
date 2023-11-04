import argostranslate.package
import argostranslate.translate
import sys
from lang_codes import lang_codes


def forge_stone(first_header, start_code, end_code):
    first_header = '#'.join([x.strip() for x in first_header.split('#')]).replace(' ','-').lower()
    new_header = first_header + '-' + translate_text(start_code, end_code, lang_codes[end_code]).lower()
    print(new_header,'!!!')
    ROSETTA_STONE = f"""
<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages](https://www.github.com/juleshenry/readme_rosetta)
| About | |
| ------ | ---- |
| English | [Link to Head of Docs]({first_header}) |
| {lang_codes[end_code]} | [Link to Head of Docs]({new_header}) |
"""
    return ROSETTA_STONE

def readme_md_translate(md_text, start_code, end_code):
    for line in md_text.split("\n"):
        line
    result = ""
    first_header = ""
    with open(md_text) as f:
        i = 0
        lnz = ""
        for o in f.readlines():
            if '#' in o and not len(first_header):
                first_header = o
                o = o.replace('\n',' ') + lang_codes.get(end_code) + '\n'
                print(o)
            i += 1
            i %= 10
            lnz += o
            if not i:
                lnz.replace("\n", "")
                a = translate_text(start_code, end_code, lnz)
                result += a
                lnz = ""
        # Finish up remaining
        lnz.replace("\n", "")
        a = translate_text(start_code, end_code, lnz)
        result += a
    print('finishing translate')
    rosetta = forge_stone(first_header, start_code, end_code)
    return result, rosetta


def translate_text(from_code, to_code, text):
    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    available_package = list(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages,
        )
    )[0]
    download_path = available_package.download()
    argostranslate.package.install_from_path(download_path)

    # Translate
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = list(filter(lambda x: x.code == from_code, installed_languages))[0]
    to_lang = list(filter(lambda x: x.code == to_code, installed_languages))[0]
    translation = from_lang.get_translation(to_lang)
    translatedText = translation.translate(text)
    return translatedText


if __name__ == "__main__":
    CLI = False
    if CLI:
        sa = sys.argv
        if len(sa) != 4:
            error = "Must have form ~`babel aa bb target.file`"
            raise ValueError(error)
        print(f"Converting {sa[1]} => {sa[2]} on {sa[3]}...")
        Iam = translate_text(*sa[1:])
        print(Iam)
    md_text = 'README.md'
    with open("newREADME.md", "a+") as nrm:
        to_write = ""
        with open(md_text) as og:
            to_write += og.read()
            to_write += '\n<!-- toc -->\n'
        translated_text, rosetta_table = readme_md_translate(md_text, 'en', 'es')
        # print(rosetta_table)
        nrm.write(rosetta_table)
        nrm.write(to_write)
        nrm.write(translated_text)

