import argostranslate.package
import argostranslate.translate
import sys


def readme_md_translate(md_text):
    for line in md_text.split("\n"):
        line
    translated_text = ""
    with open(md_text) as f:
        i = 0
        lnz = ""
        for o in f.readlines():
            i += 1
            i %= 10
            lnz += o
            if not i:
                lnz.replace("\n", "")
                a = translate_text("en", "es", lnz)
                translated_text += a
                lnz = ""
        # Finish up remaining
        lnz.replace("\n", "")
        a = translate_text("en", "es", lnz)
        translated_text += a
    return translated_text


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
    md_text = 'pytorch_README.md'
    with open("newREADME.md", "a+") as nrm:
        with open(md_text) as og:
            nrm.write(og.read())
            nrm.write('<!-- toc -->')
        nrm.write(readme_md_translate(md_text))
