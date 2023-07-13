import argostranslate.package
import argostranslate.translate
import sys


def trans(from_code, to_code, text):
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
    sa = sys.argv
    if not len(sa) == 4:
        error = "Must have form ~`babel aa bb target.file`"
        raise ValueError(error)
    print(f"Converting {sa[1]} => {sa[2]} on {sa[3]}...")
    Iam = trans(*sa[1:])
    print(Iam)
