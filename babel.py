import sys
import argostranslate.package
import argostranslate.translate

from_code = "en"
to_code = "es"

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

# Translate
translatedText = argostranslate.translate.translate("Hello World", from_code, to_code)
print(translatedText)
# '¡Hola Mundo!'

if __name__=='__main__':
	sa = sys.argv
	if not len(sa) == 4:
		error = "Must have form ~`babel aa bb target.file`"
		raise ValueError(error)
	print(f"Converting {sa[1]} => {sa[2]} on {sa[3]}...")
