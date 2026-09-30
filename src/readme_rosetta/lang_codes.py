"""
Language metadata: ISO codes, English names, native names and writing systems.

The script column drives the output-language check in ``langcheck.py``: a
translation into a language written in a non-Latin script must actually be
written (mostly) in that script.
"""

import re
from typing import Dict, NamedTuple, Optional, Tuple


class Language(NamedTuple):
    code: str
    name: str
    native: str
    scripts: Tuple[str, ...]


# code | English name | native name | scripts (comma separated)
_TABLE = """
af  | Afrikaans        | Afrikaans          | Latn
am  | Amharic          | አማርኛ               | Ethi
ar  | Arabic           | العربية            | Arab
ast | Asturian         | Asturianu          | Latn
az  | Azerbaijani      | Azərbaycanca       | Latn
ba  | Bashkir          | Башҡортса          | Cyrl
be  | Belarusian       | Беларуская         | Cyrl
bg  | Bulgarian        | Български          | Cyrl
bn  | Bengali          | বাংলা               | Beng
br  | Breton           | Brezhoneg          | Latn
bs  | Bosnian          | Bosanski           | Latn
ca  | Catalan          | Català             | Latn
ceb | Cebuano          | Cebuano            | Latn
cs  | Czech            | Čeština            | Latn
cy  | Welsh            | Cymraeg            | Latn
da  | Danish           | Dansk              | Latn
de  | German           | Deutsch            | Latn
el  | Greek            | Ελληνικά           | Grek
en  | English          | English            | Latn
es  | Spanish          | Español            | Latn
et  | Estonian         | Eesti              | Latn
fa  | Persian          | فارسی              | Arab
ff  | Fulah            | Fulfulde           | Latn
fi  | Finnish          | Suomi              | Latn
fr  | French           | Français           | Latn
fy  | Western Frisian  | Frysk              | Latn
ga  | Irish            | Gaeilge            | Latn
gd  | Gaelic           | Gàidhlig           | Latn
gl  | Galician         | Galego             | Latn
gu  | Gujarati         | ગુજરાતી              | Gujr
ha  | Hausa            | Hausa              | Latn
he  | Hebrew           | עברית              | Hebr
hi  | Hindi            | हिन्दी               | Deva
hr  | Croatian         | Hrvatski           | Latn
ht  | Haitian          | Kreyòl ayisyen     | Latn
hu  | Hungarian        | Magyar             | Latn
hy  | Armenian         | Հայերեն            | Armn
id  | Indonesian       | Bahasa Indonesia   | Latn
ig  | Igbo             | Igbo               | Latn
ilo | Iloko            | Ilokano            | Latn
is  | Icelandic        | Íslenska           | Latn
it  | Italian          | Italiano           | Latn
ja  | Japanese         | 日本語              | Jpan
jv  | Javanese         | Basa Jawa          | Latn
ka  | Georgian         | ქართული            | Geor
kk  | Kazakh           | Қазақша            | Cyrl
km  | Central Khmer    | ខ្មែរ                | Khmr
kn  | Kannada          | ಕನ್ನಡ               | Knda
ko  | Korean           | 한국어              | Kore
lb  | Luxembourgish    | Lëtzebuergesch     | Latn
lg  | Ganda            | Luganda            | Latn
ln  | Lingala          | Lingála            | Latn
lo  | Lao              | ລາວ                | Laoo
lt  | Lithuanian       | Lietuvių           | Latn
lv  | Latvian          | Latviešu           | Latn
mg  | Malagasy         | Malagasy           | Latn
mk  | Macedonian       | Македонски         | Cyrl
ml  | Malayalam        | മലയാളം             | Mlym
mn  | Mongolian        | Монгол             | Cyrl
mr  | Marathi          | मराठी               | Deva
ms  | Malay            | Bahasa Melayu      | Latn
my  | Burmese          | မြန်မာ               | Mymr
ne  | Nepali           | नेपाली               | Deva
nl  | Dutch            | Nederlands         | Latn
no  | Norwegian        | Norsk              | Latn
oc  | Occitan          | Occitan            | Latn
or  | Odia             | ଓଡ଼ିଆ                | Orya
pa  | Punjabi          | ਪੰਜਾਬੀ               | Guru
pl  | Polish           | Polski             | Latn
ps  | Pashto           | پښتو               | Arab
pt  | Portuguese       | Português          | Latn
ro  | Romanian         | Română             | Latn
ru  | Russian          | Русский            | Cyrl
sd  | Sindhi           | سنڌي               | Arab
si  | Sinhala          | සිංහල              | Sinh
sk  | Slovak           | Slovenčina         | Latn
sl  | Slovenian        | Slovenščina        | Latn
sn  | Shona            | chiShona           | Latn
so  | Somali           | Soomaali           | Latn
sq  | Albanian         | Shqip              | Latn
sr  | Serbian          | Српски             | Cyrl,Latn
su  | Sundanese        | Basa Sunda         | Latn
sv  | Swedish          | Svenska            | Latn
sw  | Swahili          | Kiswahili          | Latn
ta  | Tamil            | தமிழ்               | Taml
te  | Telugu           | తెలుగు              | Telu
th  | Thai             | ไทย                | Thai
tl  | Tagalog          | Tagalog            | Latn
tr  | Turkish          | Türkçe             | Latn
uk  | Ukrainian        | Українська         | Cyrl
ur  | Urdu             | اردو               | Arab
uz  | Uzbek            | Oʻzbekcha          | Latn
vi  | Vietnamese       | Tiếng Việt         | Latn
wo  | Wolof            | Wolof              | Latn
xh  | Xhosa            | isiXhosa           | Latn
yi  | Yiddish          | ייִדיש              | Hebr
yo  | Yoruba           | Yorùbá             | Latn
zh  | Chinese          | 中文               | Hani
zu  | Zulu             | isiZulu            | Latn
"""

# Regional variants that deserve their own label.
_VARIANTS = """
pt-BR | Portuguese (Brazil)    | Português (Brasil) | Latn
pt-PT | Portuguese (Portugal)  | Português (Portugal) | Latn
zh-CN | Chinese (Simplified)   | 简体中文            | Hani
zh-TW | Chinese (Traditional)  | 繁體中文            | Hani
es-MX | Spanish (Mexico)       | Español (México)   | Latn
fr-CA | French (Canada)        | Français (Canada)  | Latn
"""


def _parse(table: str) -> Dict[str, Language]:
    out = {}
    for row in table.strip().splitlines():
        code, name, native, scripts = (c.strip() for c in row.split("|"))
        out[code] = Language(code, name, native, tuple(scripts.split(",")))
    return out


LANGUAGES: Dict[str, Language] = {**_parse(_TABLE), **_parse(_VARIANTS)}

# Backwards-compatible mapping of code -> English name.
lang_codes: Dict[str, str] = {code: lang.name for code, lang in LANGUAGES.items()}

_CODE_RE = re.compile(r"^([a-zA-Z]{2,3})(?:[-_]([a-zA-Z]{2}|[a-zA-Z]{4}))?$")


def normalize_code(code: str) -> Optional[str]:
    """
    Normalizes a language code (``PT_br`` -> ``pt-BR``) and returns ``None`` if
    it is not a code we know about. Unknown regions of a known base language
    (``de-AT``) are accepted.
    """
    m = _CODE_RE.match(code.strip())
    if not m:
        return None
    base = m.group(1).lower()
    region = m.group(2)
    if base not in LANGUAGES:
        return None
    if not region:
        return base
    region = region.upper() if len(region) == 2 else region.title()
    return f"{base}-{region}"


def get_language(code: str) -> Language:
    """Returns metadata for a (normalized) code, falling back to its base language."""
    if code in LANGUAGES:
        return LANGUAGES[code]
    base = code.split("-")[0]
    if base in LANGUAGES:
        lang = LANGUAGES[base]
        return Language(
            code, f"{lang.name} ({code})", f"{lang.native} ({code})", lang.scripts
        )
    return Language(code, code, code, ("Latn",))
