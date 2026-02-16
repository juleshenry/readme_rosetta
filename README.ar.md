# 🗿 README Rosetta 

**README Rosetta** هو أداة اتوماتيكية قوية تهدف إلى ترجمة تعليماتك إلى لغات متعددة باستخدام المحاكي المحلية للLMIA через [Ollama](https://ollama.ai/). تمنح لك هذه الأداة صلاحيةProject accessibility العالمية بينما تحافظ على صيغة الماركดาว وتركيبه الأمين.
## 

قصيدة Rosetta تعمل على تحويل مشروع جهاذك الدولي مع الأقل إمكانيات.

- ** الدعم المultilanguage:** ترجم `README.md` إلى dozens من اللغاتsimultaneously .
- ** طريقة التبويب (قبل الدفعة):** يضيف automatically tab "stone" (table) في أعلى README، allowing users to quickly switch بين languages.
- ** القimes Modes:**
    - ** mode Split (Default):** generates separate files (e.g., `README.es.md`, `README.fr.md`) for clean project structure.
    - ** Unified Mode (`--no-split`):** adds all translations to main `README.md` file, separated by HTML comments.
- ** الحفاظ على الوسائط الماركداون:** intelligently handles headers، lists، and code blocks to ensure the translated output remains functional and well-formatted.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```
##🛠 
البرمجية الحرفية للترتيب الإجرائي (CLI) 
تصميمها يتضمن البهجة و القوة.
###instalation
```bash
pip install readme-rosetta
```

*ملاحظة: تتطلب برمجية Ollama being installed and running on your system.*
 
    [Ollama](https://ollama.ai/)
### خيارات العالمية

| إجراء | التوصيف | القيمة الأصلية |
| :--- | :--- | :--- |
| `path` | مسار الملفsource أو إدارة المشروع. | `README.md` |
| `--langs` | قائمة Kod الهدف اللغة (مثل `es fr de`). | `[]` |
| `--src-lang` | kod لغة المصدر. | `en` |
| `--model` | ID môdel Ollama الذي يُستخدم. | `llama3.2` |
| `--readme` | مسار الملف الرئيسي README الذي يُكتتب. | `README.md` |
| `--no-split` | إضافة ترجمات إلى ملف واحد. | `False` |
| `--dry-run` | simulationprocess_without_writing_files. | `False` |
| `--verbose` | تحفيز التسجيل التفصيلي لعملية الدعوة. | `False` |
## 📚 ت integration Sphinx 

توسع documentation لlevels المهنيين مع الدعم الت tựمية Sphinx i18n .

عند التنفيذ مع لامبة `--sphinx` :
1.  **تحصين Sphinx:** تأسيس目录 `docs/` إذا لم يكن موجوداً.
2.  **التنويع الإتجاهي :** تحديث `conf.py` بالضريبة `locale_dirs` و`gettext` .
3.  **استخراج الكلمات التي يمكن الترجمة :** التنفيذ `gettext` để العثور على جميع الكلمات المترجمة في التدوين.
4.  **التحويل الملفات PO :** استخدامه LLM لتطوير translation `.po` الملفات، مع احترام syntaxa Sphinx المعنية مثل `:role:` أو `.. directive::`.
5.  **إنتاج الطريقة الإلكترونية :** إنتاج الطريقة الإلكترونية المحلية للكلام لكل لغة الهدف.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 دعم GitBook

سهل الحفظ للتنسيق المULTILINGUAL لgitbook.

تعمل الرمز `--gitbook` على إنتاج ملف `SUMMARY.md` يحدد مappingsของคتابات README الترجمة إلى بنية متوافق مع النavigation في gitbook.

- **المتابعة tựياً:** تربط الاقتراح introducing معREADME الرئيسي و tạo نقاط الملفات الإضافية لكل نسخة الترجمة.
- **الاسمات اللغوية:** تعمل على تحديد الكود اللغوي (كمثل `es`) في إسماء الكامل (كما `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ الإعدادات 

حفظ الوقت بتحديد تفضيلات المشروع الأساسية في `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
# 🚨 خ้อผิดพลาด & giới hạnในการทำงาน

ترجمةโดยอัตโนมัตสử dụng LLM มีพลัง แต่สามารถนำถึงรอยขีดข่วนของการออกแบบบางประเภทได้ โดยเฉพาะในสภาพแวดล้อม Sphinx/RSTที่ซับซ้อน
### المشكلات الشائعة
- **انفصال الكتابة المضغوطة:** يمكن أن يؤثر على LLM التأنيث في  `` `` `` or `` `` ` `مفتوحة.
- **طول الرسومات:** إذا قام المॉडل بحرفي البُتون (`**`) على العلاقة، قد تتغير أطوال السيرفرة للاطلاق.
### script cleanup
نحن نقدم سكريتUtilities لاكتشاف وتعديل الأخطاء المشتركة للترجمة في ملفات `.po`. إذا تم تعديل الترجمة، سوف يستخدم Sphinx الترجمة الإصدارية الإنجليزية الأصلية للestring.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*ملاحظة: دائماً تحقق من بنية المستندات. في حين يعتمد Rosetta على الاكتمال، إلا ان تعديلات الملفات المحلية المزودة ل `.po` Sometimes आवशكيتها للدокументация العالي.
```
# license

هذا المشروع يرخص bajo лиسنس ميت (MIT) - انظر الملف `[LICENSE]`  لتفاصيل.
```
