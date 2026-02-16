#  🗿 README Rosetta

**README Rosetta** bir güçlü otomatik araçdır. Dokumanızın multiple dil olarak çevirilmesi için yerel LLM'leri [Ollama](https://ollama.ai/) üzerinden kullanır. Bu sayede projeniz küresel bir audience'a erişilebilir hale gelirken perfekte Markdown formatı ve dokunma structure'yi korur.

---
## 🌍 README ÇIKIŞ

README Rosetta, GitHub projeto'nun international olmasını kolaylaştıran özelistikte uzmanlık yapar.

- **Çok dili Destekleme:** `README.md`'i dozens of dillerde одновno translate edebilirsiniz.
- **Navigasyon Tablo:** Kullanıcılar hızlıca between dilleri switch etmek için README'nin 上undayız bir navigasyon "kaya" (tablo) ekler.
- **Etkil Mode'lari:**
    - **Bölünmüş Mod (Orjinal):** Clean proje yapısı için separate files (e.g., `README.es.md`, `README.fr.md`) genererir.
    - **Birleşik Mod (`--no-split`):** Tüm çevirileri `README.md` dosyasına ekler, HTML komентlerle ayrılır.
- **Çoklu Yazı Tipi Koruma:** Başlıklar, listeler ve kod blokları gibi konularda intelligently handles ede, çeviri outputu funcional ve well-formatted kalır.
## 🔧 Satır Komut Sunucusu (CLI)

Satır komut sunucusunun diseñlendiği intuitive ve güçlü being expected.
### Kurulumu

```bash
pip install readme-rosetta
```

*Not: [Ollama](https://ollama.ai/)'u sisteminizde yükleyin ve运行 olmasını đòilu.*
### Globaller Ayarlar

| Ayar | Açıklama | Standart |
| :--- | :--- | :--- |
| `path` | Kaynaк dosyasının veya projenin yolunun location'u. | `README.md` |
| `--langs` | hedef dil kodları listesi (ör. `es fr de`). | `[]` |
| `--src-lang` | Kaynak language kodu. | `en` |
| `--model` | Ollama model ID'i kullanmak için. | `llama3.2` |
| `--readme` | Başlıк output README dosyasının yolunun location'u. | `README.md` |
| `--no-split` | Kaynaşarı translationleri tek bir dosyaya eklemek için. | `False` |
| `--dry-run` | Processu simule etmek ve dosyalara yazmak_without without yazmak için. | `False` |
| `--verbose` | Geliştirme için ayrıntılı logging'i enabled etmek için. | `False` |
## 📚 Sphinx Integrementsi

Sphinx'n dokumentasyonunu profesyonelli seviyeye ölçeklendirme ve otomatik Sphinx i18n destekü với otomatik rosetta integrasyonu.

İstișniye when `--sphinx` flag ile çalıştırılır:
1.  **İçerikleri Tanımlayıcı:** `docs/` directory'sini oluşturur jika yoktur.
2.  **I18n Otomatik Ayarlar:** `conf.py`'i güncelleyerek `locale_dirs` ve `gettext` ayarları ile ilgili tudo hazırlar.
3.  **Kelime Dizileri ÇıkArthur:** `gettext` ile todoocumentasyonunuzdaki tüm перекlıklı kelimeleri bulur.
4.  **PO Dosyasını Tradükleme:** LLM kullanarak `.po` dosyasını tradükleme yapılır, Sphinx-t specific syntax like `:role:` veya `.. directive::` gibi.
5.  **HTML İyileştirme:** Her hedef dili için otomatik olarak yerelize edilmiş HTML builds oluşturulur.
## 📖 GitBook Support

Easily maintain bir multitelifreliGitBook.

`--gitbook` bayrağı, çevirisinin READMEsini `SUMMARY.md` dosyasına mapping yapmasını sağlar ki bu structureyle GitBook'nin navigasyonuna uyumlu hale gelir.
- **Automatik Bağlantı:** Tanınmış bir introduksiyonu main README ile bağlantılı hale getirir ve her çeviriyi list item olaraً oluşturur.
- **Dil Adılar:** Dil kodlarına (`es` gibi)_AUTOMATIK olarak tam isimleri (`Spanish` gibi) olarak çözür.

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfigürasyon

Zaman kazandıran projenin standardlarını definitionelendi `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
### ⚠️ Hata Ayıkları & Sınırlamalar

Otomatik çeviriler using LLMs güçlü ama occasionaldir, özellikle karmaşık Sphinx/RST ortamlarında formatlardaki öznellikler girmekle birlikte.
### Common Issues
- **Mismatched Backticks:** LLMs might fail to close bir `` `` `` or `` `` `` string.
- **Header Lengths:** If bir LLM bolding (`**`) to a title, Sphinx underline may no longer match text length.
- **Structural Hallucinations:** The model might try to add own summaries or "helpful" code blocks that aren't in source.
### Temizlik Skripti
Rosetta tarafından önerilen bir utility skripti sunulmaktadır, votre `.po` dosyasında common çevirileri tanımlamak ve silmek için usedいます. Ayrıca, çevirilerin cleared oldugu durumlarda Sphinx, o stringin orijinal İngilizce metnine döndürür.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Not: Her zaman documentation buildsınızı kontrol edin. Rosetta'nin perfekt olmasa da, manual correction of localized `.po` dosyaları कभiden कभine high-stakes documentation için gereklidir.*
## 📜 Lisans

Bu proje MIT Lisansı altında yer almaktadır - detallesin [LICENSE](LICENSE) dosyasında befindir.
