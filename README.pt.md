# 🗿 README Rosetta

**README Rosetta** é uma ferramenta poderosa de automação projetada para traduzir sua documentação em várias línguas utilizando LLMs locais via [Ollama](https://ollama.ai/). Garante que seu projeto seja acessível a um público global enquanto mantém a formatação Markdown perfeita e estrutura do documento.
## 🌍 README Translation

README Rosetta especializa-se em tornar seu projeto do GitHub internacional com minimal esforço.

- **Suporte a Vários Idiomas:** Traduzir `README.md` para décadas de idiomas simultaneamente.
- **Tabela de Navegação:** Adicionar automaticamente uma "pedra de navegação" (tabela) na parte superior do README, permitindo que os usuários mude rapidamente entre idiomas.
- **Modos Flexíveis:
    - **Modo Partido (Padrão):** Gerar arquivos separados (por exemplo, `README.es.md`, `README.fr.md`) para uma estrutura de projeto limpa.
    - **Modo Unificado (`--no-split`):** Adicionar todas as traduções no arquivo `README.md` principal, separadas por comentários HTML.
- **Preservação de Markdown:** Tratar inteligentemente cabeçalhos, listas e blocos de código para garantir que o saída traduzida permaneça funcional e bem-formada.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## CLI Intuito
A interface de linha de comando é projetada para ser intuitiva e poderosa.
### Instalação

```bash
pip install readme-rosetta
```

*Nota: Requer [Ollama](https://ollama.ai/) ser instalado e executado em seu sistema.*
### Opções Globais

| Opção | Descrição | Padrão |
| :--- | :--- | :--- |
| `path` | Caminho do arquivo de origem ou diretório do projeto. | `README.md` |
| `--langs` | Lista de códigos de idioma-alvo (por exemplo, `es fr de`). | `[]` |
| `--src-lang` | Código de idioma fonte. | `en` |
| `--model` | ID do modelo Ollama a usar. | `llama3.2` |
| `--readme` | Caminho para o arquivo principal README de saída. | `README.md` |
| `--no-split` | Adicionar traduções para um único arquivo. | `False` |
| `--dry-run` | Simular o processo sem escrever arquivos. | `False` |
| `--verbose` | Habilitar log detalhado para depuração. | `False` |
## 📚 Integriação com Sphinx

Aumente a escala da documentação para níveis profissionais com suporte automático de i18n em Sphinx.

Quando você executa com a bandeira `--sphinx`, README Rosetta:
1.  **Inicia Sphinx:** Estabelece um diretório `docs/` se não existir.
2.  **Automatiza configuração de i18n:** Atualiza `conf.py` com as configurações necessárias de `locale_dirs` e `gettext`.
3.  **Extrae Strings:** Executa `gettext` para encontrar todas as strings transletáveis no seu manual de documentação.
4.  **Traduz PO Files:** Usa o LLM para traduzir arquivos `.po`, preservando a sintaxe específica da Sphinx como `:role:` ou `.. directive::`.
5.  **Cria HTML Localizado:** Gerencia automaticamente builds de HTML localizados para cada idioma-alvo.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 Suporte a GitBook

Mantenha facilmente um livro de GitBook em múltiplas línguas.

A bandeira `--gitbook` gera um arquivo `SUMMARY.md` que mapeia seus READMEs traduzidos em uma estrutura compatível com a navegação do GitBook.

- **Automatização de Ligação:** Liga a Introdução ao seu principal README e cria itens da lista para cada versão traduzida.
- **Nomes de Línguas:** Resolve automaticamente os códigos de língua (como `es`) em seus nomes completos (como `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## Configuração 

Defina seus padrões de projeto para economizar tempo em `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```

---
## Atenção: Desvios e Limitações

A tradução automática com LLM é poderosa, mas pode ocasionalmente introduzir artefatos de formato, especialmente em ambientes Sphinx/RST complexos.
### Problemas Comuns
- **Diferenças de Acentos:** LLMs podem falhar ao fechar uma string com `` ` `` or ` `` ` `
- **Comprimentos de Títulos:** Se o modelo adiciona negrito (`**`) a um título, a sublinha do Sphinx não mais se alinha com a duração do texto.
- **Hallucinações Estruturais:** O modelo pode tentar adicionar resumos próprios ou blocos de código "ajudosos" que não estão no original.
### Script de Limpeza
Forneça um script de utilidade para identificar e limpar erros comuns de tradução nos seus arquivos `.po`. Se a tradução for eliminada, Sphinx utilizará simplesmente o texto original em inglês para essa string.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Nota: Sempre verifique seus builds de documentação. Embora Rosetta busque perfeição, a correção manual dos arquivos `.po` localizados é às vezes necessária para documentações de alto estresse.*
## 📜 Licença

Este projeto está licenciado sob o License do MIT - consulte o arquivo [LICENSE](LICENSE) para detalhes.
