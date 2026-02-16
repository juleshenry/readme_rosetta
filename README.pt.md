#  README Rosetta

**README Rosetta** é uma ferramenta poderosa de automação projetada para traduzir sua documentação em vários idiomas usando LLMs locais via [Ollama](https://ollama.ai/). Garante que seu projeto seja acessível a uma audiência global enquanto mantém a formaatação Markdown perfeita e estrutura do documento.

---

##  README Tradução

README Rosetta especializa-se em tornar seu projeto de GitHub internacional com pouco esforço.

- **Suporte a vários idiomas:** Traduza `README.md` em dezenas de idiomas simultaneamente.
- **Tabela de navegação:** Gera uma "piedra" de navegação ( tabela) automaticamente na parte superior do seu README, permitindo aos usuários saltar entre os idiomas com facilidade.
- **Modos flexíveis:**
    - **Modo partido (Padrão):** Cria arquivos separados (por exemplo, `README.es.md`, `README.fr.md`) para uma estrutura de projeto limpa.
    - **Modo unificado (`--no-split`):** Apenas os tradutores são adicionados ao arquivo principal `README.md`, separados por comentários HTML.
- **Preservação do Markdown:** Trata inteligentemente cabeçalhos, listas e blocos de código para garantir que o saída traduzida fique funcional e bem-formada.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

##  Interface de Linha de Comando (CLI)

A CLI é projetada para ser intuitiva e poderosa.

### Instalação

```bash
pip install readme-rosetta
```

*Nota: Requer [Ollama](https://ollama.ai/) ser instalado e rodando no seu sistema.*

### Opções Globais

| Opção | Descrição | Valor padrão |
| :--- | :--- | :--- |
| `path` | Caminho do arquivo fonte ou diretório do projeto. | `README.md` |
| `--langs` | Lista de códigos de idioma-alvo (por exemplo, `es fr de`). | `[]` |
| `--src-lang` | Código de idioma fonte. | `en` |
| `--model` | ID do modelo Ollama a usar. | `llama3.2` |
| `--readme` | Caminho para o arquivo principal do README de saída. | `README.md` |
| `--no-split` | Adicionar traduções em um único arquivo. | `False` |
| `--dry-run` | Simular o processo sem escrever arquivos. | `False` |
| `--verbose` | Ativar log detalhado para depuração. | `False` |

---

##  Integração com Sphinx

Aumente sua documentação profissionalmente com suporte automático de i18n da Sphinx.

Ao executar com a opção `--sphinx`, README Rosetta:
1.  **Inicializa Sphinx:** Cria um diretório `docs/` se não existir.
2.  **Auto- configura i18n:** Atualiza `conf.py` com as necessárias `locale_dirs` e `gettext` configurações.
3.  **Extrae Strings:** Roda `gettext` para encontrar todas as strings tradutíveis em sua documentação.
4.  **Traduzir PO Files:** Usa o LLM para traduzir os arquivos `.po`, preservando a sintaxe Sphinx-specific como `:role:` ou `.. directive::`.
5.  **Gerar HTML:** Gera builds HTML personalizados localizados para cada idioma-alvo.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

##  Suporte a GitBook

Mantenha facilmente um livro de Git em vários idiomas.

A opção `--gitbook` gera um arquivo `SUMMARY.md` que mapeia as suas traduções do README para uma estrutura compatível com a navegação da GitBook.

- **Link automático:** Ligue o Introdução ao seu principal README e crie itens de lista para cada versão traduzida.
- **Nomes de idioma:** Resolve os códigos de idioma (como `es`) em seus nomes completos (como `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

##  Configuração

Economize tempo definindo padrões para seu projeto em `pyproject.toml`:

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

##  Licença

Este projeto está licenciado sob a licença MIT - consulte o arquivo [LICENSE](LICENSE) para detalhes.
