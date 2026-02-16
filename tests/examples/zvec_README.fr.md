<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="logo zvec" />
  </picture>
</div>

<p align="center">
  <a href="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml/badge.svg?branch=main" alt="CI Linux x64"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/linux_arm64_docker_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_arm64_docker_ci.yml/badge.svg?branch=main" alt="CI Linux ARM64"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/mac_arm64_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/mac_arm64_ci.yml/badge.svg?branch=main" alt="CI macOS ARM64"/></a>
  <br>
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="Couverture de code"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="Lancement sur PyPI"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/pyversions/zvec.svg" alt="Versions de Python supportées"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Lisence Apache 2.0"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/en/docs/quickstart/">🚀 <strong>Préambule rapide</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>Demande d'aide</strong> </a> |
  <a href="https://zvec.org/en/docs/">📚 <strong>Dossiers de documentation</strong> </a> |
  <a href="https://zvec.org/en/docs/benchmarks/">📊 <strong>Benchmarks</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/zvec_ai">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** est une base de données vectorielle open source, exécutée en processus — légère, très rapide et conçue pour intégrer directement dans les applications. Construite sur **Proxima** (le moteur de recherche vectoriel de battle testé par Alibaba), elle offre des recherches d'approximation de production de qualité, basées sur latence faible et scalable sans grande configuration.

## 💫 Caractéristiques

- **Blazing Fast**: Recherche des milliards de vecteurs en millisecondes.
- **Simple, Juste Fonctionne**: [Installer](#-installation) et démarrer la recherche dans secondes. Aucun serveur, aucune configuration, pas de souci.
- **Vecteurs Denses + Épais**: Fonctionnent avec les vecteurs denses et les vecteurs épais, avec un support native pour des requêtes multi-vecteurs dans une seule appelle.
- **Recherche Hybride**: Combinez la similité sémantique avec les filtres structurés pour des résultats précis.
- **Exécution Partout**: Comme une bibliothèque en processus, Zvec exécute partout où votre code exécute — notebooks, serveurs, outils CLI, ou même dispositifs de bord.
## 📦 Installation

Pour installer la bibliothèque Python requise, vous pouvez utiliser pip :

```bash
pip install python-requirement
```

Cette commande installe les dépendances nécessaires pour la bibliothèque.
### [Python](https://pypi.org/project/zvec/)

**Requises** : Python 3.10 - 3.12

```bash
pip install zvec
```
### [Node.js](https://www.npmjs.com/package/@zvec/zvec)

```bash
npm install @zvec/zvec
``` 

*   Un outil Node.js nommé `zvec` est disponible sur npm.
*   Ce package permet d'analyser des données de haute performance.
*   Il est adapté à la gestion de grande quantité de données.
### 🎯 Platesformes supportées

* Linux (x86_64, ARM64)
* macOS (ARM64)
### 🛠️ Construire depuis la source

Si vous préférez construire Zvec à partir de la source, veuillez consulter la [guide de construction depuis la source](https://zvec.org/fr/docs/build/).
## Exemple de 1 minute

```python
import zvec

# Define collection schema
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# Create collection
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# Insert documents
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

# Search by vector similarity
results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# Results: list of {'id': str, 'score': float, ...}, sorted by relevance
print(results)
```
## 📈 Performance à l'échelle

Zvec offre une vitesse et une efficacité exceptionnelles, la rendant idéale pour les charges de travail exigeantes.

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Études de performances de Zvec" />

Pour obtenir une approfondie vue sur la méthode de benchmark, les configurations et les résultats complets, veuillez consulter notre [Documentation des benchmarks](https://zvec.org/en/docs/benchmarks/).
## 🤝 Rejoignez Notre Communauté

<div align="center">

Restez à jour et obtenez de l'aide — scannez ou cliquez :

<table align="center" style="border-collapse: collapse; margin: 16px auto; width: 100%; max-width: 520px;">
  <tr>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">💬 DingTalk</div>
      <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" alt="DingTalk QR Code" width="100" style="border-radius: 8px; border: 1px solid #ddd;">
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">📱 WeChat</div>
      <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png" alt="WeChat QR Code" width="100" style="border-radius: 8px; border: 1px solid #ddd;">
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">🎮 Discord</div>
      <a href="https://discord.gg/rKddFBBu9z" target="_blank" style="display: inline-block; width: 100px; height: 100px; background: #5865F2; border-radius: 8px; text-decoration: none; color: white; font-size: 12px; display: flex; align-items: center; justify-content: center; line-height: 1;">
        Rejoindre le serveur
      </a>
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">🐦 X (Twitter)</div>
      <a href="https://x.com/zvec_ai" target="_blank" style="display: inline-block; width: 100px; height: 100px; background: #000; border-radius: 8px; text-decoration: none; color: white; font-size: 12px; display: flex; align-items: center; justify-content: center; line-height: 1;">
        Suivez @zvec_ai
      </a>
    </td>
  </tr>
</table>

</div>
## 🎉 Contribuant

Nous accueillons et remercions les contributions de la communauté ! Quels que soient le bug à corriger, la fonctionnalité à ajouter ou la documentation à améliorer, votre aide rend Zvec plus agréable pour tout le monde.

Consultez notre [Guide contribuant](./CONTRIBUTING.md) pour commencer!
