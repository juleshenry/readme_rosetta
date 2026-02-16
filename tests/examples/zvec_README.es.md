<div align=" center">
  <picture>
    <source media="(prefiere esquema de colores: oscuro)" srcset="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_log_2.svg" />
    <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/logo/github_logo_1.svg" width="400" alt="logo de zvec" />
  </picture>
</div>

<p align="center">
  <a href="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_x64_docker_ci.yml/badge.svg?branch=main" alt="CI Linux x64"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/linux_arm64_docker_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/linux_arm64_docker_ci.yml/badge.svg?branch=main" alt="CI Linux ARM64"/></a>
  <a href="https://github.com/alibaba/zvec/actions/workflows/mac_arm64_ci.yml"><img src="https://github.com/alibaba/zvec/actions/workflows/mac_arm64_ci.yml/badge.svg?branch=main" alt="CI macOS ARM64"/></a>
  <br>
  <a href="https://codecov.io/github/alibaba/zvec"><img src="https://codecov.io/github/alibaba/zvec/graph/badge.svg?token=O81CT45B66" alt="Cobertura de código"/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/v/zvec.svg" alt="Lanzamiento en PyPI "/></a>
  <a href="https://pypi.org/project/zvec/"><img src="https://img.shields.io/pypi/pyversions/zvec.svg" alt="Versión de Python de zvec"/></a>
  <a href="https://github.com/alibaba/zvec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Licencia"/></a>
</p>

<p align="center">
  <a href="https://zvec.org/en/docs/quickstart/">🚀 <strong>Quickstart</strong> </a> |
  <a href="https://zvec.org/en/">🏠 <strong>Home</strong> </a> |
  <a href="https://zvec.org/en/docs/">📚 <strong>Docs</strong> </a> |
  <a href="https://zvec.org/en/docs/benchmarks/">📊 <strong>Benchmarks</strong> </a> |
  <a href="https://discord.gg/rKddFBBu9z">🎮 <strong>Discord</strong> </a> |
  <a href="https://x.com/zvec_ai">🐦 <strong>X (Twitter)</strong> </a>
</p>

**Zvec** es una base de datos vectorial de código abierto, en proceso — ligera, rápida y diseñada para insertarse directamente en aplicaciones. Construida sobre **Proxima** (el motor de búsqueda vectorial probado por Alibaba), entrega búsquedas de similitud de producción de grados bajos latencias, escalables con minimalidad de configuración.
## 💫 Características (Spanish)

- **Veloz como fuego**: Busca mil millones de vectores en milisegundos.
- **Sencillo y funciona sin problemas**: [Instalar](#-instalación) y comienza a buscar en segundos. No servidores, no configuración, ni complicaciones.
- **Vectores denso + esparcio**: Trabaja con tanto enlace embeddings como vectores esparcidos, con soporte nativo para solicitudes de múltiples vectores en una sola llamada.
- **Búsqueda híbrida**: Combinando la similitud semántica con los filtros estructurados para resultados precisos.
- **Corre en cualquier lugar**: Como biblioteca in-process, Zvec corre donde corra tu código — notebooks, servidores, herramientas de línea de comandos, incluso dispositivos de borde.
## 📦 Instalación
### [Python](https://pypi.org/project/zvec/)

**Requisitos**: Python 3.10 - 3.12
### [Node.js](https://www.npmjs.com/package/@zvec/zvec)

**@zvec/zvec**

```bash
npm install @zvec/zvec
```
### 🚀 Plataformas soportadas

- Linux (x86_64, ARM64)
- macOS (ARM64)
### 🛠️ Construyendo desde Fuente

Si prefiere construir Zvec desde la fuente, por favor revise el documento [Construyendo desde Fuente](https://zvec.org/en/docs/build/) en inglés.
## Ejemplo de un Minuto

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
## 📈 Desempeño a Escala

Zvec entrega una excelente velocidad y eficiencia, lo que la hace ideal para trabajos de carga pesada en producción.

<img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qps_10M.svg" width="800" alt="Benchmarkas de rendimiento de Zvec" />

Para obtener información detallada sobre el método de evaluación, configuraciones y resultados completos, por favor consulte nuestra [Documentación de Benchmarks](https://zvec.org/en/docs/benchmarks/).
## 🤝 Únete a Nuestra Comunidad

<div align="center">

Mantén actualizado y obtén soporte — escanear o hacer clic:

<table align="center" style="border-collapse: collapse; margin: 16px auto; width: 100%; max-width: 520px;">
  <tr>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">💬 DingTalk</div>
      <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/dingding.png" alt="Código QR de DingTalk" width="100" style="border-radius: 8px; border: 1px solid #ddd;">
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">📱 WeChat</div>
      <img src="https://zvec.oss-cn-hongkong.aliyuncs.com/qrcode/wechat.png" alt="Código QR de WeChat" width="100" style="border-radius: 8px; border: 1px solid #ddd;">
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">🎮 Discord</div>
      <a href="https://discord.gg/rKddFBBu9z" target="_blank" style="display: inline-block; width: 100px; height: 100px; background: #5865F2; border-radius: 8px; text-decoration: none; color: blanco; font-size: 12px; display: flex; align-items: center; justify-content: center; line-height: 1;">
        Unirse al servidor
      </a>
    </td>
    <td align="center" style="padding: 8px; width: 25%;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">🐦 X (Twitter)</div>
      <a href="https://x.com/zvec_ai" target="_blank" style="display: inline-block; width: 100px; height: 100px; background: #000; border-radius: 8px; text-decoration: none; color: blanco; font-size: 12px; display: flex; align-items: center; justify-content: center; line-height: 1;">
        Sigue a @zvec_ai
      </a>
    </td>
  </tr>
</table>

</div>
## ❤️ Contribuyendo

Nos complace y apreciamos las contribuciones de la comunidad. Ya sea que estés arreglando un bug, agregando una característica o mejorando la documentación, tu ayuda hace que Zvec sea mejor para todos.

Check out our [Contributing Guide](./CONTRIBUTING.md) to get started!
