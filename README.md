![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

PyTorch is a Python package that provides two high-level features:
- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy, and Cython to extend PyTorch when needed.

Our trunk health (Continuous Integration signals) can be found at [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

<!-- toc -->

- [More About PyTorch](#more-about-pytorch)
  - [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
  - [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
  - [Python First](#python-first)
  - [Imperative Experiences](#imperative-experiences)
  - [Fast and Lean](#fast-and-lean)
  - [Extensions Without Pain](#extensions-without-pain)
- [Installation](#installation)
  - [Binaries](#binaries)
    - [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
  - [From Source](#from-source)
    - [Prerequisites](#prerequisites)
    - [Install Dependencies](#install-dependencies)
    - [Get the PyTorch Source](#get-the-pytorch-source)
    - [Install PyTorch](#install-pytorch)
      - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Docker Image](#docker-image)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image yourself](#building-the-image-yourself)
  - [Building the Documentation](#building-the-documentation)
  - [Previous Versions](#previous-versions)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Communication](#communication)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

At a granular level, PyTorch is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | A Tensor library like NumPy, with strong GPU support |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | A neural networks library deeply integrated with autograd designed for maximum flexibility |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader and other utility functions for convenience |

Usually, PyTorch is used either as:

- A replacement for NumPy to use the power of GPUs.
- A deep learning research platform that provides maximum flexibility and speed.

Elaborating Further:

### A GPU-Ready Tensor Library

If you use NumPy, then you have used Tensors (a.k.a. ndarray).

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the
computation by a huge amount.

We provide a wide variety of tensor routines to accelerate and fit your scientific computation needs
such as slicing, indexing, mathematical operations, linear algebra, reductions.
And they are fast!

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch has a unique way of building neural networks: using and replaying a tape recorder.

Most frameworks such as TensorFlow, Theano, Caffe, and CNTK have a static view of the world.
One has to build a neural network and reuse the same structure again and again.
Changing the way the network behaves means that one has to start from scratch.

With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to
change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes
from several research papers on this topic, as well as current and past work such as
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date.
You get the best of speed and flexibility for your crazy research.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is not a Python binding into a monolithic C++ framework.
It is built to be deeply integrated into Python.
You can use it naturally like you would use [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
You can write your new neural network layers in Python itself, using your favorite libraries
and use packages such as [Cython](https://cython.org/) and [Numba](http://numba.pydata.org/).
Our goal is to not reinvent the wheel where appropriate.

### Imperative Experiences

PyTorch is designed to be intuitive, linear in thought, and easy to use.
When you execute a line of code, it gets executed. There isn't an asynchronous view of the world.
When you drop into a debugger or receive error messages and stack traces, understanding them is straightforward.
The stack trace points to exactly where your code was defined.
We hope you never spend hours debugging your code because of bad stack traces or asynchronous and opaque execution engines.

### Fast and Lean

PyTorch has minimal framework overhead. We integrate acceleration libraries
such as [Intel MKL](https://software.intel.com/mkl) and NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) to maximize speed.
At the core, its CPU and GPU Tensor and neural network backends
are mature and have been tested for years.

Hence, PyTorch is quite fast — whether you run small or large neural networks.

The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.
We've written custom memory allocators for the GPU to make sure that
your deep learning models are maximally memory efficient.
This enables you to train bigger deep learning models than before.

### Extensions Without Pain

Writing new neural network modules, or interfacing with PyTorch's Tensor API was designed to be straightforward
and with minimal abstractions.

You can write new neural network layers in Python using the torch API
[or your favorite NumPy-based libraries such as SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
No wrapper code needs to be written. You can see [a tutorial here](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [an example here](https://github.com/pytorch/extension-cpp).


## Installation

### Binaries
Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.


### From Source

#### Prerequisites
If you are installing from source, you will need:
- Python 3.8 or later (for Linux, Python 3.8.1+ is needed)
- A compiler that fully supports C++17, such as clang or gcc (especially for aarch64, gcc 9.4.0 or newer is required)

We highly recommend installing an [Anaconda](https://www.anaconda.com/download) environment. You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your Linux distro.

If you want to compile with CUDA support, [select a supported version of CUDA from our support matrix](https://pytorch.org/get-started/locally/), then install the following:
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 or above
- [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardware

If you want to disable CUDA support, export the environment variable `USE_CUDA=0`.
Other potentially useful environment variables may be found in `setup.py`.

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

If you want to compile with ROCm support, install
- [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation
- ROCm is currently supported only for Linux systems.

If you want to disable ROCm support, export the environment variable `USE_ROCM=0`.
Other potentially useful environment variables may be found in `setup.py`.

#### Install Dependencies

**Common**

```bash
conda install cmake ninja
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
pip install -r requirements.txt
```

**On Linux**

```bash
conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

# (optional) If using torch.compile with inductor/triton, install the matching version of triton
# Run from the pytorch directory after cloning
make triton
```

**On MacOS**

```bash
# Add this package on intel x86 processor machines only
conda install mkl mkl-include
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

**On Windows**

```bash
conda install mkl mkl-include
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

#### Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

#### Install PyTorch
**On Linux**

If you would like to compile PyTorch with [new C++ ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) enabled, then first run this command:
```bash
export _GLIBCXX_USE_CXX11_ABI=1
```

If you're compiling for AMD ROCm then first run this command:
```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

Install PyTorch
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

> _Aside:_ If you are using [Anaconda](https://www.anaconda.com/distribution/#download-section), you may experience an error caused by the linker:
>
> ```plaintext
> build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
> collect2: error: ld returned 1 exit status
> error: command 'g++' failed with exit status 1
> ```
>
> This is caused by `ld` from the Conda environment shadowing the system `ld`. You should use a newer version of Python that fixes this issue. The recommended Python version is 3.8.1+.

**On macOS**

```bash
python3 setup.py develop
```

**On Windows**

Choose Correct Visual Studio Version.

PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
Professional, or Community Editions. You can also install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
come with Visual Studio Code by default.

If you want to build legacy python code, please refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU-only builds**

In this mode PyTorch computations will run on your CPU, not your GPU

```cmd
conda activate
python setup.py develop
```

Note on OpenMP: The desired OpenMP implementation is Intel OpenMP (iomp). In order to link against iomp, you'll need to manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. The instruction [here](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) is an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

**CUDA based build**

In this mode PyTorch computations will leverage your GPU via CUDA for faster number crunching

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) is needed to build Pytorch with CUDA.
NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto an already installed CUDA run CUDA installation once again and check the corresponding checkbox.
Make sure that CUDA with Nsight Compute is installed after Visual Studio.

Currently, VS 2017 / 2019, and Ninja are supported as the generator of CMake. If `ninja.exe` is detected in `PATH`, then Ninja will be used as the default generator, otherwise, it will use VS 2017 / 2019.
<br/> If Ninja is selected as the generator, the latest MSVC will get selected as the underlying toolchain.

Additional libraries such as
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Please refer to the [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) to install them.

You can refer to the [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) script for some other environment variables configurations


```cmd
cmd

:: Set the environment variables after you have downloaded and unzipped the mkl package,
:: else CMake would throw an error as `Could NOT find OpenMP`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Read the content in the previous section carefully before you proceed.
:: [Optional] If you want to override the underlying toolset used by Ninja and Visual Studio with CUDA, please run the following script block.
:: "Visual Studio 2019 Developer Command Prompt" will be run automatically.
:: Make sure you have CMake >= 3.12 before you do this when you use the Visual Studio generator.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py develop

```

##### Adjust Build Options (Optional)

You can adjust the configuration of cmake variables optionally (without building first), by doing
the following. For example, adjusting the pre-detected directories for CuDNN or BLAS can be done
with such a step.

On Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

On macOS
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

You can also pull a pre-built docker image from Docker Hub and run with docker v19.03+

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Please note that PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g.
for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you
should increase shared memory size either with `--ipc=host` or `--shm-size` command line options to `nvidia-docker run`.

#### Building the image yourself

**NOTE:** Must be built with a docker version > 18.06

The `Dockerfile` is supplied to build images with CUDA 11.1 support and cuDNN v8.
You can pass `PYTHON_VERSION=x.y` make variable to specify which Python version is to be used by Miniconda, or leave it
unset to use the default.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

You can also pass the `CMAKE_VARS="..."` environment variable to specify additional CMake variables to be passed to CMake during the build.
See [setup.py](./setup.py) for the list of available variables.

```bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile
```

### Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```bash
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running `make <format>` from the
`docs/` folder. Run `make` to get a list of all available output formats.

If you get a katex error run `npm install katex`.  If it persists, try
`npm install -g katex`

> Note: if you installed `nodejs` with a different package manager (e.g.,
`conda`) then `npm` will probably install a version of `katex` that is not
compatible with your version of `nodejs` and doc builds will fail.
A combination of versions that is known to work is `node@6.13.1` and
`katex@0.13.18`. To install the latter with `npm` you can run
```npm install -g katex@0.13.18```

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found
on [our website](https://pytorch.org/previous-versions).


## Getting Started

Three-pointers to get you started:
- [Tutorials: get you started with understanding and using PyTorch](https://pytorch.org/tutorials/)
- [Examples: easy to understand PyTorch code across all domains](https://github.com/pytorch/examples)
- [The API Reference](https://pytorch.org/docs/)
- [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Resources

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Examples](https://github.com/pytorch/examples)
* [PyTorch Models](https://pytorch.org/hub/)
* [Intro to Deep Learning with PyTorch from Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Intro to Machine Learning with PyTorch from Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Deep Neural Networks with PyTorch from Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Communication
* Forums: Discuss implementations, research, etc. https://discuss.pytorch.org
* GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
* Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv
* Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch
* For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Typically, PyTorch has three minor releases a year. Please let us know if you encounter a bug by [filing an issue](https://github.com/pytorch/pytorch/issues).

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

To learn more about making a contribution to Pytorch, please see our [Contribution page](CONTRIBUTING.md). For more information about PyTorch releases, see [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project with several skillful engineers and researchers contributing to it.

PyTorch is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions coming from hundreds of talented individuals in various forms and means.
A non-exhaustive but growing list needs to mention: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch) with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.

## License

PyTorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.<!-- toc -->![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

Respuesta

PyTorch es un paquete Python que proporciona dos características de alto nivel:
- Computación de tensor (como NumPy) con fuerte aceleración GPU
- Redes neuronales profundas construidas en un sistema de autogrado basado en cintas

Usted puede reutilizar sus paquetes Python favoritos como NumPy, SciPy y Cython para ampliar PyTorch cuando sea necesario.

Nuestra salud del tronco (Señales de integración continua) se puede encontrar en [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

¡No!

- [Más sobre PyTorch](#more-about-pytorch)
- [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
- [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
- [Python First] (#python-first)
- [Experiencias Imperiales] (Experiencias-imperativas)
- [Fast and Lean] (#fast-and-lean)
- [Extensiones sin dolor]
- [Instalación](#instalación)
- [Binarios]
- [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
- [De Fuente](#de fuente)
- [Prerequisitos]
- [Install Dependencies](#install-dependencies)
- [Obtenga la fuente PyTorch] (#get-the-pytorch-source)
- [Install PyTorch](#install-pytorch)
- [Opciones de Construir Ajustar (Opcional)](#ajust-build-options-optional)
- [Imágen Docker](#docker-image)
- [Usando imágenes preconstruidas] (imagenes preconstruidas)
- [Construyendo la imagen usted mismo] (construyendo-la-image-yourself)
- [Construyendo la Documentación] (construyendo la documentación)
- [ Versiones anteriores] (reversiones anteriores)
- [Empezar]
- [Recursos]
- [Comunicación](#comunicaciones)
- [Releases and Contributing](#releases-and- contributed)
- [El equipo](#el equipo)
- [License](#license)

- Tocstop...

## More About PyTorch

[Aprenda los fundamentos de PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

A nivel granular, PyTorch es una biblioteca que consta de los siguientes componentes:

Silencio Componente Silencio Descripción Silencio
Silencio...
[**torch**](https://pytorch.org/docs/stable/torch.html) ← Una biblioteca de Tensor como NumPy, con fuerte apoyo de GPU
TEN [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)
[**torch.jit**](https://pytorch.org/docs/stable/jit.html)
TEN [**torch.nn**](https://pytorch.org/docs/stable/nn.html) TEN A neural networks library deeply integrated with autograd designed for maximum flexibility TEN
(https://pytorch.org/docs/stable/multiprocessing.html) TEN Python multiprocesamiento, pero con la memoria mágica compartir Tensors a través de procesos. Útil para la carga de datos y entrenamiento de Hogwild
[**torch.utils**](https://pytorch.org/docs/stable/data.html) TEN DataLoader y otras funciones de utilidad para conveniencia TEN

Por lo general, PyTorch es utilizado como:
- Un reemplazo para NumPy para utilizar el poder de las GPU.
- Una plataforma de investigación de aprendizaje profundo que proporciona máxima flexibilidad y velocidad.

Elaborating Further:

## A GPU-Ready Tensor Library

Si usas NumPy, entonces has utilizado Tensors (a.k.a. ndarray).

![Ilustración del tensor](./docs/source/_static/img/tensor_illustration.png)

PyTorch proporciona Tensores que pueden vivir en la CPU o en la GPU y acelera el
computación por una gran cantidad.

Ofrecemos una amplia variedad de rutinas de tensor para acelerar y adaptarse a sus necesidades de computación científica
como slicing, indexación, operaciones matemáticas, álgebra lineal, reducciones.
¡Y son rápidos!

### Dynamic Neural Networks: Tape-Based Autograd
PyTorch tiene una forma única de construir redes neuronales: usar y reproducir un grabador de cinta.

La mayoría de los marcos como TensorFlow, Theano, Caffe y CNTK tienen una visión estática del mundo.
Uno tiene que construir una red neuronal y reutilizar la misma estructura una y otra vez.
Cambiar la forma en que se comporta la red significa que hay que empezar desde cero.

Con PyTorch, utilizamos una técnica llamada autodiferenciación de movimiento inverso, que le permite
cambiar la forma en que su red se comporta arbitrariamente con cero retraso o sobrecabeza. Nuestra inspiración viene
de varios documentos de investigación sobre este tema, así como trabajos actuales y pasados, como
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

Aunque esta técnica no es única para PyTorch, es una de las implementaciones más rápidas hasta la fecha.
Usted obtiene la mejor velocidad y flexibilidad para su investigación loca.

(https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

## Python First
PyTorch no es un Python encuadernado en un marco C++ monolítico.
Se construye para estar profundamente integrado en Python.
Puede utilizarlo naturalmente como lo haría [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
Puedes escribir tus nuevas capas de red neuronales en Python, usando tus bibliotecas favoritas
y utilizar paquetes como [Cython](https://cython.org/) y [Numba](http://numba.pydata.org/).
Nuestro objetivo es no reinventar la rueda cuando sea apropiado.

#### Imperative Experiences

PyTorch está diseñado para ser intuitivo, lineal en el pensamiento, y fácil de usar.
Cuando ejecutas una línea de código, se ejecuta. No hay una visión asincrónica del mundo.
Cuando usted cae en un depurador o recibe mensajes de error y apilar rastros, entenderlos es sencillo.
El rastro de la pila indica exactamente dónde se definió su código.
Esperamos que nunca pases horas depurando tu código debido a los malos rastros o motores de ejecución asincrónica y opaca.

## Fast and Lean

PyTorch tiene una sobrecarga de marco mínima. Integramos las bibliotecas de aceleración
como [Intel MKL](https://software.intel.com/mkl) y NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl))))) para maximizar la velocidad.
En el núcleo, su CPU y GPU Tensor y redes neuronales backends
son maduros y han sido probados durante años.

Por lo tanto, PyTorch es bastante rápido, ya sea que se ejecutan redes neuronales pequeñas o grandes.

El uso de memoria en PyTorch es extremadamente eficiente en comparación con la antorcha o algunas de las alternativas.
Hemos escrito adiestradores de memoria personalizados para la GPU para asegurarse de que
tus modelos de aprendizaje profundo son maximalmente eficientes en la memoria.
Esto le permite entrenar modelos de aprendizaje profundo más grandes que antes.

## Extensiones sin dolor

Escribir nuevos módulos de red neuronales, o interactuar con la API de Tensor de PyTorch fue diseñado para ser directo
y con abstracciones mínimas.

Puede escribir nuevas capas de red neuronales en Python usando la API de antorcha
[o sus bibliotecas basadas en NumPy favoritas como SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

Si desea escribir sus capas en C/C+++, proporcionamos una API de extensión conveniente que es eficiente y con caldera mínima.
No es necesario escribir ningún código de envoltura. Puede ver [un tutorial aquí](https://pytorch.org/tutorials/advanced/cpp_extension.html) y [un ejemplo aquí](https://github.com/pytorch/extension-cpp).
## Instalación

### Binaries
Los comandos para instalar binarios a través de Conda o ruedas pip están en nuestro sitio web: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.


### From Source

##### Prerequisites
Si usted está instalando de la fuente, usted necesitará:
- Python 3.8 o posterior (para Linux, Python 3.8.1+ es necesario)
- Un compilador que apoya plenamente C+17, como clang o gcc (especialmente para aarch64, gcc 9.4.0 o más nuevo es necesario)

Recomendamos encarecidamente la instalación de un entorno [Anaconda](https://www.anaconda.com/download). Tendrás una biblioteca BLAS de alta calidad (MKL) y obtendrás versiones de dependencia controladas independientemente de tu distro Linux.

Si desea compilar con soporte CUDA, [seleccione una versión soportada de CUDA de nuestra matriz de soporte](https://pytorch.org/get-started/locally/), instale lo siguiente:
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 o más
- [Compiler](https://gist.github.com/ax3l/9489132) compatible con CUDA

Nota: Usted podría referirse a la matriz de soporte [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) para las versiones de cuDNN con las diversas versiones compatibles de CUDA, controlador CUDA y hardware NVIDIA
Si desea desactivar el soporte CUDA, exporte la variable de entorno `USE_CUDA=0`.
Otras variables ambientales potencialmente útiles se pueden encontrar en 'setup.py'.

Si usted está construyendo para las plataformas Jetson de NVIDIA (Jetson Nano, TX1, TX2, AGX Xavier), Instrucciones para instalar PyTorch para Jetson Nano son [disponibles aquí](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano

Si desea compilar con soporte ROCm, instale
- [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 y por encima de la instalación
- ROCm actualmente es compatible sólo para sistemas Linux.

Si desea desactivar el soporte ROCm, exporte la variable de entorno `USE_ROCM=0`.
Otras variables ambientales potencialmente útiles se pueden encontrar en 'setup.py'.

#### Install Dependencies

*Common*

``bash
conda install cmake ninja
# Ejecute este comando del directorio PyTorch después de la clonación del código fuente utilizando la sección “Obtener la Fuente PyTorch” abajo
pip install -r requisitos. Txt
`` `

*En Linux*

``bash
conda install mkl mkl-include
# CUDA only: Añadir el soporte LAPACK para la GPU si es necesario
conda install -c pytorch magma-cuda110 # o el magma-cuda* que coincide con su versión CUDA en https://anaconda.org/pytorch/repo

# (opcional) Si usa antorcha. compilar con ductor/tritón, instalar la versión concordante de tritón
# Corre desde el directorio pytorch después de la clonación
hacer tritón
`` `

En MacOS

``bash
# Añadir este paquete en las máquinas procesadoras intel x86 solamente
conda install mkl mkl-include
# Añadir estos paquetes si la antorcha. se necesita
conda install pkg-config libuv
`` `

**En Windows**

``bash
conda install mkl mkl-include
# Añadir estos paquetes si la antorcha. se necesita distribuir.
# El soporte de paquete distribuido en Windows es una función de prototipo y está sujeto a cambios.
conda install -c conda-forge libuv=1.39
`` `

################################################################################################################################################################################################################################################################ Obtener la fuente de PyTorch
``bash
git clone -recursivo https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
`` `

#### Install PyTorch
*En Linux*

Si desea compilar PyTorch con [nuevo C++ ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) habilitado, ejecute primero este comando:
``bash
export _GLIBCXX_USE_CX11_ABI=1
`` `

Si usted está compilando para AMD ROCm entonces primero ejecutar este comando:
``bash
# Sólo corre esto si estás compilando para ROCm
python tools/amd_build/build_amd.py
`` `

Instalar PyTorch
``bash
Exportación CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(que conda))/./"}
Python setup. py desarrollo
`` `

■ _Además:_ Si está utilizando [Anaconda](https://www.anaconda.com/distribution/#download-section), puede experimentar un error causado por el linker:
■
.
■ build/temp.linux-x86_64-3.7/torch/csrc/stub.o: archivo no reconocido: formato de archivo no reconocido
Ø collect2: error: ld devuelto 1 estado de salida
error >: comando 'g+' falló con el estado de salida 1
&quot; `
■
■ Esto es causado por `ld` desde el entorno de Conda que sombra el sistema `ld`. Usted debe utilizar una versión más nueva de Python que soluciona este problema. La versión recomendada de Python es 3.8.1+.

En macOS

``bash
python3 setup.py desarrollar
`` `

**En Windows**

Elija la versión correcta de Visual Studio.

PyTorch CI utiliza Visual C++ BuildTools, que vienen con Visual Studio Enterprise,
Ediciones profesionales o comunitarias. También puede instalar las herramientas de construcción desde
https://visualstudio.microsoft.com/visual-cpp-build-tools/. Las herramientas de construcción *no*
ven con Visual Studio Código por defecto.

Si desea construir un código de pitón legado, consulte [Construir el código hereditario y CUDA] (https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU solo construye**
En este modo las computaciones de PyTorch se ejecutarán en su CPU, no su GPU

``cmd
conda activate
Python setup. py desarrollo
`` `

Nota sobre OpenMP: La implementación de OpenMP deseada es Intel OpenMP (iomp). Para conectarse contra el iómp, tendrá que descargar manualmente la biblioteca y configurar el entorno de construcción mediante el tweaking `CMAKE_INCLUDE_PATH` y `LIB`. The instruction [here](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) is an example for setting up both MKL and Intel OpenMP. Sin estas configuraciones para CMake, se utilizará Microsoft Visual C OpenMP (vcomp).

**Construcción basada en la Dependencia* *

En este modo, las computaciones de PyTorch aprovecharán su GPU a través de CUDA para una reducción de número más rápida

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) es necesario construir Pytorch con CUDA.
NVTX es parte de la distribución CUDA, donde se llama "Nsight Compute". Para instalarlo en una instalación ya instalada CUDA run CUDA una vez más y comprobar la casilla correspondiente.
Asegúrese de que CUDA con Nsight Compute se instala después de Visual Studio.

Actualmente, VS 2017 / 2019, y Ninja son compatibles como el generador de CMake. Si `ninja.exe` se detecta en `PATH`, entonces Ninja será utilizado como el generador predeterminado, de lo contrario, utilizará VS 2017 / 2019.
Si Ninja es seleccionada como generador, el último MSVC será seleccionado como la cadena de herramientas subyacente.
Bibliotecas adicionales, como
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN o DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Sírvanse consultar el [instalación-ayuda](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) para instalarlos.

Puede consultar el script [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) para algunas otras configuraciones de variables ambientales


``cmd
cmd

:: Establecer las variables ambientales después de haber descargado y desactivado el paquete mkl,
:: Si no, CMake lanzaría un error como &quot; NO podría encontrar OpenMP &quot; .
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Leer el contenido en la sección anterior cuidadosamente antes de proceder.
:: [Opcional] Si desea anular el conjunto de herramientas subyacentes utilizado por Ninja y Visual Studio con CUDA, por favor ejecute el siguiente bloque de script.
:: "Visual Studio 2019 Developer Command Prompt" se ejecutará automáticamente.
:: Asegúrate de tener CMake 3.12 antes de hacer esto cuando utiliza el generador de Visual Studio.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
para /f "usebackq tokens=*" %i in (`"%ProgramaFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -versión [15^,17^) -productos * -más reciente -instalación propiedadPath`) llamen "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Opcional] Si desea anular el compilador de host CUDA
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

Python setup. py desarrollo

`` `

##### Ajuste las opciones de construcción (Opcional)

Puede ajustar la configuración de variables cmake opcionalmente (sin construir primero), haciendo
el siguiente. Por ejemplo, se puede ajustar los directorios predetectados para CuDNN o BLAS
con tal paso.

En Linux
``bash
Exportación CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(que conda))/./"}
Python setup. py build --cmake-only
ccmake construir # o cmake-gui construir
`` `

En macOS
``bash
Exportación CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(que conda))/./"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CX=clang++Python setup. py build --cmake-only
ccmake construir # o cmake-gui construir
`` `

### Docker Image

################################################################################################################################################################################################################################################################ Utilizando imágenes preconstruidas

También puede extraer una imagen de muelle preconstruida de Docker Hub y correr con docker v19.03+

``bash
Docker Run --Gpus all --rm -ti --ipc=host pytorch/pytorch:latest
`` `

Tenga en cuenta que PyTorch utiliza memoria compartida para compartir datos entre procesos, por lo que si se utiliza multiprocesamiento de antorcha (por ejemplo.
para cargadores de datos multithreaded) el tamaño de segmento de memoria compartido predeterminado con el que se ejecuta contenedor no es suficiente, y usted
debe aumentar el tamaño de la memoria compartida ya sea con '--ipc=host` o '--shm-size' opciones de línea de comandos para 'nvidia-docker run`.

################################################################################################################################################################################################################################################################ Construyendo la imagen usted mismo

**NOTE:** Debe construirse con una versión de docker 18.06

El `Dockerfile` se suministra para construir imágenes con soporte CUDA 11.1 y cuDNN v8.
Puede pasar `PYTHON_VERSION=x.y` hacer variable para especificar qué versión Python debe ser utilizado por Miniconda, o dejarlo
unset para usar el predeterminado.

``bash
Hacer -f docker. Makefile
# Las imágenes son etiquetadas como docker.io/${your_docker_username}/pytorch
`` `

También puede pasar la variable ambiente 'CMAKE_VARS="..."` para especificar variables adicionales de CMake que se pasarán a CMake durante la construcción.
Véase [setup.py](./setup.py) para la lista de variables disponibles.

``bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker. Makefile
`` `

## Building the Documentation

Para construir documentación en diversos formatos, necesitará [Sphinx](http://www.sphinx-doc.org) y el
leer el tema de losdocs.

``bash
cd docs/
pip install -r requisitos. Txt
`` `
A continuación, puede construir la documentación ejecutando `make &apos; formatot &apos; del
Carpeta 'docs/`. Ejecute `make` para obtener una lista de todos los formatos de salida disponibles.

Si usted consigue un error de katex ejecutar `npm instalar katex`. Si persiste, intente
`npm install -g katex `

■ Nota: si instalas `nodejs` con un administrador de paquetes diferente (por ejemplo,
`conda`) entonces `npm` probablemente instalará una versión de `katex` que no
compatible con su versión de 'nodejs' y doc builds fallará.
Una combinación de versiones que se sabe que funcionan es `node@6.13.1` y
`katex@0.13.18`. Para instalar este último con `npm` se puede ejecutar
```npm install -g katex@0.13.18````

## Anterior Versiones

Instrucciones de instalación y binarios para versiones anteriores de PyTorch se pueden encontrar
[nuestro sitio web](https://pytorch.org/previous-versions).
## Get Started

Tres puntos para empezar:
- [Tutoriales: empezar con la comprensión y el uso de PyTorch](https://pytorch.org/tutorials/)
- [Examples: easy to understand PyTorch code across all domains](https://github.com/pytorch/examples)
- [The API Reference](https://pytorch.org/docs/)
- [Glosario](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Resources

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Ejemplos](https://github.com/pytorch/examples)
* [PyTorch Models](https://pytorch.org/hub/)
* [Intro to Deep Learning with PyTorch from Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Intro to Machine Learning with PyTorch from Udacity] (https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Deep Neural Networks with PyTorch from Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Communication
* Foros: Discutir las implementaciones, la investigación, etc. https://discuss.pytorch.org
* GitHub Issues: Informes de errores, solicitudes de características, problemas de instalación, RFCs, pensamientos, etc.
* Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. Si usted es un principiante que busca ayuda, el medio primario es [PyTorch Forums](https://discuss.pytorch.org). Si usted necesita una invitación de slack, rellene este formulario: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Newsletter: No-noise, un boletín electrónico de una sola dirección con importantes anuncios sobre PyTorch. Puede registrarse aquí: https://eepurl.com/cbG0rv
* Página de Facebook: Anuncios importantes sobre PyTorch. https://www.facebook.com/pytorch
* Para las directrices de marca, visite nuestro sitio web en [pytorch.org](https://pytorch.org/)
## Releases and Contributing

Típicamente, PyTorch tiene tres versiones menores al año. Sírvase informarnos si se encuentra en un error [llenando un problema] (https://github.com/pytorch/pytorch/issues).

Agradecemos todas las contribuciones. Si usted está planeando contribuir de nuevo bug-fixes, por favor hágalo sin más discusión.

Si planea aportar nuevas características, funciones de utilidad o extensiones al núcleo, por favor, abra primero un problema y discuta la función con nosotros.
Enviar un PR sin discusión podría terminar resultando en un PR rechazado porque podríamos estar tomando el núcleo en una dirección diferente de la que usted podría estar consciente.

Para obtener más información sobre hacer una contribución a Pytorch, consulte nuestra [página de contribución] (CONTRIBUTING.md). Para obtener más información sobre las versiones de PyTorch, consulte [Página de publicación] (RELEASE.md).

## El Equipo

PyTorch es un proyecto impulsado por la comunidad con varios ingenieros y investigadores hábiles que lo contribuyen.

PyTorch se mantiene actualmente por [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions coming from hundreds of talented individuals in various forms and means.
Una lista no exhaustiva pero creciente necesita mencionar: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin

Nota: Este proyecto no está relacionado con [hughperkins/pytorch](https://github.com/hughperkins/pytorch) con el mismo nombre. Hugh es un valioso contribuyente a la comunidad de la antorcha y ha ayudado con muchas cosas a la antorcha y PyTorch.
## License

PyTorch tiene una licencia de estilo BSD, como se encuentra en el archivo [LICENSE](LICENSE).