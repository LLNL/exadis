# ExaDiS

ExaDiS version 0.1

ExaDiS (Exascale Dislocation Simulator) is a set of software modules written to enable numerical simulations of large groups of moving and interacting dislocations, line defects in crystals responsible for crystal plasticity. By tracking time evolution of sufficiently large dislocation ensembles, ExaDiS predicts plasticity response and plastic strength of crystalline materials.

ExaDiS is built around a portable library of core functions for Discrete Dislocation Dynamics (DDD) method specifically written to perform efficient computations on new HPC architectures (e.g. GPUs). Simulations can be driven through the C++ or python interfaces. The python binding module is designed to be interfaced with the [OpenDiS](https://github.com/OpenDiS/OpenDiS) framework, for which ExaDiS is the HPC core engine.

Note: Although ExaDiS is a fully functional code, it is currently under active development and is subject to frequent updates and bug fixes. There is no guarantee of stability and one should expect occasional breaking changes to the code.


## Quick start

ExaDiS is part of the [OpenDiS](https://github.com/OpenDiS/OpenDiS) framework, where this repository is included as a submodule. To obtain the code as part of OpenDiS (preferred way), follow the instructions at the [OpenDiS documentation](https://opendis.github.io/OpenDiS/installation/index.html).


## ExaDiS as a standalone

Alternatively, the code can be obtained as a standalone. ExaDiS is implemented using the [Kokkos](https://kokkos.org) framework and built using the CMake build system. A typical standalone installation of the code follows the steps below:

* Step 1: Clone this repository and submodules
```
git clone --recursive https://github.com/LLNL/exadis.git
cd exadis
```
Alternatively, you can use the following commands to achieve the same
```
git clone https://github.com/LLNL/exadis.git
cd exadis
git submodule init
git submodule update
```

* Step 2: Configure the build for your system by passing build options to the `configure.sh` script. (See list of options in the Build Options section below.)
    * Example: default build with `SERIAL` and `OPENMP` backends
    ```
    ./configure.sh
    ```
    * Example: build with `CUDA` backend and device architecture `VOLTA70`
    ```
    ./configure.sh \
        -DKokkos_ENABLE_CUDA=On \
        -DKokkos_ENABLE_CUDA_LAMBDA=On \
        -DKokkos_ARCH_VOLTA70=On  
    ```
    * You can also use pre-defined build options and/or create your own build options by setting the options in files `cmake/sys.cmake.<mysystem>`, and then passing build argument `-DSYS=<mysystem>`. E.g., to build for `SYS=lassen` (i.e. using options set in file `cmake/sys.cmake.lassen`):
    ```
    ./configure.sh -DSYS=lassen
    ```

* Step 3: Build the code
```
cmake --build build -j8
```
Note: building for GPU (e.g. with `nvcc` or `hipcc`) may be pretty slow, please be patient! 
For additional building options and troubleshooting see section Detailed build instructions below.

* Step 4: Test your installation by running an example (assuming `-DEXADIS_PYTHON_BINDING=On`)
```
cd examples/02_frank_read_src
python test_frank_read_src.py
```

## Detailed build instructions

### Dependencies

* Kokkos:
    * ExaDiS is implemented using the Kokkos framework. Kokkos is included as a submodule to the repository and will be automatically cloned to the `kokkos/` folder when using the git submodule commands or cloning with the `--recursive` option (see Step 1 of Quick Start section). By default, Kokkos will be built in-tree while building ExaDiS. ExaDiS will be compiled for the backend(s) selected to build Kokkos. For instance, if Kokkos is built to run on GPUs (e.g. with build option `-DKokkos_ENABLE_CUDA=ON`), then ExaDiS will be compiled to run on GPUs. If a prior Kokkos installation exists on the machine, its installation path can be provided with ExaDiS build option `-DKokkos_ROOT`, in which case Kokkos will not be built in-tree. Instructions on how to configure/install Kokkos are found at https://github.com/kokkos/kokkos.
    
* FFT libraries
    * ExaDiS uses FFT libraries to compute long-range elastic interactions. To compile ExaDiS without this module (e.g. if no FFT library is available) use build option `-DEXADIS_FFT=Off`. Otherwise (default), different FFT libraries are invoked depending on the target backend:
        * Serial/OpenMP backend: uses FFTW. Include and library directories can be specified with build options `FFTW_INC_DIR` and `FFTW_LIB_DIR`, respectively.
        * Cuda backend: uses cuFFT
        * HIP backend: uses hipFFT
        
* pybind11
    * ExaDiS uses [pybind11](https://github.com/pybind/pybind11) for the python binding module. pybind11 is included as a submodule to the repository and will be automatically cloned to the `python/pybind11` folder when using the git submodule commands or cloning with the `--recursive` option (see Step 1 of Quick Start section).
    To use a specific python version/executable, use build option `PYTHON_EXECUTABLE`. If needed, the include path to the `python-dev` package (containing file `Python.h`) can be provided with build option `PYTHON_DEV_INC_DIR`.
    To compile ExaDiS without this module, use build option `-DEXADIS_PYTHON_BINDING=Off`.


### Build options

Below is a list of the various CMake build option specific to ExaDiS. The build options are passed as arguments to the cmake command as `-D<BUILD_OPTION_NAME>=<value>`.

* `EXADIS_PYTHON_BINDING` (optional, default=`On`): enable/disable compilation of the python module
* `PYTHON_EXECUTABLE` (optional, default=''): specifies the path of a specific python version to be used
* `PYTHON_DEV_INC_DIR` (optional, default=''): specifies the path to the python-dev include directory
* `EXADIS_FFT` (optional, default=`On`): enable/disable compilation of the FFT-based long-range force calculation module
* `FFTW_INC_DIR` (optional, default=''): specifies the path of the FFTW include directory
* `FFTW_LIB_DIR` (optional, default=''): specifies the path of the FFTW library directory
* `EXADIS_BUILD_EXAMPLES` (optional, default=`Off`): builds examples that are in the `examples/` folder
* `EXADIS_BUILD_TESTS` (optional, default=`Off`): builds test cases that are in the `tests/` folder

Kokkos related main build options: (see full list [here](https://kokkos.org/kokkos-core-wiki/keywords.html))
* `Kokkos_ENABLE_SERIAL` (optional, default=`On`): enable/disable compilation with the serial (CPU) backend
* `Kokkos_ENABLE_OPENMP` (optional, default=`On`): enable/disable compilation with the OpenMP backend
* `Kokkos_ENABLE_CUDA` (optional, default=`Off`): enable/disable compilation with the CUDA backend. If `On`, option `-DKokkos_ENABLE_CUDA_LAMBDA=On` is also required, and a device architecture must be provided, e.g. `-DKokkos_ARCH_VOLTA70=On`.
* `Kokkos_ENABLE_HIP` (optional, default=`Off`): enable/disable compilation with the HIP backend.
* `Kokkos_ROOT` (optional, default=none) : specifies the path to a pre-existing Kokkos installation. Do not specify any of the above Kokkos options if this option is used; ExaDiS will be built with the backends that the pre-existing Kokkos installation was built for.


## Project structure

Brief description of the directories within this repository:

* `cmake/` : pre-defined build system options
* `examples/` : examples of scripts and simulations
* `python/` : files related to the python binding implementation
* `src/` : C++ source and header files (`*.cpp`, `*.h`)
* `tests/` : files for testing and debugging


## Simulation examples

There are several examples of simulation files located in the `examples/` folder. These examples show the different ways that ExaDiS simulations can be setup and run.

For instance, folder `examples/02_frank_read_src` provides an example of a simple Frank-Read source simulation, driven either through the C++ interface or the python interface. 

Folders `examples/21_bcc_Ta_100nm_2e8` and `examples/22_fcc_Cu_15um_1e3` provide examples of typical large-scale DDD production runs (a BCC and a FCC simulation) driven through the C++ or the python interfaces.

The python simulations requires the code to be compiled with the python binding module using build option `-DEXADIS_PYTHON_BINDING=On`. The C++ simulations can be compiled by using build option `-DEXADIS_BUILD_EXAMPLES=On`.


## Documentation

The full documentation of ExaDiS is available at the [OpenDiS documentation](https://opendis.github.io/OpenDiS/core_libraries/exadis_documentation/index.html).


## License

ExaDiS is released under the BSD-3 license. See [LICENSE](LICENSE) for details.

LLNL-CODE-862972


## Author
Nicolas Bertin (bertin1@llnl.gov)
