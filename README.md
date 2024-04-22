# ExaDiS

ExaDiS version 0.1

ExaDiS (Exascale Dislocation Simulator) is a set of software modules written to enable numerical simulations of large groups of moving and interacting dislocations, line defects in crystals responsible for crystal plasticity. By tracking time evolution of sufficiently large dislocation ensembles, ExaDiS predicts plasticity response and plastic strength of crystalline materials.

ExaDiS is built around a portable library of core functions for Discrete Dislocation Dynamics (DDD) method specifically written to perform efficient computations on new HPC architectures (e.g. GPUs). Simulations can be driven through the C++ or python interfaces. The python binding module can also be interfaced with the upcoming [OpenDiS](https://github.com/opendis/) framework.

Note: Although ExaDiS is a fully functional code, it is currently under active development and is subject to frequent updates and bug fixes. There is no guarantee of stability and one should expect occasional breaking changes to the code.


## Installation

### Quick start

ExaDiS is implemented using the [Kokkos](https://kokkos.org) framework and built using the CMake build system.

* Step 1: Install Kokkos
    * https://github.com/kokkos/kokkos
      
* Step 2: Build ExaDiS
    * Clone the repository
    ```
    git clone https://github.com/LLNL/exadis.git
    cd exadis
    ```
    * Initialize the submodules (required if enabling the python binding)
    ```
    git submodule init
    git submodule update
    ```
    * Build the code. Examples of building scripts are provided at the root of the exadis project, e.g. see file `build_mac.sh`. The Kokkos root path must be specified with option `-DKokkos_ROOT`. A typical build instruction will look like:
    ```
    mkdir build && cd build
    cmake \
        -DKokkos_ROOT=/path/to/your/kokkos/install/lib/cmake/Kokkos \
        -DCMAKE_CXX_COMPILER=c++ \
        -DPYTHON_BINDING=On \
        ..
    make -j8
    ```
    Note: building with nvcc (Cuda) may be pretty slow, please be patient! 
    For additional building options and troubleshooting see section Detailed build instructions below.
    
* Step 3: Test your installation by running an example (assuming `-DPYTHON_BINDING=On`)
```
cd examples/02_frank_read_src
python test_frank_read_src.py
```

### Detailed build instructions

#### Dependencies

* Kokkos:
    * ExaDiS is implemented using the Kokkos framework. Kokkos must be installed in the machine prior to building ExaDiS. Instructions on how to configure/install Kokkos are found at https://github.com/kokkos/kokkos. ExaDiS will be compiled with the backend(s) that Kokkos was built for. For instance, if Kokkos was built to run on GPUs (e.g. compiled with option `-DKokkos_ENABLE_CUDA=ON`), then ExaDiS will be compiled to run on GPUs. The path to the Kokkos installation must be provided with ExaDiS build option `-DKokkos_ROOT`.
* FFT libraries
    * ExaDiS uses FFT libraries to compute long-range elastic interactions. To compile ExaDiS without this module (e.g. if no FFT library is available) use build option `-DEXADIS_FFT=Off`. Otherwise (default), different FFT libraries are invoked depending on the target backend:
        * Serial/OpenMP backend: uses FFTW. Include and library directories can be specified with build options `FFTW_INC_DIR` and `FFTW_LIB_DIR`, respectively.
        * Cuda backend: uses cuFFT
        * HIP backend: uses hipFFT
        
* pybind11
    * ExaDiS uses [pybind11](https://github.com/pybind/pybind11) for the python binding module. pybind11 is included as a submodule to the repository and will be automatically cloned to the `python/pybind11` folder when using git submodule:
    ```
    git submodule init
    git submodule update
    ```
    To use a specific python version/executable, use build option `PYTHON_EXECUTABLE`.
    To compile ExaDiS without this module, use build option `-DPYTHON_BINDING=Off`.


#### Build options

Below is a list of the various CMake build option specific to ExaDiS. The build options are passed as arguments to the cmake command as `-D<BUILD_OPTION_NAME>=<value>`.

* `Kokkos_ROOT` (required) : specifies the path of the Kokkos installation
* `PYTHON_BINDING` (optional, default=`On`): enable/disable compilation of the python module
* `PYTHON_EXECUTABLE` (optional, default=''): specifies the path of a specific python version to be used
* `EXADIS_FFT` (optional, default=`On`): enable/disable compilation of the FFT-based long-range force calculation module
* `FFTW_INC_DIR` (optional, default=''): specifies the path of the FFTW include directory
* `FFTW_LIB_DIR` (optional, default=''): specifies the path of the FFTW library directory
* `EXADIS_BUILD_EXAMPLES` (optional, default=`Off`): builds examples that are in the `examples/` folder
* `EXADIS_BUILD_TESTS` (optional, default=`Off`): builds test cases that are in the `tests/` folder


## Project structure

Brief description of the directories within this repository:

* `examples/` : examples of scripts and simulations
* `python/` : files related to the python binding implementation
* `src/` : C++ source and header files (`*.cpp`, `*.h`)

## Simulation examples

There are several examples of simulation files located in the `examples/` folder. These examples show the different ways that ExaDiS simulations can be setup and run.

For instance, folder `examples/02_frank_read_src` provides an example of a simple Frank-Read source simulation, driven either through the C++ interface or the python interface. 

Folders `examples/21_bcc_Ta_100nm_2e8` and `examples/22_fcc_Cu_15um_1e3` provide examples of typical large-scale DDD production runs (a BCC and a FCC simulation) driven through the C++ or the python interfaces.

The python simulations requires the code to be compiled with the python binding module using build option `-DPYTHON_BINDING=On`. The C++ simulations can be compiled by using build option `-DEXADIS_BUILD_EXAMPLES=On`.


## License

ExaDiS is released under the BSD-3 license. See [LICENSE](LICENSE) for details.

LLNL-CODE-862972


## Author
Nicolas Bertin (bertin1@llnl.gov)
