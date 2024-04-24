#bin/bash

#################################################################
# This script is an attempt to simplify the buidling process
# of exadis by having it build both kokkos and exadis.
# It is intented to be used if kokkos is not installed on your 
# machine and if you are not too sure how to install it.
#
# The script will first clone the kokkos repo and try to install it.
# Then it will try to build exadis using the local kokkos install.
#
# This script should work in simple situations (e.g. building
# for CPU), but may fail for more complex environments.
# If it fails, it is recommended that the user builds kokkos
# and exadis independently as detailed in the exadis README file.
#
# Below, the user can configure build options for their own system
# (SYS=user) or use pre-configured options (e.g. SYS=lassen).
#
# Nicolas Bertin (bertin1@llnl.gov)
#################################################################

SYS=user


#-------------------------------------------------
# USER DEFINED SYSTEM
# Configure your build options here
#-------------------------------------------------
# C++ compiler
CXX_COMPILER_user=c++
# Kokkos backend: Serial, OpenMP, CUDA and/or HIP
SERIAL_user=On
OPENMP_user=On
CUDA_user=Off
HIP_user=Off
ARCH_user=
# ExaDiS options
PYTHON_BINDING_user=On
PYTHON_EXECUTABLE_user=
FFTW_INC_DIR_user=
FFTW_LIB_DIR_user=
#-------------------------------------------------


#-------------------------------------------------
# Nicolas mac config
#-------------------------------------------------
# C++ compiler
CXX_COMPILER_mac_nicolas=c++
# Kokkos backend: Serial, OpenMP, CUDA and/or HIP
SERIAL_mac_nicolas=On
OPENMP_mac_nicolas=Off
CUDA_mac_nicolas=Off
HIP_mac_nicolas=Off
ARCH_mac_nicolas=
# ExaDiS options
PYTHON_BINDING_mac_nicolas=On
PYTHON_EXECUTABLE_mac_nicolas=/Users/bertin1/opt/anaconda3/bin/python
FFTW_INC_DIR_mac_nicolas=/Users/bertin1/Documents/Codes/p365_bcc_nl_mob/ext/include/fftw
FFTW_LIB_DIR_mac_nicolas=/Users/bertin1/Documents/Codes/p365_bcc_nl_mob/ext/lib
#-------------------------------------------------

#-------------------------------------------------
# LLNL lassen with CUDA backend (Volta70)
#-------------------------------------------------
# C++ compiler
CXX_COMPILER_lassen=c++
# Kokkos backend: Serial, OpenMP, CUDA and/or HIP
SERIAL_lassen=On
OPENMP_lassen=On
CUDA_lassen=On
HIP_lassen=Off
ARCH_lassen=VOLTA70
# ExaDiS options
PYTHON_BINDING_lassen=On
PYTHON_EXECUTABLE_lassen=/usr/tcetmp/bin/python3
FFTW_INC_DIR_lassen=
FFTW_LIB_DIR_lassen=
#-------------------------------------------------

#-------------------------------------------------
# LLNL lassen with OpenMP backend only
#-------------------------------------------------
# C++ compiler
CXX_COMPILER_lassen_omp=c++
# Kokkos backend: Serial, OpenMP, CUDA and/or HIP
SERIAL_lassen_omp=On
OPENMP_lassen_omp=On
CUDA_lassen_omp=Off
HIP_lassen_omp=Off
ARCH_lassen_omp=
# ExaDiS options
PYTHON_BINDING_lassen_omp=On
PYTHON_EXECUTABLE_lassen_omp=/usr/tcetmp/bin/python3
FFTW_INC_DIR_lassen_omp=
FFTW_LIB_DIR_lassen_omp=
#-------------------------------------------------



# Substitute system variables
cxx=CXX_COMPILER_${SYS}
serial=SERIAL_${SYS}
openmp=OPENMP_${SYS}
cuda=CUDA_${SYS}
hip=HIP_${SYS}
arch=ARCH_${SYS}
python_bind=PYTHON_BINDING_${SYS}
python_exec=PYTHON_EXECUTABLE_${SYS}
fftw_inc=FFTW_INC_DIR_${SYS}
fftw_lib=FFTW_LIB_DIR_${SYS}

# Clone, build and instal Kokkos
echo ""
echo "BUILDING KOKKOS ... "
echo ""
if [ ! -d "kokkos" ]; then
   git clone https://github.com/kokkos/kokkos.git --branch 4.2.00
else
   echo "Kokkos directory exists"
fi
cd kokkos
rm -rf build_${SYS}
mkdir build_${SYS} && cd build_${SYS}
kokkos_arch=
if [ -n "${!arch}" ]; then
    # a device architecture has been specified
    kokkos_arch="-DKokkos_ARCH_${!arch}=On" 
fi
cmake \
    -DCMAKE_CXX_COMPILER=${!cxx} \
    -DCMAKE_INSTALL_PREFIX=../install_${SYS} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=On \
    -DKokkos_ENABLE_SERIAL=${!serial} \
    -DKokkos_ENABLE_OPENMP=${!openmp} \
    -DKokkos_ENABLE_CUDA=${!cuda} \
    -DKokkos_ENABLE_CUDA_LAMBDA=${!cuda} \
    -DKokkos_ENABLE_HIP=${!hip} \
    ${kokkos_arch} \
    ..
make -j8
make install
cd ..
if [ ! -d "install_${SYS}" ]; then
    echo "Error while building kokkos, see above"
    exit 1
fi
cd ..
echo ""
echo "FINISHED BUILDING KOKKOS"
echo ""

# Build ExaDiS
echo ""
echo "BUILDING EXADIS ... "
echo ""
git submodule init
git submodule update
exadis_cxx=${!cxx}
if [ ${!cuda} == "On" ]; then
    exadis_cxx=$(pwd)/kokkos/install_${SYS}/bin/nvcc_wrapper
fi
rm -rf build_${SYS}
mkdir build_${SYS} && cd build_${SYS}
cmake \
    -DKokkos_ROOT=../kokkos/install_${SYS} \
    -DCMAKE_CXX_COMPILER=${exadis_cxx} \
    -DPYTHON_BINDING=${!python_bind} \
    -DPYTHON_EXECUTABLE=${!python_exec} \
    -DFFTW_INC_DIR=${!fftw_inc} \
    -DFFTW_LIB_DIR=${!fftw_lib} \
    ..
make -j8
cd ..
echo ""
echo "FINISHED BUILDING EXADIS"
echo ""
