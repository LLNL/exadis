#bin/bash

BUILD_DIR=build
KOKKOS_DIR=/usr/workspace/xtalstr/Nicolas/kokkos4.2.0/install_lassen_gcc_omp

echo "Building ExaDiS"

rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

cmake \
    -DKokkos_ROOT=${KOKKOS_DIR}/lib64/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/usr/tce/packages/gcc/gcc-8.3.1/bin/g++ \
    -DPYTHON_BINDING=On \
    -DPYTHON_EXECUTABLE=/usr/tcetmp/bin/python3 \
    ..

make -j8
