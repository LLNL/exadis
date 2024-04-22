#bin/bash

BUILD_DIR=build
KOKKOS_DIR=/Users/bertin1/Documents/Codes/Libraries/kokkos-4.2.00/install_macos_gxx

echo "Building ExaDiS"

rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

cmake \
    -DKokkos_ROOT=${KOKKOS_DIR}/lib/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-8 \
    -DPYTHON_BINDING=On \
    -DPYTHON_EXECUTABLE=/Users/bertin1/opt/anaconda3/bin/python \
    ..
