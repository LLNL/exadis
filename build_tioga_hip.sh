#bin/bash

#device=mi250
device=mi300

echo "Building ExaDiS"

module load cmake/3.29.2

module load rocm/6.2.1
module load rocmcc/6.2.1-cce-18.0.1a-magic

export CRAYPE_LINK_TYPE=dynamic

BUILD_DIR=build_tioga_${device}

rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

# --save-temps

#-DCMAKE_BUILD_TYPE=Debug 
cmake -DSYS=tioga_${device} -DEXADIS_PYTHON_BINDING=Off -DEXADIS_BUILD_TESTS=On ..

make -j8

