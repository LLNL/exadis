#!/bin/sh

# ./configure.sh -DSYS=mac_nicolas

BUILD_DIR=build

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
echo "cd ${BUILD_DIR} ; cmake $@ -S .."
echo ""
cmake $@ -S ..

echo ""
echo "to build:"
echo "  cmake --build ${BUILD_DIR} -j8"
echo ""
