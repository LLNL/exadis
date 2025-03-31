#!/bin/sh

# ./configure.sh -DSYS=mac_nicolas
# ./configure.sh build_mac -DSYS=mac_nicolas

# Get the directory name from the argument (optional)
BUILD_DIR=$1

# Check if the directory argument starts with "-D"
if [[ -z $BUILD_DIR || $BUILD_DIR == "-D"* ]]; then
    BUILD_DIR=build
fi

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
