#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

rm -rf ctest_nightly.cmake 
rm -rf modules.out 

unset http_proxy
unset https_proxy

source blake_intel_modules.sh >& modules.out  

export OMP_NUM_THREADS=1

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanySerialSFad.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_sfad.cmake" > $LOG_FILE 2>&1

