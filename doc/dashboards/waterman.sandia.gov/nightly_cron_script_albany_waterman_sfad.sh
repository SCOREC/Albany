#!/bin/csh

BASE_DIR=/home/projects/albany/waterman
cd $BASE_DIR

unset http_proxy
unset https_proxy

export jenkins_albany_dir=/home/projects/albany/waterman/repos/Albany
export jenkins_trilinos_dir=/home/projects/albany/waterman/repos/Trilinos

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_watermanAlbanySFAD.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albanySFAD.cmake" > $LOG_FILE 2>&1

