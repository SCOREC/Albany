#!/bin/bash

#-------------------------------------------
#  
# Prototype script to execute regression
# tests
#
# This script is executed from run_master_tpetra.sh
#
# BvBW  10/06/08
# AGS  04/09
#-------------------------------------------

#-------------------------------------------
# setup and housekeeping
#-------------------------------------------

cd $NIGHTLYDIR

if [ ! -d $ALBOUTDIR ]; then mkdir $ALBOUTDIR
fi

#-------------------------------------------
# run tests in Albany
#-------------------------------------------

cd $ALBDIR/build_albanyTonly
echo "------------------CTEST----------------------" \
     > $ALBOUTDIR/albany_runtests_albanyTonly.out

ctest >> $ALBOUTDIR/albany_runtests_albanyTonly.out

echo >> $ALBOUTDIR/albany_runtests_albanyTonly.out
