#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/two_bars3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/two_bars3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputWriteRestartIntervalT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputWriteRestartIntervalT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputRestartIntervalFromFileT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputRestartIntervalFromFileT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# 2. Name the test with the directory name, prepended by SCOREC_
get_filename_component(testRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName "SCOREC_${testRoot}")

# 3. Create the test with this name and standard executable
IF(ALBANY_IFPACK2)
  add_test(NAME ${testName}_WriteRestartIntervalT COMMAND ${Albany.exe} inputWriteRestartIntervalT.yaml)
  set_tests_properties(${testName}_WriteRestartIntervalT PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC")
  add_test(NAME ${testName}_RestartIntervalFromFileT COMMAND ${Albany.exe} inputRestartIntervalFromFileT.yaml)
  set_tests_properties(${testName}_RestartIntervalFromFileT PROPERTIES LABELS "LCM;Tpetra;Forward;SCOREC" DEPENDS ${testName}_WriteRestartT)
ENDIF()
