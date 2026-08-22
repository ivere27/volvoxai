# Runs the receipt digit reader example (--help smoke) then its self-contained
# C test. Invoked as: cmake -DBIN=<example> -DTEST_BIN=<test> -P this-file
execute_process(COMMAND ${BIN} --help RESULT_VARIABLE _r OUTPUT_QUIET ERROR_QUIET)
if(NOT _r EQUAL 0)
  message(FATAL_ERROR "receipt digit reader '--help' failed (exit ${_r}).")
endif()
execute_process(COMMAND ${TEST_BIN} RESULT_VARIABLE _r2)
if(NOT _r2 EQUAL 0)
  message(FATAL_ERROR "receipt digit reader native test failed (exit ${_r2}).")
endif()
