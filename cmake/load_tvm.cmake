set(TILELANG_PREBUILT_TVM_LIB_DIR "" CACHE PATH
    "Directory containing compatible prebuilt TVM runtime and compiler libraries")
if(NOT TILELANG_PREBUILT_TVM_LIB_DIR AND
   DEFINED ENV{TILELANG_PREBUILT_TVM_LIB_DIR} AND
   NOT "$ENV{TILELANG_PREBUILT_TVM_LIB_DIR}" STREQUAL "")
  set(TILELANG_PREBUILT_TVM_LIB_DIR "$ENV{TILELANG_PREBUILT_TVM_LIB_DIR}")
endif()

if(TILELANG_PREBUILT_TVM_LIB_DIR)
  if(WIN32)
    message(FATAL_ERROR "TILELANG_PREBUILT_TVM_LIB_DIR currently supports POSIX builds only")
  endif()
  set(TVM_BUILD_FROM_SOURCE FALSE)
else()
  set(TVM_BUILD_FROM_SOURCE TRUE)
endif()
set(TVM_SOURCE ${CMAKE_SOURCE_DIR}/3rdparty/tvm)

if(DEFINED ENV{TVM_ROOT})
  if(EXISTS $ENV{TVM_ROOT}/cmake/config.cmake)
    set(TVM_SOURCE $ENV{TVM_ROOT})
    message(STATUS "Using TVM_ROOT from environment variable: ${TVM_SOURCE}")
  endif()
endif()

message(STATUS "Using TVM source: ${TVM_SOURCE}")

if(NOT TVM_BUILD_FROM_SOURCE)
  cmake_path(ABSOLUTE_PATH TILELANG_PREBUILT_TVM_LIB_DIR
             NORMALIZE OUTPUT_VARIABLE TILELANG_PREBUILT_TVM_LIB_DIR)
  set(TVM_PREBUILT_RUNTIME_LIBRARY
      "${TILELANG_PREBUILT_TVM_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}tvm_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(TVM_PREBUILT_COMPILER_LIBRARY
      "${TILELANG_PREBUILT_TVM_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}tvm_compiler${CMAKE_SHARED_LIBRARY_SUFFIX}")
  foreach(_tilelang_prebuilt_tvm_lib IN ITEMS
      TVM_PREBUILT_RUNTIME_LIBRARY TVM_PREBUILT_COMPILER_LIBRARY)
    if(NOT EXISTS "${${_tilelang_prebuilt_tvm_lib}}")
      message(FATAL_ERROR
              "Missing prebuilt TVM library: ${${_tilelang_prebuilt_tvm_lib}}")
    endif()
  endforeach()
  message(STATUS
          "Using prebuilt TVM libraries from: ${TILELANG_PREBUILT_TVM_LIB_DIR}")
endif()

set(TVM_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
)

if(EXISTS ${TVM_SOURCE}/ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/ffi/include)
elseif(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
endif()

if(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()
