cmake_minimum_required(VERSION 3.8)

# project name init
project(
	raccoon
	VERSION 0.0.1
	DESCRIPTION "A small autograd library for C."
	LANGUAGES C
)

# setting default build type
set(CMAKE_C_STANDARD 11)
set(DEFAULT_BUILD_TYPE "Debug")
set(CMAKE_C_FLAGS "-Wall -Wpedantic -Wextra -Wreturn-type -Wswitch -Wunused -Werror -O2")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# add subproject
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/vita)

# finding all headers and sources
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.c)
file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/inc/*.h)

# include directories
include_directories(${PROJECT_SOURCE_DIR}/inc/)
include_directories(${PROJECT_SOURCE_DIR}/third_party/vita/inc)

# building library/binary
add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS}) # for libraries
# add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})   # for binaries
