cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(FabricationLab)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(LIBIGL_WITH_OPENGL       "Use OpenGL" ON)
option(LIBIGL_WITH_OPENGL_GLFW  "Use GLFW"   ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui" ON)
option(LIBIGL_WITH_MATLAB 	    "Use MATLAB" ON)
option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" ON)

find_package(MKL)
find_package(LIBIGL REQUIRED QUIET)

add_subdirectory("libs")
add_subdirectory("plugins")
add_subdirectory("apps")
