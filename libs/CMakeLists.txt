project(optimization_lib)

add_definitions(-D_USE_MATH_DEFINES)

file(GLOB "${PROJECT_NAME}_files"
	*.cpp
	*.h
	)

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_files})

target_link_libraries(${PROJECT_NAME} PRIVATE igl::core)

if(LIBIGL_WITH_MATLAB)
	message(matlab library)
	target_link_libraries(${PROJECT_NAME} PRIVATE igl::matlab)
endif()
if(MKL_FOUND)
	set(EIGEN_USE_MKL_ALL)
	set(MKL_LP64)
	message("MKL found. MKL Include dir: " ${MKL_INCLUDE_DIR})
	include_directories( ${MKL_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL_LIBRARIES})
endif()
