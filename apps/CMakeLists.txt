project(main_app)

add_executable(${PROJECT_NAME} 
	"main.cpp" 
	"natvis/eigen.natvis"
	)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        igl::core
		igl::opengl
        igl::opengl_glfw
        igl::opengl_glfw_imgui
		optimization_lib
		plugins
		)

if(MKL_FOUND)
	set(EIGEN_USE_MKL_ALL)
	set(MKL_LP64)
	message("MKL found. MKL Include dir: " ${MKL_INCLUDE_DIR})
	include_directories( ${MKL_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL_LIBRARIES})
endif()