#include "..//plugins/deformation_plugin.h"

int main()
{
	igl::opengl::glfw::Viewer viewer;
	deformation_plugin plugin;
	viewer.plugins.push_back(&plugin);
	viewer.launch();
	return EXIT_SUCCESS;
}
