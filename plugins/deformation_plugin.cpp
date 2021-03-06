#include "..//..//plugins/deformation_plugin.h"
#include <igl/file_dialog_open.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <igl/writeOFF.h>
#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/adjacency_matrix.h>



#define ADDING_WEIGHT_PER_HINGE_VALUE 10.0f
#define MAX_WEIGHT_PER_HINGE_VALUE  500.0f //50.0f*ADDING_WEIGHT_PER_HINGE_VALUE
#define MAX_SIGMOID_PER_HINGE_VALUE  40.0f //50.0f*ADDING_WEIGHT_PER_HINGE_VALUE


deformation_plugin::deformation_plugin() :
	igl::opengl::glfw::imgui::ImGuiMenu(){}

IGL_INLINE void deformation_plugin::init(igl::opengl::glfw::Viewer *_viewer)
{
	ImGuiMenu::init(_viewer);
	if (!_viewer)
		return;
	for (int i = 0; i < 7; i++)
		CollapsingHeader_prev[i] = CollapsingHeader_curr[i] = false;
	UserInterface_UpdateAllOutputs = false;
	CollapsingHeader_change = false;
	neighbor_distance = brush_radius = 0.3;
	initSphereAuxVariables = OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS;
	isLoadNeeded = false;
	IsMouseDraggingAnyWindow = false;
	isMinimizerRunning = false;
	energies_window = results_window = outputs_window = true;
	neighbor_Type = app_utils::Neighbor_Type::CURR_FACE;
	isModelLoaded = false;
	isUpdateAll = true;
	face_coloring_Type = app_utils::Face_Colors::NO_COLORS;
	clustering_brightness_w = 0.65;
	faceColoring_type = 0;
	optimizer_type = Cuda::OptimizerType::Adam;
	linesearch_type = OptimizationUtils::LineSearch::FUNCTION_VALUE;
	view = app_utils::View::HORIZONTAL;
	Max_Distortion = 5;
	Vertex_Energy_color = RED_COLOR;
	Highlighted_face_color = Eigen::Vector3f(153 / 255.0f, 0, 153 / 255.0f);
	Neighbors_Highlighted_face_color = Eigen::Vector3f(1, 102 / 255.0f, 1);
	center_sphere_color = Eigen::Vector3f(0, 1, 1);
	center_vertex_color = Eigen::Vector3f(128 / 255.0f, 128 / 255.0f, 128 / 255.0f);
	Color_sphere_edges = Color_normal_edge = Eigen::Vector3f(0 / 255.0f, 100 / 255.0f, 100 / 255.0f);
	face_norm_color = Eigen::Vector3f(0, 1, 1);
	Fixed_vertex_color = BLUE_COLOR;
	Dragged_vertex_color = GREEN_COLOR;
	model_color = GREY_COLOR;
	text_color = BLACK_COLOR;
	//update input viewer
	inputCoreID = viewer->core_list[0].id;
	viewer->core(inputCoreID).background_color = Eigen::Vector4f(1, 1, 1, 0);
	viewer->core(inputCoreID).is_animating = true;
	viewer->core(inputCoreID).lighting_factor = 0.5;
	//Load multiple views
	Outputs.push_back(OptimizationOutput(viewer, optimizer_type, linesearch_type));
	core_size = 1.0 / (Outputs.size() + 1.0);
	//maximize window
	glfwMaximizeWindow(viewer->window);
}

void deformation_plugin::load_new_model(const std::string modelpath) 
{
	if (isModelLoaded)
		clear_sellected_faces_and_vertices();
	modelPath = modelpath;
	if (modelPath.length() == 0)
		return;
	modelName = app_utils::ExtractModelName(modelPath);
	stop_all_minimizers_threads();
	if (isModelLoaded) 
	{
		//remove previous data
		while (Outputs.size() > 0)
			remove_output(0);
		viewer->load_mesh_from_file(modelPath.c_str());
		viewer->erase_mesh(0);
	}
	else 
		viewer->load_mesh_from_file(modelPath.c_str());
	inputModelID = viewer->data_list[0].id;
	for (int i = 0; i < Outputs.size(); i++)
	{
		viewer->load_mesh_from_file(modelPath.c_str());
		Outputs[i].ModelID = viewer->data_list[i + 1].id;
		init_objective_functions(i);
	}
	if (isModelLoaded)
		add_output();
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	for (int i = 0; i < Outputs.size(); i++)
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
	
	//set rotation type to 3D mode
	viewer->core(inputCoreID).trackball_angle = Eigen::Quaternionf::Identity();
	viewer->core(inputCoreID).orthographic = false;
	viewer->core(inputCoreID).set_rotation_type(igl::opengl::ViewerCore::RotationType(1));
	isModelLoaded = true;
	isLoadNeeded = false;
}

IGL_INLINE void deformation_plugin::draw_viewer_menu()
{
	if (isModelLoaded && ui.status != app_utils::UserInterfaceOptions::NONE)
	{
		CollapsingHeader_user_interface();
		Draw_output_window();
		Draw_results_window();
		Draw_energies_window();
		return;
	}
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0)))
	{
		modelPath = igl::file_dialog_open();
		isLoadNeeded = true;
	}
	if (isLoadNeeded)
	{
		load_new_model(modelPath);
		isLoadNeeded = false;
	}
	if (!isModelLoaded)
		return;
	ImGui::SameLine();
	if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0)))
		viewer->open_dialog_save_mesh();

	ImGui::Combo("Active output", (int*)(&ActiveOutput), app_utils::build_outputs_list(Outputs.size()));
		
	if (ImGui::Button("save Sphere", ImVec2((w - p) / 2.f, 0)) && Outputs[ActiveOutput].clustering_faces_indices.size()) {
		// Multiply all the mesh by "factor". Relevant only for spheres. 
		double factor = 1;
		for (auto& obj : Outputs[ActiveOutput].totalObjective->objectiveList) {
			auto fR = std::dynamic_pointer_cast<fixRadius>(obj);
			if (fR != NULL && fR->w != 0)
				factor = fR->alpha;
		}
		// Get mesh data
		OptimizationOutput O = Outputs[ActiveOutput];
		Eigen::MatrixX3d colors = O.clustering_faces_colors;
		Eigen::MatrixXd V_OUT = factor * OutputModel(ActiveOutput).V;
		Eigen::MatrixXd V_IN = factor * InputModel().V;
		Eigen::MatrixXi F = OutputModel(ActiveOutput).F;
		Eigen::VectorXd Radiuses = factor * Outputs[ActiveOutput].getRadiusOfSphere();
		Eigen::MatrixXd Centers = factor * Outputs[ActiveOutput].getCenterOfSphere();
		
		// Create new Directory for saving the data
		std::string main_file_path = OptimizationUtils::ProjectPath() + "models\\OutputModels\\" + modelName + app_utils::CurrentTime() + "\\";
		std::string aux_file_path = main_file_path + "Auxiliary_Variables\\";
		std::string parts_file_path = main_file_path + "Sphere_Parts\\";
		std::string parts_color_file_path = main_file_path + "Sphere_Parts_With_Colors\\";
		std::string file_name = modelName + std::to_string(ActiveOutput);
		if (mkdir(main_file_path.c_str()) == -1 ||
			mkdir(parts_file_path.c_str()) == -1 ||
			mkdir(aux_file_path.c_str()) == -1 ||
			mkdir(parts_color_file_path.c_str()) == -1)
		{
			std::cerr << "Error :  " << strerror(errno) << std::endl;
			exit(1);
		}

		// Save each cluster in the new directory
		for (int clus_index = 0; clus_index < O.clustering_faces_indices.size(); clus_index++)
		{
			std::vector<int> clus_faces_index = O.clustering_faces_indices[clus_index];
			Eigen::MatrixX3i clus_faces_val(clus_faces_index.size(), 3);
			Eigen::MatrixX3d clus_faces_color(clus_faces_index.size(), 3);

			double sumRadius = 0;
			Eigen::RowVector3d sumCenters(0, 0, 0);
			for (int fi = 0; fi < clus_faces_index.size(); fi++)
			{
				sumRadius += Radiuses(clus_faces_index[fi]);
				sumCenters += Centers.row(clus_faces_index[fi]);
				clus_faces_val.row(fi) = F.row(clus_faces_index[fi]);
				clus_faces_color.row(fi) = colors.row(clus_faces_index[fi]);
			}
			Eigen::RowVector3d avgCenter = sumCenters / clus_faces_index.size();
			double avgRadius = sumRadius / clus_faces_index.size();

			Eigen::MatrixX3d clus_vertices(V_OUT.rows(), 3);
			for (int vi = 0; vi < V_OUT.rows(); vi++)
				clus_vertices.row(vi) = V_OUT.row(vi);
			// Save the current cluster in "off" file format
			std::string clus_file_name = parts_file_path + file_name + "_sphere_" + std::to_string(clus_index) + ".off";
			std::string clus_file_name_colors = parts_color_file_path + file_name + "_sphere_" + std::to_string(clus_index) + "_withColors.off";
			app_utils::writeOFFwithColors(clus_file_name_colors, clus_vertices, clus_faces_val, clus_faces_color);
			igl::writeOFF(clus_file_name, clus_vertices, clus_faces_val);
		}
		// Save the final mesh in "off" file format
		igl::writeOFF(main_file_path + file_name + "_Output.off", V_OUT, F);
		igl::writeOFF(main_file_path + file_name + "_Input.off", V_IN, F);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Output_withColors.off", V_OUT, F, colors);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Input_withColors.off", V_IN, F, colors);
		app_utils::writeTXTFile(main_file_path + file_name + "ReadMe.txt", modelName, true,
			O.clustering_faces_indices, V_OUT, F, colors, Radiuses, Centers);
		//save auxiliary variables
		Eigen::MatrixXi temp(1, 3);
		temp << 1, 3, 2;
		igl::writeOFF(aux_file_path + file_name + "_Aux_Centers.off", Centers, temp);
		Eigen::MatrixXd mat_radiuses(Radiuses.size(), 3);
		mat_radiuses.setZero();
		mat_radiuses.col(0) = Radiuses;
		igl::writeOFF(aux_file_path + file_name + "_Aux_Radiuses.off", mat_radiuses, temp);
	}
	ImGui::SameLine();
	if (ImGui::Button("Save Planar", ImVec2((w - p) / 2.f, 0)) && Outputs[ActiveOutput].clustering_faces_indices.size()) {
		// Get mesh data
		OptimizationOutput O = Outputs[ActiveOutput];
		Eigen::MatrixX3d colors = O.clustering_faces_colors;
		Eigen::MatrixXd V_OUT = OutputModel(ActiveOutput).V;
		Eigen::MatrixXd V_IN = InputModel().V;
		Eigen::MatrixXi F = OutputModel(ActiveOutput).F;
		Eigen::VectorXd Radiuses = Outputs[ActiveOutput].getRadiusOfSphere();
		Eigen::MatrixXd Centers = Outputs[ActiveOutput].getCenterOfSphere();
		Eigen::MatrixXd Normals = Outputs[ActiveOutput].getFacesNormals();

		// Create new Directory for saving the data
		std::string main_file_path = OptimizationUtils::ProjectPath() + "models\\OutputModels\\" + modelName + app_utils::CurrentTime() + "\\";
		std::string aux_file_path = main_file_path + "Auxiliary_Variables\\";
		std::string parts_file_path = main_file_path + "Polygon_Parts\\";
		std::string parts_color_file_path = main_file_path + "Polygon_Parts_With_Colors\\";
		std::string file_name = modelName + std::to_string(ActiveOutput);
		if (mkdir(main_file_path.c_str()) == -1 ||
			mkdir(parts_file_path.c_str()) == -1 ||
			mkdir(aux_file_path.c_str()) == -1 ||
			mkdir(parts_color_file_path.c_str()) == -1)
		{
			std::cerr << "Error :  " << strerror(errno) << std::endl;
			exit(1);
		}
		
		// Save each cluster in the new directory
		for (int polygon_index = 0; polygon_index < O.clustering_faces_indices.size(); polygon_index++)
		{
			std::vector<int> clus_F_indices = O.clustering_faces_indices[polygon_index];
			const int clus_Num_Faces = clus_F_indices.size();
			Eigen::MatrixX3i clus_F(clus_Num_Faces, 3);
			Eigen::MatrixX3d clus_color(clus_Num_Faces, 3);

			for (int fi = 0; fi < clus_Num_Faces; fi++)
			{
				clus_F.row(fi) = F.row(clus_F_indices[fi]);
				clus_color.row(fi) = colors.row(clus_F_indices[fi]);
			}
			// Save the current cluster in "off" file format
			std::string clus_file_name = parts_file_path + file_name + "_polygon_" + std::to_string(polygon_index) + ".off";
			std::string clus_file_name_colors = parts_color_file_path + file_name + "_polygon_" + std::to_string(polygon_index) + "_withColors.off";
			igl::writeOFF(clus_file_name, V_OUT, clus_F);
			app_utils::writeOFFwithColors(clus_file_name_colors, V_OUT, clus_F, clus_color);
		}
		// Save the final mesh in "off" file format
		igl::writeOFF(main_file_path + file_name + "_Input.off", V_IN, F);
		igl::writeOFF(main_file_path + file_name + "_Output.off", V_OUT, F);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Input_withColors.off", V_IN, F, colors);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Output_withColors.off", V_OUT, F, colors);
		app_utils::writeTXTFile(main_file_path + file_name + "ReadMe.txt", modelName, false,
			O.clustering_faces_indices, V_OUT, F, colors, Radiuses, Centers);
		//save auxiliary variables
		Eigen::MatrixXi temp(1, 3);
		temp << 1, 3, 2;
		igl::writeOFF(aux_file_path + file_name + "_Aux_Normals.off", Normals, temp);
	}

	ImGui::Checkbox("Outputs window", &outputs_window);
	ImGui::Checkbox("Results window", &results_window);
	ImGui::Checkbox("Energy window", &energies_window);
	CollapsingHeader_face_coloring();
	CollapsingHeader_screen();
	CollapsingHeader_clustering();
	CollapsingHeader_minimizer();
	CollapsingHeader_cores(viewer->core(inputCoreID), viewer->data(inputModelID));
	CollapsingHeader_models(viewer->data(inputModelID));
	CollapsingHeader_colors();
	Draw_output_window();
	Draw_results_window();
	Draw_energies_window();
	CollapsingHeader_update();
}

void deformation_plugin::CollapsingHeader_update()
{
	CollapsingHeader_change = false;
	int changed_index = NOT_FOUND;
	for (int i = 0; i < 7; i++)
	{
		if (CollapsingHeader_curr[i] && !CollapsingHeader_prev[i])
		{
			changed_index = i;
			CollapsingHeader_change = true;
		}
	}
	if (CollapsingHeader_change)
	{
		for (int i = 0; i < 7; i++)
			CollapsingHeader_prev[i] = CollapsingHeader_curr[i] = false;
		CollapsingHeader_prev[changed_index] = CollapsingHeader_curr[changed_index] = true;
	}
}

void deformation_plugin::CollapsingHeader_colors()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[0]);
	if (ImGui::CollapsingHeader("colors"))
	{
		CollapsingHeader_curr[0] = true;
		ImGui::ColorEdit3("Highlighted face", Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center sphere", center_sphere_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center vertex", center_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Sphere edge", Color_sphere_edges.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Normal edge", Color_normal_edge.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Face norm", face_norm_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Neighbors Highlighted face", Neighbors_Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed vertex", Fixed_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged vertex", Dragged_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Model", model_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Vertex Energy", Vertex_Energy_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit4("Text", text_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
	}
}

void deformation_plugin::CollapsingHeader_face_coloring()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[1]);
	if (ImGui::CollapsingHeader("Face coloring"))
	{
		CollapsingHeader_curr[1] = true;
		ImGui::Combo("type", (int *)(&faceColoring_type), app_utils::build_color_energies_list(Outputs[0].totalObjective));
		ImGui::PushItemWidth(80 * menu_scaling());
		ImGui::DragFloat("Max Distortion", &Max_Distortion, 0.05f, 0.01f, 10000.0f);
		ImGui::PopItemWidth();
	}
}

void deformation_plugin::CollapsingHeader_screen()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[2]);
	if (ImGui::CollapsingHeader("Screen options"))
	{
		CollapsingHeader_curr[2] = true;
		if (ImGui::Combo("View type", (int *)(&view), app_utils::build_view_names_list(Outputs.size())))
		{
			int frameBufferWidth, frameBufferHeight;
			glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
			post_resize(frameBufferWidth, frameBufferHeight);
		}
		if (view == app_utils::View::HORIZONTAL ||
			view == app_utils::View::VERTICAL)
		{
			if (ImGui::SliderFloat("Core Size", &core_size, 0, 1.0 / Outputs.size(), std::to_string(core_size).c_str(), 1))
			{
				int frameBufferWidth, frameBufferHeight;
				glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
				post_resize(frameBufferWidth, frameBufferHeight);
			}
		}
	}
}

void deformation_plugin::CollapsingHeader_user_interface()
{
	if (!ImGui::CollapsingHeader("User Interface"))
	{
		ImGui::Checkbox("Update UI together", &UserInterface_UpdateAllOutputs);
		ImGui::Combo("Neighbor type", (int *)(&neighbor_Type), "Curr Face\0Local Sphere\0Global Sphere\0Local Normals\0Global Normals\0\0");
		ImGui::DragFloat("Neighbors Distance", &neighbor_distance, 0.0005f, 0.00001f, 10000.0f,"%.5f");
		ImGui::DragFloat("Brush Radius", &brush_radius);
		if (ImGui::Button("Clear sellected faces & vertices"))
			clear_sellected_faces_and_vertices();
	}
}

void deformation_plugin::CollapsingHeader_clustering()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[3]);
	if (ImGui::CollapsingHeader("Clustering"))
	{
		CollapsingHeader_curr[3] = true;
		ImGui::Combo("Face Colors Type", (int*)(&face_coloring_Type), "No Colors\0Normals Clustering\0Spheres Clustering\0Sigmoid Parameter\0\0");
		ImGui::DragFloat("Bright. Weight", &clustering_brightness_w, 0.001f, 0, 1);
		if (ImGui::Combo("Clus. type", (int*)(&clusteringType), "None\0Radius\0Agglomerative hierarchical\0\0")) {
			if (clusteringType == app_utils::ClusteringType::Agglomerative_hierarchical) {
				for (auto& o : Outputs) {
					Agglomerative_hierarchical_clustering(o.getValues(face_coloring_Type), Clustering_MSE_Threshold, InputModel().F.rows(), o.clustering_faces_indices);
					o.clustering_colors.getFacesColors(o.clustering_faces_indices, InputModel().F.rows(), clustering_brightness_w, o.clustering_faces_colors);
					//set faces colors
					for (int fi = 0; fi < InputModel().F.rows(); fi++)
						o.setFaceColors(fi, o.clustering_faces_colors.row(fi).transpose());
				}
			}
		}
		if (ImGui::Button("Change Clusters Colors")) {
			for (auto& o : Outputs)
				o.clustering_colors.changeColors();
		}
		if (ImGui::Checkbox("self-intersection", &isChecking_SelfIntersection)) {
			if (isChecking_SelfIntersection) {
				for (int oi = 0; oi < Outputs.size(); oi++) {
					const Eigen::MatrixXd V = OutputModel(oi).V;
					const Eigen::MatrixXi F = OutputModel(oi).F;
					Outputs[oi].SelfIntersection_pairs.clear();
					for (int f1 = 0; f1 < F.rows(); f1++) {
						for (int f2 = f1 + 1; f2 < F.rows(); f2++) {
							if (app_utils::doTrianglesIntersect(f1, f2, V, F)) {
								//There is self-intersection!!!
								Outputs[oi].SelfIntersection_pairs.push_back(std::pair<int, int>(f1, f2));
								std::cout << "Self-intersection found between: " << f1 << ", " << f2 << std::endl;
							}
						}
					}
				}
			}
			else {
				for (auto& o : Outputs)
					o.SelfIntersection_pairs.clear();
			}
		}
		if (ImGui::Checkbox("flipped-faces", &isChecking_FlippedFaces)) {
			if (isChecking_FlippedFaces) {
				for (int oi = 0; oi < Outputs.size(); oi++) {
					const Eigen::MatrixXd V = OutputModel(oi).V;
					const Eigen::MatrixXi F = OutputModel(oi).F;
					auto& AS = Outputs[oi].Energy_auxSpherePerHinge;
					Outputs[oi].flippedFaces_pairs = app_utils::getFlippedFaces(V, F, AS->hinges_faceIndex);
				}
			}
			else {
				for (auto& o : Outputs)
					o.flippedFaces_pairs.clear();
			}
		}
		if (ImGui::Button("print clusters")) {
			for (int oi = 0; oi < Outputs.size(); oi++) {
				for (std::vector<int>& clusfaces : Outputs[oi].clustering_faces_indices) {
					std::cout << "=================\n";
					app_utils::getPlanarCoordinates(OutputModel(oi).V, OutputModel(oi).F, clusfaces);
				}
			}
		}
		ImGui::DragFloat("MSE threshold", &Clustering_MSE_Threshold, 0.000001f, 0, 1,"%.8f");
	}
}

void deformation_plugin::CollapsingHeader_minimizer()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[4]);
	if (ImGui::CollapsingHeader("Minimizer"))
	{
		CollapsingHeader_curr[4] = true;
		if (ImGui::Button("Run one iter"))
			run_one_minimizer_iter();
		if (ImGui::Checkbox("Run Minimizer", &isMinimizerRunning))
			isMinimizerRunning ? start_all_minimizers_threads() : stop_all_minimizers_threads();
		if (ImGui::Combo("Optimizer", (int *)(&optimizer_type), "Gradient Descent\0Adam\0\0"))
			change_minimizer_type(optimizer_type);
		if (ImGui::Combo("init sphere var", (int *)(&initSphereAuxVariables), "Sphere Fit\0Mesh Center\0Minus Normal\0\0"))
			init_aux_variables();
		if (initSphereAuxVariables == OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS &&
			ImGui::DragFloat("radius length", &radius_length_minus_normal, 0.01f, 0.0f, 1000.0f, "%.7f"))
			init_aux_variables();
		if (initSphereAuxVariables == OptimizationUtils::InitSphereAuxVariables::SPHERE_FIT) 
		{
			if (ImGui::DragInt("Neigh From", &(InitMinimizer_NeighLevel_From), 1, 1, 200))
				init_aux_variables();
			if (ImGui::DragInt("Neigh To", &(InitMinimizer_NeighLevel_To), 1, 1, 200))
				init_aux_variables();
		}

		if (ImGui::Combo("line search", (int *)(&linesearch_type), "Gradient Norm\0Function Value\0Constant Step\0\0")) {
			for (auto& o : Outputs)
				o.minimizer->lineSearch_type = linesearch_type;
		}
		if (linesearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP && ImGui::DragFloat("Step value", &constantStep_LineSearch, 0.0001f, 0.0f, 1.0f, "%.7f")) {
			for (auto& o : Outputs)
				o.minimizer->constantStep_LineSearch = constantStep_LineSearch;	
		}
		if (ImGui::Button("Check gradients"))
			checkGradients();
	}
}

void deformation_plugin::CollapsingHeader_cores(igl::opengl::ViewerCore& core, igl::opengl::ViewerData& data)
{
	if (!outputs_window)
		return;
	ImGui::PushID(core.id);
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[5]);
	if (ImGui::CollapsingHeader(("Core " + std::to_string(data.id)).c_str()))
	{
		CollapsingHeader_curr[5] = true;
		if (ImGui::Button("Center object", ImVec2(-1, 0)))
			core.align_camera_center(data.V, data.F);
		if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
			viewer->snap_to_canonical_quaternion();
		// Zoom & Lightining factor
		ImGui::PushItemWidth(80 * menu_scaling());
		ImGui::DragFloat("Zoom", &(core.camera_zoom), 0.05f, 0.1f, 100000.0f);
		ImGui::DragFloat("Lighting factor", &(core.lighting_factor), 0.05f, 0.1f, 20.0f);
		// Select rotation type
		int rotation_type = static_cast<int>(core.rotation_type);
		static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
		static bool orthographic = true;
		if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\02D Mode\0\0"))
		{
			using RT = igl::opengl::ViewerCore::RotationType;
			auto new_type = static_cast<RT>(rotation_type);
			if (new_type != core.rotation_type)
			{
				if (new_type == RT::ROTATION_TYPE_NO_ROTATION)
				{
					trackball_angle = core.trackball_angle;
					orthographic = core.orthographic;
					core.trackball_angle = Eigen::Quaternionf::Identity();
					core.orthographic = true;
				}
				else if (core.rotation_type == RT::ROTATION_TYPE_NO_ROTATION)
				{
					core.trackball_angle = trackball_angle;
					core.orthographic = orthographic;
				}
				core.set_rotation_type(new_type);
			}
		}
		if(ImGui::Checkbox("Orthographic view", &(core.orthographic)) && isUpdateAll)
			for (auto& c : viewer->core_list)
				c.orthographic = core.orthographic;
		ImGui::PopItemWidth();
		if (ImGui::ColorEdit4("Background", core.background_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) && isUpdateAll)
			for (auto& c : viewer->core_list)
				c.background_color = core.background_color;
	}
	ImGui::PopID();
}

void deformation_plugin::CollapsingHeader_models(igl::opengl::ViewerData& data)
{
	if (!outputs_window)
		return;
	auto make_checkbox = [&](const char *label, unsigned int &option) {
		bool temp = option;
		bool res = ImGui::Checkbox(label, &temp);
		option = temp;
		return res;
	};
	ImGui::PushID(data.id);
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[6]);
	if (ImGui::CollapsingHeader((modelName + " " + std::to_string(data.id)).c_str()))
	{
		CollapsingHeader_curr[6] = true;
		if (ImGui::Checkbox("Face-based", &(data.face_based)))
		{
			data.dirty = igl::opengl::MeshGL::DIRTY_ALL;
			if(isUpdateAll)
			{
				for (auto& d : viewer->data_list)
				{
					d.dirty = igl::opengl::MeshGL::DIRTY_ALL;
					d.face_based = data.face_based;
				}
			}
		}
		if (make_checkbox("Show texture", data.show_texture) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_texture = data.show_texture;
		if (ImGui::Checkbox("Invert normals", &(data.invert_normals))) {
			if (isUpdateAll)
			{
				for (auto& d : viewer->data_list)
				{
					d.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
					d.invert_normals = data.invert_normals;
				}
			}
			else
				data.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
		}
		if (make_checkbox("Show overlay", data.show_overlay) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_overlay = data.show_overlay;
		if (make_checkbox("Show overlay depth", data.show_overlay_depth) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_overlay_depth = data.show_overlay_depth;
		if (ImGui::ColorEdit4("Line color", data.line_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.line_color = data.line_color;
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		if (ImGui::DragFloat("Shininess", &(data.shininess), 0.05f, 0.0f, 100.0f) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.shininess = data.shininess;
		ImGui::PopItemWidth();
		if (make_checkbox("Wireframe", data.show_lines) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_lines = data.show_lines;
		if (make_checkbox("Fill", data.show_faces) && isUpdateAll)
			for(auto& d: viewer->data_list)
				d.show_faces = data.show_faces;
		if (ImGui::Checkbox("Show vertex labels", &(data.show_vertid)) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_vertid = data.show_vertid;
		if (ImGui::Checkbox("Show faces labels", &(data.show_faceid)) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_faceid = data.show_faceid;
	}
	ImGui::PopID();
}

void deformation_plugin::Draw_energies_window()
{
	if (!energies_window)
		return;
	ImGui::SetNextWindowPos(energies_window_position);
	ImGui::Begin("Energies & Timing", NULL, ImGuiWindowFlags_AlwaysAutoResize);
	int id = 0;
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
	if (ImGui::Button(("Add one more " + modelName).c_str()))
		add_output();
	ImGui::PopStyleColor();
	
	//add automatic lambda change
	if (ImGui::BeginTable("Lambda table", 12, ImGuiTableFlags_Resizable))
	{
		ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Max Update", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("On/Off", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Start from", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Stop at", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("#iter//lambda", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("#iter", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Curr Time [ms]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Avg Time [ms]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Total Time [m]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("lineSearch step size", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("lineSearch #iter", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableAutoHeaders();
		ImGui::Separator();
		ImGui::TableNextRow();
		ImGui::PushItemWidth(80);
		for (auto&out : Outputs) {
			ImGui::PushID(id++);
			const int  i64_zero = 0, i64_max = 100000.0;
			ImGui::Text((modelName + std::to_string(out.ModelID)).c_str());
			ImGui::TableNextCell();
			ImGui::Checkbox("##Max_Update", &out.minimizer->isUpdateLambdaWhenConverge);
			ImGui::TableNextCell();
			ImGui::Checkbox("##On/Off", &out.minimizer->isAutoLambdaRunning);
			ImGui::TableNextCell();
			ImGui::DragInt("##From", &(out.minimizer->autoLambda_from), 1, i64_zero, i64_max);
			ImGui::TableNextCell();
			ImGui::DragInt("##count", &(out.minimizer->autoLambda_count), 1, i64_zero, i64_max, "2^%d");
			ImGui::TableNextCell();
			ImGui::DragInt("##jump", &(out.minimizer->autoLambda_jump), 1, 1, i64_max);
			
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->getNumiter()).c_str());
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->timer_curr).c_str());
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->timer_avg).c_str());
			ImGui::TableNextCell();

			double tot_time = out.minimizer->timer_sum / 1000;
			int sec = int(tot_time) % 60;
			int min = (int(tot_time) - sec) / 60;

			ImGui::Text((std::to_string(min)+":" +std::to_string(sec)).c_str());
			ImGui::TableNextCell();
			ImGui::Text(("2^" + std::to_string(int(log2(out.minimizer->init_step_size)))).c_str());
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->linesearch_numiterations).c_str());
			ImGui::PopID();
			ImGui::TableNextRow();
		}
		ImGui::PopItemWidth();
		ImGui::EndTable();
	}
	
	if (Outputs.size() != 0) {
		if (ImGui::BeginTable("Unconstrained weights table", Outputs[0].totalObjective->objectiveList.size() + 3, ImGuiTableFlags_Resizable))
		{
			ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
			ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
			for (auto& obj : Outputs[0].totalObjective->objectiveList) {
				ImGui::TableSetupColumn(obj->name.c_str(), ImGuiTableColumnFlags_WidthAlwaysAutoResize);
			}
			ImGui::TableAutoHeaders();
			ImGui::Separator();
			
			ImGui::TableNextRow();
			for (int i = 0; i < Outputs.size(); i++) 
			{
				ImGui::Text((modelName + std::to_string(Outputs[i].ModelID)).c_str());
				ImGui::TableNextCell();
				ImGui::PushID(id++);
				if (ImGui::Button("On/Off")) {
					if (Outputs[i].minimizer->is_running)
						stop_one_minimizer_thread(Outputs[i]);
					else
						start_one_minimizer_thread(Outputs[i]);
				}
				ImGui::PopID();
				ImGui::TableNextCell();
				ImGui::PushItemWidth(80);
				for (auto& obj : Outputs[i].totalObjective->objectiveList) {
					ImGui::PushID(id++);
					ImGui::DragFloat("##w", &(obj->w), 0.05f, 0.0f, 100000.0f);
					auto SD = std::dynamic_pointer_cast<SDenergy>(obj);
					auto fR = std::dynamic_pointer_cast<fixRadius>(obj);

					auto ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
					auto AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
					auto BN = std::dynamic_pointer_cast<BendingNormal>(obj);

					if (obj->w) {
						if (fR != NULL) {
							ImGui::DragInt("min", &(fR->min));
							fR->min = fR->min < 1 ? 1 : fR->min;
							ImGui::DragInt("max", &(fR->max));
							fR->max = fR->max > fR->min ? fR->max : fR->min + 1;
							ImGui::DragFloat("alpha", &(fR->alpha), 0.001);
							Eigen::VectorXd Radiuses = Outputs[ActiveOutput].getRadiusOfSphere();
							if (ImGui::Button("update Alpha")) {
								fR->alpha = fR->max / Radiuses.maxCoeff();
							}
							ImGui::Text(("R max: " + std::to_string(Radiuses.maxCoeff() * fR->alpha)).c_str());
							ImGui::Text(("R min: " + std::to_string(Radiuses.minCoeff() * fR->alpha)).c_str());
							
						}

						if (ABN != NULL)
							ImGui::Combo("Function", (int*)(&(ABN->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (BN != NULL)
							ImGui::Combo("Function", (int*)(&(BN->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (AS != NULL)
							ImGui::Combo("Function", (int*)(&(AS->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						
						if (ABN != NULL && ABN->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(ABN->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->Inc_SigmoidParameter();
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->Dec_SigmoidParameter();
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(ABN->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(ABN->w2), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w3", ImGuiDataType_Double, &(ABN->w3), 0.05f, &f64_zero, &f64_max);
						}
						if (AS != NULL && AS->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(AS->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->Inc_SigmoidParameter();
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->Dec_SigmoidParameter();
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(AS->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(AS->w2), 0.05f, &f64_zero, &f64_max);
						}
						if (BN != NULL && BN->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(BN->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
								BN->Inc_SigmoidParameter();
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
								BN->Dec_SigmoidParameter();
						}
					}
					ImGui::TableNextCell();
					ImGui::PopID();
				}
				ImGui::PushID(id++);
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.0f, 0.0f, 1.0f));
				if (Outputs.size() > 1 && ImGui::Button("Remove"))
					remove_output(i);
				ImGui::PopStyleColor();
				ImGui::PopID();
				ImGui::PopItemWidth();
				ImGui::TableNextRow();
			}	
			ImGui::EndTable();
		}
	}
	ImVec2 w_size = ImGui::GetWindowSize();
	energies_window_position = ImVec2(0.5 * global_screen_size[0] - 0.5 * w_size[0], global_screen_size[1] - w_size[1]);
	//close the window
	ImGui::End();
}

void deformation_plugin::Draw_output_window()
{
	if (!outputs_window)
		return;
	for (auto& out : Outputs) 
	{
		ImGui::SetNextWindowSize(ImVec2(200, 300));
		ImGui::SetNextWindowPos(out.outputs_window_position);
		ImGui::Begin(("Output settings " + std::to_string(out.CoreID)).c_str(),
			NULL,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove
		);
		ImGui::Checkbox("Update all models together", &isUpdateAll);
		CollapsingHeader_cores(viewer->core(out.CoreID), viewer->data(out.ModelID));
		CollapsingHeader_models(viewer->data(out.ModelID));

		ImGui::Text("Show:");
		if (ImGui::Checkbox("Norm", &(out.showFacesNorm)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showFacesNorm = out.showFacesNorm;
		ImGui::SameLine();
		if (ImGui::Checkbox("Norm Edges", &(out.showNormEdges)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showNormEdges = out.showNormEdges;
		if (ImGui::Checkbox("Sphere", &(out.showSphereCenters)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showSphereCenters = out.showSphereCenters;
		ImGui::SameLine();
		if (ImGui::Checkbox("Sphere Edges", &(out.showSphereEdges)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showSphereEdges = out.showSphereEdges;
		if (ImGui::Checkbox("Face Centers", &(out.showTriangleCenters)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showTriangleCenters = out.showTriangleCenters;
		ImGui::End();
	}
}

void deformation_plugin::Draw_results_window()
{
	if (!results_window)
		return;
	for (auto& out : Outputs)
	{
		bool bOpened2(true);
		ImColor c(text_color[0], text_color[1], text_color[2], 1.0f);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
		ImGui::Begin(("Text " + std::to_string(out.CoreID)).c_str(), &bOpened2,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoScrollWithMouse |
			ImGuiWindowFlags_NoBackground |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoFocusOnAppearing |
			ImGuiWindowFlags_NoBringToFrontOnFocus);
		ImGui::SetWindowPos(out.results_window_position);
		ImGui::SetWindowSize(out.screen_size);
		ImGui::SetWindowCollapsed(false);
		
		
		ImGui::TextColored(c, (
			std::string("Num Faces: ") +
			std::to_string(InputModel().F.rows()) +
			std::string("\tNum Vertices: ") +
			std::to_string(InputModel().V.rows()) +
			std::string("\nGrad Size: ") +
			std::to_string(out.totalObjective->objectiveList[0]->grad.size) +
			std::string("\tNum Clusters: ") +
			std::to_string(out.clustering_faces_indices.size())
			).c_str());
		ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" energy ") + std::to_string(out.totalObjective->energy_value)).c_str());
		ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" gradient ") + std::to_string(out.totalObjective->gradient_norm)).c_str());
		for (auto& obj : out.totalObjective->objectiveList) {
			if (obj->w)
			{
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" energy ") + std::to_string(obj->energy_value)).c_str());
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" gradient ") + std::to_string(obj->gradient_norm)).c_str());
			}
		}
		if (clusteringType != app_utils::ClusteringType::NO_CLUS) {
			double maxMSE = app_utils::getMaxMSE(out.getValues(face_coloring_Type), out.clustering_faces_indices);
			ImGui::TextColored(c, ("Max MSE = " + std::to_string(maxMSE * 1000) + "m").c_str());
			
			double minMSE = app_utils::getMinMSE(out.getValues(face_coloring_Type), out.clustering_faces_indices);
			ImGui::TextColored(c, ("Min MSE = " + std::to_string(minMSE * 1000) + "m").c_str());
			
			double avgMSE = app_utils::getAvgMSE(out.getValues(face_coloring_Type), out.clustering_faces_indices);
			ImGui::TextColored(c, ("avg MSE = " + std::to_string(avgMSE * 1000) + "m").c_str());

			double stdMSE = app_utils::getStdMSE(out.getValues(face_coloring_Type), out.clustering_faces_indices);
			ImGui::TextColored(c, ("std MSE = " + std::to_string(stdMSE * 1000) + "m").c_str());

			int output_index, face_index;
			Eigen::Vector3f intersec_point;
			int cluster_index = -1;
			if (pick_face(output_index, face_index, intersec_point)) {
				//find the cluster index
				for (int ci = 0; ci < out.clustering_faces_indices.size(); ci++) {
					std::vector<int>& currClus = out.clustering_faces_indices[ci];
					if (std::find(currClus.begin(), currClus.end(), face_index) != currClus.end())
						cluster_index = ci;
				}
				//output the cluster data
				ImGui::TextColored(c, ("Cluster = " + std::to_string(cluster_index)).c_str());
				double clusMaxMSE = app_utils::getMaxMSE_1Cluster(out.getValues(face_coloring_Type), out.clustering_faces_indices[cluster_index]);
				ImGui::TextColored(c, ("Clus. Max MSE = " + std::to_string(clusMaxMSE * 1000) + "m").c_str());
			}
		}
		ImGui::End();
		ImGui::PopStyleColor();
	}
}

void deformation_plugin::clear_sellected_faces_and_vertices() 
{
	for (auto& o : Outputs) {
		o.Energy_auxSpherePerHinge->Clear_HingesWeights();
		o.Energy_auxBendingNormal->Clear_HingesWeights();
		o.Energy_BendingNormal->Clear_HingesWeights();
		o.Energy_FixChosenVertices->clearConstraints();
	}
}

void deformation_plugin::update_parameters_for_all_cores() 
{
	if (!isUpdateAll)
		return;
	for (auto& core : viewer->core_list) 
	{
		int output_index = NOT_FOUND;
		for (int i = 0; i < Outputs.size(); i++)
			if (core.id == Outputs[i].CoreID)
				output_index = i;
		if (output_index == NOT_FOUND)
		{
			if (this->prev_camera_zoom != core.camera_zoom ||
				this->prev_camera_translation != core.camera_translation ||
				this->prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) 
			{
				for (auto& c : viewer->core_list) 
				{
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs)
				{
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}
			}
		}
		else 
		{
			if (Outputs[output_index].prev_camera_zoom != core.camera_zoom ||
				Outputs[output_index].prev_camera_translation != core.camera_translation ||
				Outputs[output_index].prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) 
			{
				for (auto& c : viewer->core_list) 
				{
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs) 
				{
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}	
			}
		}
	}
}

void deformation_plugin::remove_output(const int output_index) 
{
	stop_all_minimizers_threads();
	viewer->erase_core(1 + output_index);
	viewer->erase_mesh(1 + output_index);
	Outputs.erase(Outputs.begin() + output_index);
	
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

void deformation_plugin::add_output() 
{
	stop_all_minimizers_threads();
	Outputs.push_back(OptimizationOutput(viewer, optimizer_type,linesearch_type));
	viewer->load_mesh_from_file(modelPath.c_str());
	Outputs[Outputs.size() - 1].ModelID = viewer->data_list[Outputs.size()].id;
	init_objective_functions(Outputs.size() - 1);
	//Update the scene
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	viewer->core(inputCoreID).is_animating = true;
	for (int i = 0; i < Outputs.size(); i++) 
	{
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
		viewer->core(Outputs[i].CoreID).is_animating = true;
	}
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

IGL_INLINE void deformation_plugin::post_resize(int w, int h)
{
	if (!viewer)
		return;
	if (view == app_utils::View::HORIZONTAL) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w - w * Outputs.size() * core_size, h);
		for (int i = 0; i < Outputs.size(); i++) 
		{
			Outputs[i].screen_position = ImVec2(w - w * (Outputs.size() - i) * core_size, 0);
			Outputs[i].screen_size = ImVec2(w * core_size, h);
			Outputs[i].results_window_position = Outputs[i].screen_position;
			Outputs[i].outputs_window_position = ImVec2(w - w * (Outputs.size() - (i + 1)) * core_size - 200, 0);
		}
	}
	if (view == app_utils::View::VERTICAL) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, Outputs.size() * h * core_size, w, h - Outputs.size() * h * core_size);
		for (int i = 0; i < Outputs.size(); i++) 
		{
			Outputs[i].screen_position = ImVec2(0, (Outputs.size() - i - 1) * h * core_size);
			Outputs[i].screen_size = ImVec2(w, h * core_size);
			Outputs[i].outputs_window_position = ImVec2(w-205, h - Outputs[i].screen_position[1] - Outputs[i].screen_size[1]);
			Outputs[i].results_window_position = ImVec2(0, Outputs[i].outputs_window_position[1]);
		}
	}
	if (view == app_utils::View::SHOW_INPUT_SCREEN_ONLY) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w, h);
		for (auto&o : Outputs) 
		{
			o.screen_position = ImVec2(w, h);
			o.screen_size = ImVec2(0, 0);
			o.results_window_position = o.screen_position;
			//o.outputs_window_position = 
		}
	}
 	if (view >= app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0) 
	{
 		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, 0, 0);
 		for (auto&o : Outputs) 
		{
 			o.screen_position = ImVec2(w, h);
 			o.screen_size = ImVec2(0, 0);
 			o.results_window_position = o.screen_position;
 		}
 		// what does this means?
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].screen_position = ImVec2(0, 0);
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].screen_size = ImVec2(w, h);
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].results_window_position = ImVec2(w*0.8, 0);
 	}		
	for (auto& o : Outputs)
		viewer->core(o.CoreID).viewport = Eigen::Vector4f(o.screen_position[0], o.screen_position[1], o.screen_size[0] + 1, o.screen_size[1] + 1);
	energies_window_position = ImVec2(0.1 * w, 0.8 * h);
	global_screen_size = ImVec2(w, h);
}

IGL_INLINE bool deformation_plugin::mouse_move(int mouse_x, int mouse_y)
{
	if (!isModelLoaded || IsMouseDraggingAnyWindow)
		return true;	
	if (ui.isChoosingCluster()) {
		pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point);
		return true;
	}
	if (ui.isTranslatingVertex()) {
		Eigen::RowVector3d vertex_pos = OutputModel(ui.Output_Index).V.row(ui.Vertex_Index);
		Eigen::RowVector3d translation = app_utils::computeTranslation(mouse_x, ui.down_mouse_x, mouse_y, ui.down_mouse_y, vertex_pos, OutputCore(ui.Output_Index));
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index))
			out.first.Energy_FixChosenVertices->translateConstraint(ui.Vertex_Index, translation);
		ui.down_mouse_x = mouse_x;
		ui.down_mouse_y = mouse_y;
		return true;
	}
	if (ui.isBrushingWeightInc() && pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
		double shift = (ui.ADD_DELETE == ADD) ? ADDING_WEIGHT_PER_HINGE_VALUE : -ADDING_WEIGHT_PER_HINGE_VALUE;
		const std::vector<int> brush_faces = Outputs[ui.Output_Index].FaceNeigh(ui.intersec_point.cast<double>(), brush_radius);
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
			out.first.Energy_auxBendingNormal->Incr_HingesWeights(brush_faces, shift);
			out.first.Energy_BendingNormal->Incr_HingesWeights(brush_faces, shift);
			out.first.Energy_auxSpherePerHinge->Incr_HingesWeights(brush_faces, shift);
		}
		return true;
	}
	if (ui.isBrushingWeightDec() && pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
		const std::vector<int> brush_faces = Outputs[ui.Output_Index].FaceNeigh(ui.intersec_point.cast<double>(), brush_radius);
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
			out.first.Energy_auxBendingNormal->setOne_HingesWeights(brush_faces);
			out.first.Energy_BendingNormal->setOne_HingesWeights(brush_faces);
			out.first.Energy_auxSpherePerHinge->setOne_HingesWeights(brush_faces);
		}
		return true;
	}
	
	int output_index, vertex_index;
	if (ui.isUsingDFS() && pick_vertex(output_index, vertex_index)) {
		ui.updateVerticesListOfDFS(InputModel().F, InputModel().V.rows(), vertex_index);
		return true;
	}

	if (ui.isUsingDFS() || ui.isBrushingWeightDec() || ui.isBrushingWeightInc())
		return true;
	return false;
}

std::vector<std::pair<OptimizationOutput&, int>> deformation_plugin::listOfOutputsToUpdate(const int out_index) {
	std::vector<std::pair<OptimizationOutput&, int>> vec;
	if (out_index<0 || out_index>Outputs.size())
		return {};
	if (UserInterface_UpdateAllOutputs) {
		for (int i = 0; i < Outputs.size(); i++) {
			vec.push_back({ Outputs[i],i });
		}
		return vec;
	}
	return { { Outputs[out_index],out_index } };
}

IGL_INLINE bool deformation_plugin::mouse_scroll(float delta_y) 
{
	if (!isModelLoaded || IsMouseDraggingAnyWindow || ImGui::IsAnyWindowHovered())
		return true;
	if (ui.isBrushing()) {
		brush_radius += delta_y * 0.005;
		brush_radius = std::max<float>(0.005, brush_radius);
		return true;
	}
	if (ui.isChoosingCluster()) {
		neighbor_distance += delta_y * 0.05;
		neighbor_distance = std::max<float>(0.005, neighbor_distance);
		return true;
	}
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_up(int button, int modifier) 
{
	IsMouseDraggingAnyWindow = false;

	int output_index, vertex_index;
	if (ui.isUsingDFS() && pick_vertex(output_index, vertex_index)) {
		ui.updateVerticesListOfDFS(InputModel().F, InputModel().V.rows(), vertex_index);
		for (auto& out : listOfOutputsToUpdate(output_index)) {
			out.first.Energy_auxBendingNormal->setZero_HingesWeights(ui.DFS_vertices_list);
			out.first.Energy_BendingNormal->setZero_HingesWeights(ui.DFS_vertices_list);
			out.first.Energy_auxSpherePerHinge->setZero_HingesWeights(ui.DFS_vertices_list);
		}
	}

	if (ui.isChoosingCluster() && pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
		std::vector<int> neigh_faces = Outputs[ui.Output_Index].getNeigh(neighbor_Type, InputModel().F, ui.Face_index, neighbor_distance);
		double shift = (ui.ADD_DELETE == ADD) ? 5 * ADDING_WEIGHT_PER_HINGE_VALUE : -5 * ADDING_WEIGHT_PER_HINGE_VALUE;
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
			out.first.Energy_auxBendingNormal->Incr_HingesWeights(neigh_faces, shift);
			out.first.Energy_BendingNormal->Incr_HingesWeights(neigh_faces, shift);
			out.first.Energy_auxSpherePerHinge->Incr_HingesWeights(neigh_faces, shift);
		}
	}
	ui.clear();
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_down(int button, int modifier) 
{
	bool LeftClick = (button == GLFW_MOUSE_BUTTON_LEFT);
	bool RightClick = (button == GLFW_MOUSE_BUTTON_MIDDLE);
	if (ImGui::IsAnyWindowHovered())
		IsMouseDraggingAnyWindow = true;
	ui.down_mouse_x = viewer->current_mouse_x;
	ui.down_mouse_y = viewer->current_mouse_y;
	
	if (ui.status == app_utils::UserInterfaceOptions::FIX_FACES && LeftClick) {
		//////..............
	}
	if (ui.status == app_utils::UserInterfaceOptions::FIX_FACES && RightClick) {
		//////..............
	}
	if (ui.status == app_utils::UserInterfaceOptions::FIX_VERTICES && LeftClick)
	{
		if (pick_vertex(ui.Output_Index, ui.Vertex_Index) && ui.Output_Index != INPUT_MODEL_SCREEN) {
			for (auto& out : listOfOutputsToUpdate(ui.Output_Index))
				out.first.Energy_FixChosenVertices->insertConstraint(ui.Vertex_Index, OutputModel(out.second).V);
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::FIX_VERTICES && RightClick) {
		if (pick_vertex(ui.Output_Index, ui.Vertex_Index) && ui.Output_Index != INPUT_MODEL_SCREEN)
			for (auto& out : listOfOutputsToUpdate(ui.Output_Index))
				out.first.Energy_FixChosenVertices->eraseConstraint(ui.Vertex_Index);
		ui.clear();
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR && LeftClick) {
		if (pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
			ui.ADD_DELETE = ADD;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR && RightClick) {
		if (pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
			ui.ADD_DELETE = DELETE;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && LeftClick) {
		if (pick_vertex(ui.Output_Index, ui.Vertex_Index)) {
			ui.DFS_Vertex_Index_FROM = ui.Vertex_Index;
			ui.ADD_DELETE = ADD;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && RightClick) {
		if (pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
			ui.ADD_DELETE = DELETE;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::ADJ_WEIGHTS && LeftClick) {
		ui.ADD_DELETE = ADD;
		ui.isActive = true;
		pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point);
	}
	if (ui.status == app_utils::UserInterfaceOptions::ADJ_WEIGHTS && RightClick) {
		ui.ADD_DELETE = DELETE;
		ui.isActive = true;
		pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point);
	}

	return false;
}

IGL_INLINE bool deformation_plugin::key_pressed(unsigned int key, int modifiers) 
{
	if ((key == 'a' || key == 'A') && modifiers == 1)
	{
		modelPath = OptimizationUtils::ProjectPath() + "\\models\\InputModels\\from_2k_to_10k\\island.off";
		isLoadNeeded = true;
	}
	if ((key == 's' || key == 'S') && modifiers == 1) {
		modelPath = OptimizationUtils::ProjectPath() + "\\models\\InputModels\\Bear_without_eyes.off";
		isLoadNeeded = true;
	}
	if (!isModelLoaded)
		return ImGuiMenu::key_pressed(key, modifiers);

	if ((key == 'c' || key == 'C') && modifiers == 1)
		clear_sellected_faces_and_vertices();
	if ((key == 'x' || key == 'X') && modifiers == 1) {
		if (face_coloring_Type == app_utils::Face_Colors::NO_COLORS) {
			face_coloring_Type = app_utils::Face_Colors::SPHERES_CLUSTERING;
			if (neighbor_Type == app_utils::Neighbor_Type::LOCAL_NORMALS)
				face_coloring_Type = app_utils::Face_Colors::NORMALS_CLUSTERING;
		}
		else if (face_coloring_Type == app_utils::Face_Colors::SIGMOID_PARAMETER) {
			face_coloring_Type = app_utils::Face_Colors::NO_COLORS;
		}	
		else {
			face_coloring_Type = app_utils::Face_Colors::SIGMOID_PARAMETER;
		}
	}
	
	if ((key == 'q' || key == 'Q') && modifiers == 1) 
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_NORMALS;
		face_coloring_Type = app_utils::Face_Colors::NORMALS_CLUSTERING;
		for (auto&out : Outputs) {
			out.showFacesNorm = false;
			out.showSphereEdges = out.showNormEdges = 
				out.showTriangleCenters = out.showSphereCenters = false;
		}
		for (OptimizationOutput& out : Outputs) {
			std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(out.totalObjective->objectiveList[0]);
			std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(out.totalObjective->objectiveList[1]);
			std::shared_ptr<BendingNormal> BN = std::dynamic_pointer_cast<BendingNormal>(out.totalObjective->objectiveList[2]);
			ABN->w = 0;
			BN->w = 1.6;
			AS->w = 0;
		}
	}
	if ((key == 'e' || key == 'E') && modifiers == 1)
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_NORMALS;
		face_coloring_Type = app_utils::Face_Colors::NORMALS_CLUSTERING;
		for (auto& out : Outputs) {
			out.showFacesNorm = true;
			out.showSphereEdges = out.showNormEdges =
				out.showTriangleCenters = out.showSphereCenters = false;
		}
		for (OptimizationOutput& out : Outputs) {
			std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(out.totalObjective->objectiveList[0]);
			std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(out.totalObjective->objectiveList[1]);
			std::shared_ptr<BendingNormal> BN = std::dynamic_pointer_cast<BendingNormal>(out.totalObjective->objectiveList[2]);
			ABN->w = 1.6;
			BN->w = 0;
			AS->w = 0;
		}
	}
	if ((key == 'w' || key == 'W') && modifiers == 1) 
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_SPHERE;
		face_coloring_Type = app_utils::Face_Colors::SPHERES_CLUSTERING;
		initSphereAuxVariables = OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS;
		init_aux_variables();
		for (auto&out : Outputs) {
			out.showSphereCenters = true;
			out.showSphereEdges = out.showNormEdges =
				out.showTriangleCenters = out.showFacesNorm = false;
		}
		for (OptimizationOutput& out : Outputs) 
		{
			for (auto& obj : out.totalObjective->objectiveList) 
			{
				std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(out.totalObjective->objectiveList[0]);
				std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(out.totalObjective->objectiveList[1]);
				std::shared_ptr<BendingNormal> BN = std::dynamic_pointer_cast<BendingNormal>(out.totalObjective->objectiveList[2]);
				ABN->w = 0;
				BN->w = 0;
				AS->w = 1.6;
			}
		}
	}
	
	if ((key == ' ') && modifiers == 1)
		isMinimizerRunning ? stop_all_minimizers_threads() : start_all_minimizers_threads();
	
	return ImGuiMenu::key_pressed(key, modifiers);
}

IGL_INLINE bool deformation_plugin::key_down(int key, int modifiers)
{
	if (key == '1')
		ui.status = app_utils::UserInterfaceOptions::FIX_VERTICES;
	else if (key == '2')
		ui.status = app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR;
	else if (key == '3')
		ui.status = app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR;
	else if (key == '4')
		ui.status = app_utils::UserInterfaceOptions::ADJ_WEIGHTS;
	else if (key == '5')
		ui.status = app_utils::UserInterfaceOptions::FIX_FACES;
	
	return ImGuiMenu::key_down(key, modifiers);
}

IGL_INLINE bool deformation_plugin::key_up(int key, int modifiers)
{
	ui.status = app_utils::UserInterfaceOptions::NONE;
	return ImGuiMenu::key_up(key, modifiers);
}

IGL_INLINE void deformation_plugin::shutdown()
{
	stop_all_minimizers_threads();
	ImGuiMenu::shutdown();
}

void deformation_plugin::draw_brush_sphere() 
{
	if (!ui.isBrushing())
		return;
	//prepare brush sphere
	const int samples = 100;
	Eigen::MatrixXd sphere(samples * samples, 3);
	Eigen::RowVector3d center = ui.intersec_point.cast<double>().transpose();
	int i, j;
	for (double alfa = 0, i = 0; alfa < 360; i++, alfa += int(360 / samples)) {
		for (double beta = 0, j = 0; beta < 360; j++, beta += int(360 / samples)) {
			Eigen::RowVector3d dir;
			dir << sin(alfa), cos(alfa)* cos(beta), sin(beta)* cos(alfa);
			if (i + samples * j < sphere.rows())
				sphere.row(i + samples * j) = dir * brush_radius + center;
		}
	}
	//update data for cores
	OutputModel(ui.Output_Index).add_points(sphere, ui.getBrushColor(model_color));
}

IGL_INLINE bool deformation_plugin::pre_draw() 
{
	if (!isModelLoaded)
		return ImGuiMenu::pre_draw();
	for (auto& out : Outputs)
		if (out.minimizer->progressed)
			update_data_from_minimizer();
	Update_view();
	update_parameters_for_all_cores();

	//Update Faces Colors
	follow_and_mark_selected_faces();
	InputModel().set_colors(Outputs[ActiveOutput].color_per_face);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_colors(Outputs[i].color_per_face);

	//Update Vertices Colors
	for (int oi = 0; oi < Outputs.size(); oi++) {
		auto& m = OutputModel(oi);
		auto& o = Outputs[oi];
		auto& AS = Outputs[oi].Energy_auxSpherePerHinge;
		m.point_size = 10;
		m.clear_points();
		m.clear_edges();

		if (ui.isTranslatingVertex())
			m.add_points(m.V.row(ui.Vertex_Index), Dragged_vertex_color.cast<double>().transpose());
		for (auto vi : o.Energy_FixChosenVertices->getConstraintsIndices())
			m.add_points(m.V.row(vi), Fixed_vertex_color.cast<double>().transpose());
		if (o.showFacesNorm)
			m.add_points(o.getFacesNorm(), o.color_per_face_norm);
		if (o.showTriangleCenters)
			m.add_points(o.getCenterOfFaces(), o.color_per_vertex_center);
		if (o.showSphereCenters)
			m.add_points(o.getCenterOfSphere(), o.color_per_sphere_center);
		if (o.showSphereEdges)
			m.add_edges(o.getCenterOfFaces(), o.getSphereEdges(), o.color_per_sphere_edge);
		if (o.showNormEdges)
			m.add_edges(o.getCenterOfFaces(), o.getFacesNorm(), o.color_per_norm_edge);
			
		// Update Vertices colors for UI sigmoid weights
		int num_hinges = AS->mesh_indices.num_hinges;
		const Eigen::VectorXi& x0_index = AS->x0_GlobInd;
		const Eigen::VectorXi& x1_index = AS->x1_GlobInd;
		double* hinge_val = AS->weight_PerHinge.host_arr;
		std::set<int> points_indices;
		for (int hi = 0; hi < num_hinges; hi++) {
			if (hinge_val[hi] == 0) {
				points_indices.insert(x0_index[hi]);
				points_indices.insert(x1_index[hi]);
			}
		}
		Eigen::MatrixXd points_pos(points_indices.size(), 3);
		auto& iter = points_indices.begin();
		for (int i = 0; i < points_pos.rows(); i++) {
			int v_index = *(iter++);
			points_pos.row(i) = m.V.row(v_index);
		}
		auto color = ui.colorM.cast<double>().replicate(1, points_indices.size()).transpose();
		m.add_points(points_pos, color);
	}

	for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
		if (ui.DFS_vertices_list.size()) {
			Eigen::MatrixXd points_pos(ui.DFS_vertices_list.size(), 3);
			int i = 0;
			for (int v_index : ui.DFS_vertices_list)
				points_pos.row(i++) = OutputModel(out.second).V.row(v_index);
			OutputModel(out.second).add_points(points_pos, ui.colorTry.cast<double>().transpose());
		}
	}

	draw_brush_sphere();
	InputModel().point_size = OutputModel(ActiveOutput).point_size;
	InputModel().set_points(OutputModel(ActiveOutput).points.leftCols(3), OutputModel(ActiveOutput).points.rightCols(3));
	return ImGuiMenu::pre_draw();
}

void deformation_plugin::change_minimizer_type(Cuda::OptimizerType type)
{
	optimizer_type = type;
	stop_all_minimizers_threads();
	init_aux_variables();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].updateActiveMinimizer(optimizer_type);
}

void deformation_plugin::Update_view() 
{
	for (auto& data : viewer->data_list)
		for (auto& out : Outputs)
			data.copy_options(viewer->core(inputCoreID), viewer->core(out.CoreID));
	for (auto& core : viewer->core_list)
		for (auto& data : viewer->data_list)
			viewer->data(data.id).set_visible(false, core.id);
	InputModel().set_visible(true, inputCoreID);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_visible(true, Outputs[i].CoreID);
	for (auto& core : viewer->core_list)
		core.is_animating = true;
}

void deformation_plugin::follow_and_mark_selected_faces() 
{
	for (int i = 0; i < Outputs.size(); i++) {
		Outputs[i].initFaceColors(
			InputModel().F.rows(),
			center_sphere_color,
			center_vertex_color,
			Color_sphere_edges,
			Color_normal_edge,
			face_norm_color);

		UpdateEnergyColors(i);
		//Mark the selected faces by brush
		auto& AS = Outputs[i].Energy_auxSpherePerHinge;
		for (int hi = 0; hi < AS->mesh_indices.num_hinges; hi++) {
			const int f0 = AS->hinges_faceIndex[hi][0];
			const int f1 = AS->hinges_faceIndex[hi][1];
			if (AS->weight_PerHinge.host_arr[hi] > 1) {
				const double alpha = (AS->weight_PerHinge.host_arr[hi] - 1.0f) / MAX_WEIGHT_PER_HINGE_VALUE;
				Outputs[i].shiftFaceColors(f0, alpha, model_color, ui.colorP);
				Outputs[i].shiftFaceColors(f1, alpha, model_color, ui.colorP);
			}
		}
	}

	//Mark the highlighted face & neighbors
	if (ui.isChoosingCluster()) {
		std::vector<int> neigh = Outputs[ui.Output_Index].getNeigh(neighbor_Type, InputModel().F, ui.Face_index, neighbor_distance);
		for (int fi : neigh)
			Outputs[ui.Output_Index].setFaceColors(fi, Neighbors_Highlighted_face_color.cast<double>());
		Outputs[ui.Output_Index].setFaceColors(ui.Face_index, Highlighted_face_color.cast<double>());
	}
		
	for (auto& o:Outputs) {
		//Mark the clusters if needed
		if (clusteringType == app_utils::ClusteringType::NO_CLUS && (face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING || face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)) {
			Eigen::MatrixX3d P = o.getFacesNormals();
			if (face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING) {
				Eigen::MatrixXd C = o.getCenterOfSphere();
				Eigen::VectorXd R = o.getRadiusOfSphere();
				for (int fi = 0; fi < C.rows(); fi++)
					P.row(fi) << C(fi, 0) * R(fi), C(fi, 1), C(fi, 2);
			}
			Eigen::RowVector3d Pmin(P.col(0).minCoeff(), P.col(1).minCoeff(), P.col(2).minCoeff());
			Eigen::RowVector3d Pmax(P.col(0).maxCoeff(), P.col(1).maxCoeff(), P.col(2).maxCoeff());
			for (int fi = 0; fi < P.rows(); fi++) {
				//set the values in the range [0, 1]
				for (int xyz = 0; xyz < 3; xyz++)
					P(fi, xyz) = (P(fi, xyz) - Pmin(xyz)) / (Pmax(xyz) - Pmin(xyz));
				//Add Brightness according to user weight...
				for (int col = 0; col < 3; col++)
					P(fi, col) = (clustering_brightness_w * P(fi, col)) + (1 - clustering_brightness_w);
				//set faces colors
				o.setFaceColors(fi, P.row(fi));
			}
			o.clustering_faces_colors = P;
			o.clustering_faces_indices = {};
		}
		else if (clusteringType == app_utils::ClusteringType::RADIUS && (face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING || face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)) {
			RadiusClustering(o.getValues(face_coloring_Type), Clustering_MSE_Threshold, o.clustering_faces_indices);
			o.clustering_colors.getFacesColors(o.clustering_faces_indices, InputModel().F.rows(), clustering_brightness_w, o.clustering_faces_colors);
			//set faces colors
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, o.clustering_faces_colors.row(fi).transpose());
		}
		else if (clusteringType == app_utils::ClusteringType::Agglomerative_hierarchical && (face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING || face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)) {
			//set faces colors
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, o.clustering_faces_colors.row(fi).transpose());
		}
		else if (face_coloring_Type == app_utils::Face_Colors::SIGMOID_PARAMETER) {
			auto& AS = o.Energy_auxSpherePerHinge;
			for (int hi = 0; hi < AS->mesh_indices.num_hinges; hi++) {
				const int f0 = AS->hinges_faceIndex[hi][0];
				const int f1 = AS->hinges_faceIndex[hi][1]; 
				const double log_minus_w = -log2(AS->Sigmoid_PerHinge.host_arr[hi]);
				const double alpha = log_minus_w / MAX_SIGMOID_PER_HINGE_VALUE;
				o.shiftFaceColors(f0, alpha, model_color, ui.colorP);
				o.shiftFaceColors(f1, alpha, model_color, ui.colorP);
			}
		}

		if (isChecking_SelfIntersection) {
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, model_color.cast<double>());
			for (auto& p : o.SelfIntersection_pairs) {
				o.setFaceColors(p.first, Eigen::Vector3d(1, 0, 0));
				o.setFaceColors(p.second, Eigen::Vector3d(0, 1, 0));
			}
		}
		
		if (isChecking_FlippedFaces) {
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, model_color.cast<double>());
			for (auto& p : o.flippedFaces_pairs) {
				o.setFaceColors(p.first, Eigen::Vector3d(1, 0, 0));
				o.setFaceColors(p.second, Eigen::Vector3d(0, 1, 0));
			}
		}
	}
}
	
igl::opengl::ViewerData& deformation_plugin::InputModel() 
{
	return viewer->data(inputModelID);
}

igl::opengl::ViewerData& deformation_plugin::OutputModel(const int index) 
{
	return viewer->data(Outputs[index].ModelID);
}

igl::opengl::ViewerCore& deformation_plugin::InputCore()
{
	return viewer->core(inputCoreID);
}

igl::opengl::ViewerCore& deformation_plugin::OutputCore(const int index) 
{
	return viewer->core(Outputs[index].CoreID);
}

bool deformation_plugin::pick_face(int& out_ind, int& f_ind, Eigen::Vector3f& intersec_point)
{
	f_ind = pick_face_per_core(InputModel().V, InputModel().F, app_utils::View::SHOW_INPUT_SCREEN_ONLY, intersec_point);
	out_ind = INPUT_MODEL_SCREEN;
	for (int i = 0; i < Outputs.size(); i++)
	{
		if (f_ind == NOT_FOUND)
		{
			f_ind = pick_face_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0 + i, intersec_point);
			out_ind = i;
		}
	}
	return (f_ind != NOT_FOUND);
}

int deformation_plugin::pick_face_per_core(
	Eigen::MatrixXd& V, 
	Eigen::MatrixXi& F, 
	int CoreIndex, 
	Eigen::Vector3f& intersec_point) 
{
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::SHOW_INPUT_SCREEN_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) 
	{
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::RowVector3d pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = NOT_FOUND;
	std::vector<igl::Hit> hits;
	igl::unproject_in_mesh(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, V, F, pt, hits);
	Eigen::Vector3f s, dir;
	igl::unproject_ray(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, s, dir);
	int fi = NOT_FOUND;
	if (hits.size() > 0) 
	{
		fi = hits[0].id;
		intersec_point = s + dir * hits[0].t;
	}
	return fi;
}

bool deformation_plugin::pick_vertex(int& o_ind, int& v_index) {
	v_index = pick_vertex_per_core(InputModel().V, InputModel().F, app_utils::View::SHOW_INPUT_SCREEN_ONLY);
	o_ind = INPUT_MODEL_SCREEN;
	for (int i = 0; i < Outputs.size(); i++) {
		if (v_index == NOT_FOUND) {
			v_index = pick_vertex_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0 + i);
			o_ind = i;
		}
	}
	return (v_index != NOT_FOUND);
}

int deformation_plugin::pick_vertex_per_core(
	Eigen::MatrixXd& V, 
	Eigen::MatrixXi& F, 
	int CoreIndex) 
{
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::SHOW_INPUT_SCREEN_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = NOT_FOUND;
	std::vector<igl::Hit> hits;
	unproject_in_mesh(
		Eigen::Vector2f(x, y), 
		viewer->core(CoreID).view,
		viewer->core(CoreID).proj, 
		viewer->core(CoreID).viewport, 
		V, 
		F, 
		pt, 
		hits
	);
	if (hits.size() > 0) 
	{
		int fi = hits[0].id;
		Eigen::RowVector3d bc;
		bc << 1.0 - hits[0].u - hits[0].v, hits[0].u, hits[0].v;
		bc.maxCoeff(&vi);
		vi = F(fi, vi);
	}
	return vi;
}

void deformation_plugin::checkGradients()
{
	stop_all_minimizers_threads();
	for (auto& o: Outputs) 
	{
		Eigen::VectorXd testX = Eigen::VectorXd::Random(o.totalObjective->objectiveList[0]->grad.size);
		o.totalObjective->checkGradient(testX);
		for (auto const &objective : o.totalObjective->objectiveList)
			objective->checkGradient(testX);
	}
}

void deformation_plugin::update_data_from_minimizer()
{	
	for (int i = 0; i < Outputs.size(); i++)
	{
		Eigen::MatrixXd V;
		auto& o = Outputs[i];
		o.minimizer->get_data(V, o.center_of_sphere, o.radiuses, o.normals);
		o.center_of_faces = OptimizationUtils::center_per_triangle(V, InputModel().F);

		Eigen::MatrixX3d N;
		igl::per_face_normals(V, OutputModel(i).F, N);
		auto BN = std::dynamic_pointer_cast<BendingNormal>(Outputs[i].totalObjective->objectiveList[2]);
		if (BN->w != 0) {
			o.normals = N;
		}
		
		OutputModel(i).set_vertices(V);
		OutputModel(i).compute_normals();
	}
}

void deformation_plugin::stop_all_minimizers_threads() {
	for (auto& o : Outputs)
		stop_one_minimizer_thread(o);
}

void deformation_plugin::stop_one_minimizer_thread(const OptimizationOutput o) {
	if (o.minimizer->is_running)
		o.minimizer->stop();
	while (o.minimizer->is_running);

	isMinimizerRunning = is_Any_Minizer_running();
}
void deformation_plugin::start_all_minimizers_threads() {
	for (auto& o : Outputs)
		start_one_minimizer_thread(o);
}
void deformation_plugin::start_one_minimizer_thread(const OptimizationOutput o) {
	stop_one_minimizer_thread(o);
	std::thread minimizer_thread1 = std::thread(&Minimizer::run_new, o.minimizer.get());
	std::thread minimizer_thread2 = std::thread(&Minimizer::RunSymmetricDirichletGradient, o.minimizer.get());
	minimizer_thread1.detach();
	minimizer_thread2.detach();
	
	isMinimizerRunning = true;
}
bool deformation_plugin::is_Any_Minizer_running() {
	for (auto&o : Outputs)
		if (o.minimizer->is_running)
			return true;
	return false;
}

void deformation_plugin::init_aux_variables() 
{
	stop_all_minimizers_threads();
	if (InitMinimizer_NeighLevel_From < 1)
		InitMinimizer_NeighLevel_From = 1;
	if (InitMinimizer_NeighLevel_From > InitMinimizer_NeighLevel_To)
		InitMinimizer_NeighLevel_To = InitMinimizer_NeighLevel_From;
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].initMinimizers(
			OutputModel(i).V,
			OutputModel(i).F,
			initSphereAuxVariables,
			InitMinimizer_NeighLevel_From,
			InitMinimizer_NeighLevel_To,
			radius_length_minus_normal);
}

void deformation_plugin::run_one_minimizer_iter() 
{
	stop_all_minimizers_threads();
	for (auto& o : Outputs)
		o.minimizer->run_one_iteration();
}

void deformation_plugin::init_objective_functions(const int index)
{
	Eigen::MatrixXd V = OutputModel(index).V;
	Eigen::MatrixX3i F = OutputModel(index).F;
	stop_all_minimizers_threads();
	if (V.rows() == 0 || F.rows() == 0)
		return;
	// initialize the energy
	std::cout << console_color::yellow << "-------Energies, begin-------" << std::endl;
	std::shared_ptr <AuxBendingNormal> auxBendingNormal = std::make_unique<AuxBendingNormal>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_auxBendingNormal = auxBendingNormal;
	std::shared_ptr <AuxSpherePerHinge> auxSpherePerHinge = std::make_unique<AuxSpherePerHinge>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_auxSpherePerHinge = auxSpherePerHinge;
	std::shared_ptr <STVK> stvk = std::make_unique<STVK>(V, F);
	std::shared_ptr <SDenergy> sdenergy = std::make_unique<SDenergy>(V, F);
	std::shared_ptr <FixAllVertices> fixAllVertices = std::make_unique<FixAllVertices>(V, F);
	std::shared_ptr <fixRadius> FixRadius = std::make_unique<fixRadius>(V, F);
	std::shared_ptr <UniformSmoothness> uniformSmoothness = std::make_unique<UniformSmoothness>(V, F);
	std::shared_ptr <BendingNormal> bendingNormal = std::make_unique<BendingNormal>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_BendingNormal = bendingNormal;

	//Add User Interface Energies
	auto fixChosenVertices = std::make_shared<FixChosenConstraints>(V, F);
	Outputs[index].Energy_FixChosenVertices = fixChosenVertices;

	//init total objective
	Outputs[index].totalObjective = std::make_shared<TotalObjective>(V, F);
	Outputs[index].totalObjective->objectiveList.clear();
	auto add_obj = [&](std::shared_ptr< ObjectiveFunction> obj) 
	{
		Outputs[index].totalObjective->objectiveList.push_back(move(obj));
	};
	add_obj(auxSpherePerHinge);
	add_obj(auxBendingNormal);
	add_obj(bendingNormal);
	add_obj(stvk);
	add_obj(sdenergy);
	add_obj(fixAllVertices);
	add_obj(fixChosenVertices);
	add_obj(FixRadius);
	add_obj(uniformSmoothness);
	std::cout  << "-------Energies, end-------" << console_color::white << std::endl;
	init_aux_variables();
}

void deformation_plugin::UpdateEnergyColors(const int index) 
{
	int numF = OutputModel(index).F.rows();
	Eigen::VectorXd DistortionPerFace(numF);
	DistortionPerFace.setZero();
	if (faceColoring_type == 0) { // No colors
		DistortionPerFace.setZero();
	}
	else if (faceColoring_type == 1) { // total energy
		for (auto& obj: Outputs[index].totalObjective->objectiveList) {
			// calculate the distortion over all the energies
			if ((obj->Efi.size() != 0) && (obj->w != 0))
				DistortionPerFace += obj->Efi * obj->w;
		}
	}
	else {
		auto& obj = Outputs[index].totalObjective->objectiveList[faceColoring_type - 2];
		if ((obj->Efi.size() != 0) && (obj->w != 0))
			DistortionPerFace = obj->Efi * obj->w;
	}
	Eigen::VectorXd alpha_vec = DistortionPerFace / (Max_Distortion+1e-8);
	Eigen::VectorXd beta_vec = Eigen::VectorXd::Ones(numF) - alpha_vec;
	Eigen::MatrixXd alpha(numF, 3), beta(numF, 3);
	alpha = alpha_vec.replicate(1, 3);
	beta = beta_vec.replicate(1, 3);
	//calculate low distortion color matrix
	Eigen::MatrixXd LowDistCol = model_color.cast <double>().replicate(1, numF).transpose();
	//calculate high distortion color matrix
	Eigen::MatrixXd HighDistCol = Vertex_Energy_color.cast <double>().replicate(1, numF).transpose();
	Outputs[index].color_per_face = beta.cwiseProduct(LowDistCol) + alpha.cwiseProduct(HighDistCol);
}
