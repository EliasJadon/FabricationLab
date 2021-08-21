#pragma once

#include <iostream>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/unproject_in_mesh.h>
#include <igl/unproject_ray.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/adjacency_matrix.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/edge_lengths.h>
#include <igl/boundary_loop.h>
#include <imgui/imgui.h>
#include <chrono>
#include <vector>
#include <queue>
#include "unique_colors.h"

#include "../../libs/Minimizer.h"
#include "../../libs/STVK.h"
#include "../../libs/SDenergy.h"
#include "../../libs/FixAllVertices.h"
#include "../../libs/AuxBendingNormal.h"
#include "../../libs/AuxSpherePerHinge.h"
#include "../../libs/FixChosenConstraints.h"
#include "../../libs/fixRadius.h"
#include "../../libs/UniformSmoothness.h"
#include "../../libs/ClusterHard.h"
#include "../../libs/BendingNormal.h"

#define RED_COLOR Eigen::Vector3f(1, 0, 0)
#define BLUE_COLOR Eigen::Vector3f(0, 0, 1)
#define GREEN_COLOR Eigen::Vector3f(0, 1, 0)
#define GOLD_COLOR Eigen::Vector3f(1, 215.0f / 255.0f, 0)
#define GREY_COLOR Eigen::Vector3f(0.75, 0.75, 0.75)
#define WHITE_COLOR Eigen::Vector3f(1, 1, 1)
#define BLACK_COLOR Eigen::Vector3f(0, 0, 0)
#define M_PI 3.14159

namespace app_utils {
	enum Face_Colors { 
		NO_COLORS, 
		NORMALS_CLUSTERING, 
		SPHERES_CLUSTERING,
		SIGMOID_PARAMETER
	};
	enum View {
		HORIZONTAL = 0,
		VERTICAL,
		SHOW_INPUT_SCREEN_ONLY,
		SHOW_OUTPUT_SCREEN_ONLY_0
	};
	enum Neighbor_Type {
		CURR_FACE,
		LOCAL_SPHERE,
		GLOBAL_SPHERE,
		LOCAL_NORMALS,
		GLOBAL_NORMALS
	};
	enum UserInterfaceOptions { 
		NONE,
		FIX_VERTICES,
		FIX_FACES,
		BRUSH_WEIGHTS_INCR,
		BRUSH_WEIGHTS_DECR,
		ADJ_WEIGHTS
	};
	enum ClusteringType { NO_CLUS, RADIUS, Agglomerative_hierarchical };
	
	static bool writeOFFwithColors(
		const std::string& path,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::MatrixXd& C)
	{
		if (V.cols() != 3 || F.cols() != 3 || C.cols() != 3)
			return false;
		if (V.rows() <= 0 || F.rows() <= 0 || F.rows() != C.rows())
			return false;

		std::ofstream myfile;
		myfile.open(path);
		myfile << "OFF\n";
		myfile << V.rows() << " " << F.rows() << " 0\n";
		for (int vi = 0; vi < V.rows(); vi++) {
			myfile << V(vi, 0) << " " << V(vi, 1) << " " << V(vi, 2) << "\n";
		}
		for (int fi = 0; fi < F.rows(); fi++) {
			myfile << "3 " << F(fi, 0) << " " << F(fi, 1) << " " << F(fi, 2) << " ";
			myfile << int(255 * C(fi, 0)) << " " << int(255 * C(fi, 1)) << " " << int(255 * C(fi, 2)) << "\n";
		}
		myfile.close();
		return true;
	}

	static std::vector<int> findPathVertices_usingDFS(
		const std::vector<std::vector<int> >& A, 
		const int s, 
		const int target,
		const int numV) 
	{
		if (s < 0 || target < 0 || s >= numV || target >= numV)
			return {};
		if (s == target)
			return { s };

		std::vector<bool> seen(numV, false);
		std::vector<std::vector<std::pair<int, int>>> Vertices_per_level;
		Vertices_per_level.push_back({ {s,-1} });
		seen[s] = true;
		bool finish = false;
		while (!finish) {
			std::vector<std::pair<int, int>> currV;
			const int level = Vertices_per_level.size() - 1;
			for (std::pair<int, int> new_s : Vertices_per_level[level]) {
				for (int neighbour : A[new_s.first]) {
					if (neighbour == target) {
						finish = true;
					}
					if (!seen[neighbour]) {
						currV.push_back({ neighbour,new_s.first });
						seen[neighbour] = true;
					}
				}
			}
			Vertices_per_level.push_back(currV);
		}

		//Part 2: find vertices path
		std::vector<int> path;
		path.push_back(target);
		int last_level = Vertices_per_level.size() - 1;
		while (last_level > 0) {
			int v_target = path[path.size() - 1];
			for (auto& v : Vertices_per_level[last_level])
				if (v.first == v_target) {
					path.push_back(v.second);
				}
			last_level--;
		}
		return path;
	}

	static std::string CurrentTime() {
		char date_buffer[80] = { 0 };
		{
			time_t rawtime_;
			struct tm* timeinfo_;
			time(&rawtime_);
			timeinfo_ = localtime(&rawtime_);
			strftime(date_buffer, 80, "_%H_%M_%S__%d_%m_%Y", timeinfo_);
		}
		return std::string(date_buffer);
	}

	static bool writeTXTFile(
		const std::string& path,
		const std::string& modelName,
		const bool isSphere,
		const std::vector<std::vector<int>>& clustering_faces_indices,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::MatrixXd& C,
		const Eigen::VectorXd& Radiuses,
		const Eigen::MatrixX3d& Centers)
	{
		if (V.cols() != 3 || F.cols() != 3 || C.cols() != 3)
			return false;
		if (V.rows() <= 0 || F.rows() <= 0 || F.rows() != C.rows())
			return false;


		std::ofstream myfile;
		myfile.open(path);
		myfile << "\n\n===============================================\n";
		myfile << "Model name: \t"						<< modelName << "\n";
		myfile << "Num Faces: \t"						<< F.rows() << "\n";
		myfile << "Num Vertices: \t"					<< V.rows() << "\n";
		if (isSphere) {
			myfile << "Num spheres: \t" << clustering_faces_indices.size() << "\n";
		}
		else {
			myfile << "Num polygons: \t" << clustering_faces_indices.size() << "\n";
			myfile << "-----------------------List of polygons:" << "\n";
		}
		myfile << "===============================================\n\n\n";
		
		for (int ci = 0; ci < clustering_faces_indices.size(); ci++) {
			myfile << "\n";
			//calculating the avg center&radius for each group/cluster
			double avgRadius = 0;
			Eigen::RowVector3d avgCenter(0, 0, 0), avgColor(0, 0, 0);
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				if (isSphere) {
					avgRadius += Radiuses[face_index];
					avgCenter = avgCenter + Centers.row(face_index);
				}
				avgColor = avgColor + C.row(face_index);
			}
			if (isSphere) {
				avgRadius /= clustering_faces_indices[ci].size();
				avgCenter /= clustering_faces_indices[ci].size();
			}
			avgColor /= clustering_faces_indices[ci].size();
			

			//output data
			if (isSphere) {
				myfile << "Sphere ID:\t" << ci << "\n";
				myfile << "Radius length: " << avgRadius << "\n";
				myfile << "Center point: " << "(" << avgCenter(0) << ", " << avgCenter(1) << ", " << avgCenter(2) << ")" << "\n";
			}
			else {
				myfile << "Polygon ID:\t" << ci << "\n";
			}
			
			myfile << "color: " << "(" << avgColor(0) << ", " << avgColor(1) << ", " << avgColor(2) << ")" << "\n";
			myfile << "Num faces: " << clustering_faces_indices[ci].size() << "\n";
			myfile << "faces list: ";
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				myfile << face_index << ", ";
			}
			myfile << "\n";
			myfile << "----------------------------\n";
		}
		
		myfile.close();
		return true;
	}

	static double getMaxMSE_1Cluster(const Eigen::MatrixX4d& values, const std::vector<int>& clus)
	{
		double max = std::numeric_limits<double>::min();
		for (int f1 = 0; f1 < clus.size(); f1++) {
			for (int f2 = f1 + 1; f2 < clus.size(); f2++) {
				double MSE = (values.row(clus[f1]) - values.row(clus[f2])).squaredNorm();
				max = MSE > max ? MSE : max;
			}
		}
		return max;
	}

	static double getMinMSE_1Cluster(const Eigen::MatrixX4d& values, const std::vector<int>& clus)
	{
		double min = std::numeric_limits<double>::max();
		for (int f1 = 0; f1 < clus.size(); f1++) {
			for (int f2 = f1 + 1; f2 < clus.size(); f2++) {
				double MSE = (values.row(clus[f1]) - values.row(clus[f2])).squaredNorm();
				min = MSE < min ? MSE : min;
			}
		}
		return min;
	}

	static double getAvgMSE(
		const Eigen::MatrixX4d& values,
		const std::vector<std::vector<int>>& clusters)
	{
		double sum = 0;
		double count = 0;
		for (std::vector<int> clus : clusters) {
			for (int f1 = 0; f1 < clus.size(); f1++) {
				for (int f2 = f1 + 1; f2 < clus.size(); f2++) {
					double MSE = (values.row(clus[f1]) - values.row(clus[f2])).squaredNorm();
					sum += MSE;
					count++;
				}
			}
		}
		return (sum / count);
	}

	static double getStdMSE(
		const Eigen::MatrixX4d& values,
		const std::vector<std::vector<int>>& clusters)
	{
		double sum = 0;
		double count = 0;
		const double avg = getAvgMSE(values, clusters);
		for (std::vector<int> clus : clusters) {
			for (int f1 = 0; f1 < clus.size(); f1++) {
				for (int f2 = f1 + 1; f2 < clus.size(); f2++) {
					double MSE = (values.row(clus[f1]) - values.row(clus[f2])).squaredNorm();
					sum += pow(MSE - avg, 2);
					count++;
				}
			}
		}
		return (sum / count);
	}

	static double getMinMSE(
		const Eigen::MatrixX4d& values,
		const std::vector<std::vector<int>>& clusters)
	{
		double min = std::numeric_limits<double>::max();
		for (std::vector<int> c : clusters) {
			double ClusMSE = getMinMSE_1Cluster(values, c);
			min = ClusMSE < min ? ClusMSE : min;
		}
		return min;
	}
	
	static double getMaxMSE(
		const Eigen::MatrixX4d& values,
		const std::vector<std::vector<int>>& clusters)
	{
		double max = std::numeric_limits<double>::min();
		for (std::vector<int> c : clusters) {
			double ClusMSE = getMaxMSE_1Cluster(values, c);
			max = ClusMSE > max ? ClusMSE : max;
		}
		return max;
	}

	static void getPlanarCoordinates(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const std::vector<int>& clus) 
	{
		//get the boundary vertices
		Eigen::MatrixXi clus_F(clus.size(), 3);
		for (int i = 0; i < clus.size(); i++) {
			clus_F.row(i) = F.row(clus[i]);
		}
		Eigen::VectorXi bnd;
		igl::boundary_loop(clus_F, bnd);

		//get the best-plane edges
		Eigen::RowVector3d argminE1, argminE2;
		double min = std::numeric_limits<double>::max();
		for (int v1 = 1; v1 < bnd.size(); v1++) {
			for (int v2 = v1 + 1; v2 < bnd.size(); v2++) {
				Eigen::RowVector3d e1 = (V.row(bnd[v1]) - V.row(bnd[0])).normalized();
				Eigen::RowVector3d e2 = (V.row(bnd[v2]) - V.row(bnd[0])).normalized();
				double dot2 = pow(e1.dot(e2), 2);
				if (dot2 < min) {
					min = dot2;
					argminE1 = e1;
					argminE2 = e2;
				}
			}
		}

		//get axis X,Y,Z
		Eigen::RowVector3d X_axis = (argminE1).normalized();
		Eigen::RowVector3d Z_axis = (X_axis.cross(argminE2)).normalized();
		Eigen::RowVector3d Y_axis = (X_axis.cross(Z_axis)).normalized();

		//put all vertices on the plane
		std::vector<std::pair<double, double>> plane_coordinates;
		plane_coordinates.clear();
		for (int vi = 0; vi < bnd.size(); vi++) {
			Eigen::RowVector3d vec_V = V.row(bnd[vi]) - V.row(bnd[0]);
			double x_coordinate = vec_V.dot(X_axis);
			double y_coordinate = vec_V.dot(Y_axis);
			plane_coordinates.push_back(std::pair<double, double>(x_coordinate, y_coordinate));
		}

		//print polygon
		std::cout << "Polygon((" << plane_coordinates[0].first
			<< ", " << plane_coordinates[0].second << ")";
		for (int i = 1; i < plane_coordinates.size(); i++) {
			std::cout << ", (" << plane_coordinates[i].first
				<< ", " << plane_coordinates[i].second << ")";
		}
		std::cout << ")\n\n";
	}

	static Eigen::RowVector3d computeTranslation(
		const int mouse_x, 
		const int from_x, 
		const int mouse_y, 
		const int from_y, 
		const Eigen::RowVector3d pt3D,
		igl::opengl::ViewerCore& core) 
	{
		Eigen::Matrix4f modelview = core.view;
		//project the given point (typically the handle centroid) to get a screen space depth
		Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(), modelview, core.proj, core.viewport);
		float depth = proj[2];
		double x, y;
		Eigen::Vector3f pos1, pos0;
		//unproject from- and to- points
		x = mouse_x;
		y = core.viewport(3) - mouse_y;
		pos1 = igl::unproject(Eigen::Vector3f(x, y, depth), modelview, core.proj, core.viewport);
		x = from_x;
		y = core.viewport(3) - from_y;
		pos0 = igl::unproject(Eigen::Vector3f(x, y, depth), modelview, core.proj, core.viewport);
		//translation is the vector connecting the two
		Eigen::Vector3f translation;
		translation = pos1 - pos0;
		return Eigen::RowVector3d(translation(0), translation(1), translation(2));
	}

	static int calculateHinges(std::vector<Eigen::Vector2d>& hinges_faceIndex, const Eigen::MatrixX3i& F) {
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		assert(TT.size() == F.rows());
		hinges_faceIndex.clear();

		///////////////////////////////////////////////////////////
		//Part 1 - Find unique hinges
		for (int fi = 0; fi < TT.size(); fi++) {
			std::vector< std::vector<int>> CurrFace = TT[fi];
			assert(CurrFace.size() == 3 && "Each face should be a triangle (not square for example)!");
			for (std::vector<int> hinge : CurrFace) {
				if (hinge.size() == 1) {
					//add this "hinge"
					int FaceIndex1 = fi;
					int FaceIndex2 = hinge[0];

					if (FaceIndex2 < FaceIndex1) {
						//Skip
						//This hinge already exists!
						//Empty on purpose
					}
					else {
						hinges_faceIndex.push_back(Eigen::Vector2d(FaceIndex1, FaceIndex2));
					}
				}
				else if (hinge.size() == 0) {
					//Skip
					//This triangle has no another adjacent triangle on that edge
					//Empty on purpose
				}
				else {
					//We shouldn't get here!
					//The mesh is invalid
					assert("Each triangle should have only one adjacent triangle on each edge!");
				}

			}
		}
		return hinges_faceIndex.size(); // num_hinges
	}
	
	static std::string ExtractModelName(const std::string& str)
	{
		size_t head, tail;
		head = str.find_last_of("/\\");
		tail = str.find_last_of("/.");
		return (str.substr((head + 1), (tail - head - 1)));
	}
	
	static bool IsMesh2D(const Eigen::MatrixXd& V) {
		return (V.col(2).array() == 0).all();
	}

	static char* build_view_names_list(const int size) {
		std::string cStr("");
		cStr += "Horizontal";
		cStr += '\0';
		cStr += "Vertical";
		cStr += '\0';
		cStr += "InputOnly";
		cStr += '\0';
		for (int i = 0; i < size; i++) {
			std::string sts;
			sts = "OutputOnly " + std::to_string(i);
			cStr += sts.c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static char* build_inputColoring_list(const int size) {
		std::string cStr("");
		cStr += "None";
		cStr += '\0';
		for (int i = 0; i < size; i++) {
			cStr += "Output ";
			cStr += std::to_string(i).c_str();
			cStr += '\0';
		}
		cStr += '\0';
		
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		
		return comboList;
	}

	static char* build_color_energies_list(const std::shared_ptr<TotalObjective>& totalObjective) {
		std::string cStr("");
		cStr += "No colors";
		cStr += '\0';
		cStr += "Total energy";
		cStr += '\0';
		for (auto& obj : totalObjective->objectiveList) {
			cStr += (obj->name).c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static char* build_outputs_list(const int numOutputs) {
		std::string cStr("");
		for (int i = 0; i < numOutputs; i++) {
			cStr += std::to_string(i);
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static Eigen::RowVector3d get_face_avg(
		const igl::opengl::ViewerData& model,
		const int Translate_Index)
	{
		Eigen::RowVector3d avg; avg << 0, 0, 0;
		Eigen::RowVector3i face = model.F.row(Translate_Index);
		avg += model.V.row(face[0]);
		avg += model.V.row(face[1]);
		avg += model.V.row(face[2]);
		avg /= 3;
		return avg;
	}

	static double TriMin(const double v1, const double v2, const double v3) {
		return std::min<double>(std::min<double>(v1, v2), v3);
	}

	static double TriMax(const double v1, const double v2, const double v3) {
		return std::max<double>(std::max<double>(v1, v2), v3);
	}

	static bool grtEqual(const double v1, const double v2) {
		double diff = v1 - v2;
		if (diff < 1e-3 && diff > -1e-3)
			return true;
		return diff >= 0;
	}

	static bool lessEqual(const double v1, const double v2) {
		return grtEqual(v2, v1);
	}

	static bool areProjectionsSeparated(
		const double p0,
		const double p1,
		const double p2,
		const double q0,
		const double q1,
		const double q2)
	{
		const double min_p = TriMin(p0, p1, p2);
		const double max_p = TriMax(p0, p1, p2);
		const double min_q = TriMin(q0, q1, q2);
		const double max_q = TriMax(q0, q1, q2);
		return ((grtEqual(min_p, max_q)) || (lessEqual(max_p, min_q)));
	}

	/**
	 * @function
	 * @param {THREE.Triangle} t1 - Triangular face
	 * @param {THREE.Triangle} t2 - Triangular face
	 * @returns {boolean} Whether the two triangles intersect
	 */
	static bool doTrianglesIntersect(
		const unsigned int t1,
		const unsigned int t2,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F)
	{
		/*
		Adapated from section "4.1 Separation of Triangles" of:
		 - [Dynamic Collision Detection using Oriented Bounding Boxes]
		 (https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf)
		*/

		// Triangle 1:
		Eigen::RowVector3d A0 = V.row(F(t1, 0));
		Eigen::RowVector3d A1 = V.row(F(t1, 1));
		Eigen::RowVector3d A2 = V.row(F(t1, 2));
		Eigen::RowVector3d E0 = A1 - A0;
		Eigen::RowVector3d E1 = A2 - A0;
		Eigen::RowVector3d E2 = A2 - A1;
		Eigen::RowVector3d N = E0.cross(E1);
		// Triangle 2:
		Eigen::RowVector3d B0 = V.row(F(t2, 0));
		Eigen::RowVector3d B1 = V.row(F(t2, 1));
		Eigen::RowVector3d B2 = V.row(F(t2, 2));
		Eigen::RowVector3d F0 = B1 - B0;
		Eigen::RowVector3d F1 = B2 - B0;
		Eigen::RowVector3d F2 = B2 - B1;
		Eigen::RowVector3d M = F0.cross(F1);
		Eigen::RowVector3d D = B0 - A0;

		// Only potential separating axes for non-parallel and non-coplanar triangles are tested.
		// Seperating axis: N
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = 0;
			const double q0 = N.dot(D);
			const double q1 = q0 + N.dot(F0);
			const double q2 = q0 + N.dot(F1);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Separating axis: M
		{
			const double p0 = 0;
			const double p1 = M.dot(E0);
			const double p2 = M.dot(E1);
			const double q0 = M.dot(D);
			const double q1 = q0;
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E0 ª F0
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = -(N.dot(F0));
			const double q0 = E0.cross(F0).dot(D);
			const double q1 = q0;
			const double q2 = q0 + M.dot(E0);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E0 ª F1
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = -(N.dot(F1));
			const double q0 = E0.cross(F1).dot(D);
			const double q1 = q0 - M.dot(E0);
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E0 ª F2
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = -(N.dot(F2));
			const double q0 = E0.cross(F2).dot(D);
			const double q1 = q0 - M.dot(E0);
			const double q2 = q1;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E1 ª F0
		{
			const double p0 = 0;
			const double p1 = N.dot(F0);
			const double p2 = 0;
			const double q0 = E1.cross(F0).dot(D);
			const double q1 = q0;
			const double q2 = q0 + M.dot(E1);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E1 ª F1
		{
			const double p0 = 0;
			const double p1 = N.dot(F1);
			const double p2 = 0;
			const double q0 = E1.cross(F1).dot(D);
			const double q1 = q0 - M.dot(E1);
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E1 ª F2
		{
			const double p0 = 0;
			const double p1 = N.dot(F2);
			const double p2 = 0;
			const double q0 = E1.cross(F2).dot(D);
			const double q1 = q0 - M.dot(E1);
			const double q2 = q1;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E2 ª F0
		{
			const double p0 = 0;
			const double p1 = N.dot(F0);
			const double p2 = p1;
			const double q0 = E2.cross(F0).dot(D);
			const double q1 = q0;
			const double q2 = q0 + M.dot(E2);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E2 ª F1
		{
			const double p0 = 0;
			const double p1 = N.dot(F1);
			const double p2 = p1;
			const double q0 = E2.cross(F1).dot(D);
			const double q1 = q0 - M.dot(E2);
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		// Seperating axis: E2 ª F2
		{
			const double p0 = 0;
			const double p1 = N.dot(F2);
			const double p2 = p1;
			const double q0 = E2.cross(F2).dot(D);
			const double q1 = q0 - M.dot(E2);
			const double q2 = q1;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2))
				return false;
		}
		return true;
	}

	static std::vector<std::pair<int, int>> getFlippedFaces(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const std::vector<Eigen::Vector2d>& hinges_faceIndex)
	{
		Eigen::MatrixX3d N;
		igl::per_face_normals(V, F, N);
		std::vector<std::pair<int, int>> result;
		for (auto& h : hinges_faceIndex) {
			const int f1 = h[0];
			const int f2 = h[1];
			Eigen::RowVector3d N1 = N.row(f1).normalized();
			Eigen::RowVector3d N2 = N.row(f2).normalized();
			const double diff = (N1 + N2).squaredNorm();
			if (diff < 0.1) {
				result.push_back(std::pair<int, int>(f1, f2));
				std::cout << "Found a flipped-face between: " << f1 << ", " << f2 << std::endl;
			}
		}
		return result;
	}
}

#define INPUT_MODEL_SCREEN -1
#define NOT_FOUND -1
#define ADD false
#define DELETE true

class UI {
public:
	app_utils::UserInterfaceOptions status;
	bool isActive;
	int Vertex_Index, Output_Index, Face_index;
	int down_mouse_x, down_mouse_y;
	bool ADD_DELETE;
	Eigen::Vector3f intersec_point;
	Eigen::Vector3f colorP, colorM, colorTry;
	std::vector<int> DFS_vertices_list;
	int DFS_Vertex_Index_FROM;

	UI() {
		status = app_utils::UserInterfaceOptions::NONE;
		isActive = false;
		Output_Index = Face_index = Vertex_Index = NOT_FOUND;
		down_mouse_x = down_mouse_y = NOT_FOUND;
		colorP = Eigen::Vector3f(51 / 255.0f, 1, 1);
		colorM = Eigen::Vector3f(1, 10 / 255.0f, 1);
		colorTry = Eigen::Vector3f(1, 200 / 255.0f, 1);
	}

	void updateVerticesListOfDFS(const Eigen::MatrixXi F, const int numV, const int v_to) {
		Eigen::SparseMatrix<int> A;
		igl::adjacency_matrix(F, A);
		//convert sparse matrix into vector representation
		std::vector<std::vector<int>> adj;
		adj.resize(numV);
		for (int k = 0; k < A.outerSize(); ++k)
			for (Eigen::SparseMatrix<int>::InnerIterator it(A, k); it; ++it)
				adj[it.row()].push_back(it.col());
		//get the vertices list by DFS
		DFS_vertices_list = app_utils::findPathVertices_usingDFS(adj, DFS_Vertex_Index_FROM, v_to, numV);
	}

	bool isChoosingCluster() {
		return (status == app_utils::UserInterfaceOptions::ADJ_WEIGHTS && isActive && Face_index != NOT_FOUND);
	}
	bool isUsingDFS() {
		return status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && ADD_DELETE == ADD && isActive;
	}
	bool isTranslatingVertex() {
		return (status == app_utils::UserInterfaceOptions::FIX_VERTICES && isActive);
	}
	bool isBrushingWeightInc() {
		return (status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR && isActive);
	}
	bool isBrushingWeightDec() {
		return status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && ADD_DELETE == DELETE && isActive;
	}
	bool isBrushing() {
		return (isBrushingWeightInc() || isBrushingWeightDec()) && Face_index != NOT_FOUND;
	}

	Eigen::RowVector3d getBrushColor(const Eigen::Vector3f& model_color) {
		if (ADD_DELETE == ADD && isBrushingWeightInc())
			return colorP.cast<double>().transpose();
		return model_color.cast<double>().transpose();
	}
	void clear() {
		DFS_vertices_list.clear();
		isActive = false;
		Output_Index = Face_index = Vertex_Index = NOT_FOUND;
		down_mouse_x = down_mouse_y = NOT_FOUND;
		DFS_Vertex_Index_FROM = NOT_FOUND;
	}
	
};

