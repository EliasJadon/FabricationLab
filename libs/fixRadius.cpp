#include "fixRadius.h"

fixRadius::fixRadius(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunction{ V,F }
{
	name = "fix Radius";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

fixRadius::~fixRadius() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double fixRadius::value(Cuda::Array<double>& curr_x, const bool update) {
	double value = 0;
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		double R = getR(curr_x, fi);
		
		if (R < (min / alpha))
			value += pow(alpha * R - min, 2);
		if (R > (max / alpha))
			value += pow(alpha * R - max, 2);
		else {
			//val = pow(sin(alpha * M_PI * R), 2);
			double rounded_R = round(alpha * R) / (double)alpha;
			value += pow(R - rounded_R, 2);
		}
	}
	if (update)
		energy_value = value;
	return value;
}

void fixRadius::gradient(Cuda::Array<double>& X,const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const int startR = mesh_indices.startR;
		double R = getR(X, fi);

		if (R < (min / alpha))
			grad.host_arr[fi + startR] += 2 * alpha * (alpha * R - min);
		if (R > (max / alpha))
			grad.host_arr[fi + startR] += 2 * alpha * (alpha * R - max);
		else {
			//val = alpha * M_PI * sin(2 * alpha * M_PI * R);
			double rounded_R = round(alpha * R) / (double)alpha;
			grad.host_arr[fi + startR] += 2 * (R - rounded_R);
		}
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
