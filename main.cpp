// Copyright (c) 2020 University of Lyon and CNRS (France).
// All rights reserved.
//
//The code is the implementation of our paper
//"PCQM: A Full-Reference Quality Metric for Colored 3D Point Clouds"
//Gabriel Meynet, Yana Nehme, Julie Digne, Guillaume Lavoué
//Presented at QoMEX 2020.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

/** @file */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "PointSet.h"
#include "nanoflann.hpp"
#include <Eigen/Dense>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include "utilities.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ios;

using namespace nanoflann;
using namespace Eigen;

using Eigen::JacobiSVD;
using Eigen::Matrix3d;

std::string global_regfile;
std::string global_reffile;

/**
* \fn double interpolate1_computevalue(double x0, double x1, double y0, double y1, double x)
* \brief Find the value of x in the new range
*/
inline double interpolate1_computevalue(double x0, double x1, double y0, double y1, double x) {
	return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
}

/**
* \fn double interpolate1_process(const std::vector<double>& init_grid, const std::vector<double>& val_grid, double x)
* \brief Interpolate
*/
double interpolate1_process(const std::vector<double>& init_grid, const std::vector<double>& val_grid, double x) 
{
	auto upper = std::upper_bound(init_grid.begin(), init_grid.end(), x);
	int position = std::distance(init_grid.begin(), upper);

	if (position < init_grid.size()) {
		double res = interpolate1_computevalue(init_grid[position - 1], init_grid[position], val_grid[position - 1],
			val_grid[position], x);
		return res;
	}
	else 
	{
		return val_grid[position - 1];
	}
}

/**
* \fn double interpolate2_computevalue(...)
* \brief Find the value of x in the new range
*/
inline double interpolate2_computevalue(double q_00, double q_10, double q_01, double q_11, double x0, double x1, double y0,
	double y1, double x, double y) {
	double x1x0, y1y0, x1x, y1y, yy0, xx0;

	x1x0 = std::abs(x1 - x0);
	y1y0 = std::abs(y1 - y0);

	x1x = x1 - x;
	y1y = y1 - y;
	yy0 = y - y0;
	xx0 = x - x0;

	return 1.0 / (x1x0 * y1y0) * (q_00 * x1x * y1y + q_10 * xx0 * y1y + q_01 * x1x * yy0 + q_11 * xx0 * yy0);
}

/**
* \fn std::pair<double, double> interpolate2_process(...)
* \brief Given a grid, compute the interpolation of the point defined by x and y
*/
std::pair<double, double> interpolate2_process(const std::vector<std::vector<std::pair<double, double>>>& init_grid_AB,
	double x, double y)
{
	int x_min = std::floor(x);
	int x_max = x_min + 1;
	int y_min = std::floor(y);
	int y_max = y_min + 1;

	if (x_min == x_max && y_min == y_max) {
		return std::make_pair(init_grid_AB[x_min + 128][y_min + 128].first, init_grid_AB[x_min + 128][y_min + 128].second);
	}

	double a_interpolated = interpolate2_computevalue(
		init_grid_AB[x_min + 128][y_min + 128].first, init_grid_AB[x_max + 128][y_min + 128].first,
		init_grid_AB[x_min + 128][y_max + 128].first, init_grid_AB[x_max + 128][y_max + 128].first, x_min, x_max, y_min,
		y_max, x, y);
	double b_interpolated = interpolate2_computevalue(
		init_grid_AB[x_min + 128][y_min + 128].second, init_grid_AB[x_max + 128][y_min + 128].second,
		init_grid_AB[x_min + 128][y_max + 128].second, init_grid_AB[x_max + 128][y_max + 128].second, x_min, x_max, y_min,
		y_max, x, y);

	return std::make_pair(a_interpolated, b_interpolated);
}

/**
* \brief Initialize the 1D and 2D grid for interpolation from file
*/
void initMatLABCH(
	std::string weight_prefix,
	std::vector<double>& init_grid_L, std::vector<double>& grid_L)
{
	int size_L = 100001;
	init_grid_L.assign(size_L, 0);
	grid_L.assign(size_L, 0);

	std::ifstream data_f;
	data_f.open(weight_prefix + std::string("L_data.txt"), std::ifstream::in);

	if (!data_f.fail()) {
		std::string line1;
		int cpt = 0;
		while (getline(data_f, line1) && cpt < size_L) {
			grid_L[cpt] = std::stod(line1);
			cpt++;
		}
	}
	else
		std::cout << "Unable to open L_data.txt \t";
	data_f.close();
	for (int i = 0; i < init_grid_L.size(); i++) {
		init_grid_L[i] = i * 0.001;
	}
}

/**
* \brief Compute the projection of an origin point onto the polynomial approximation of a set of neighbors given by a list of indices.
*/
void computeProjectionAndCurvature(const Point& origin, const std::vector<Point>& refpoints,
	const std::vector<size_t>& indices, Point& proj, double& H, 
    MatrixXd& A, VectorXd& B, 
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& eig,
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd>& qr) {

	Matrix3d M;
	M.setZero();
	Vector3d mu;
	mu.setZero();
	int nneighbors = indices.size();

	for (int i = 0; i < nneighbors; ++i) {
		Point p = refpoints[indices[i]];
		Vector3d neighbor(p.x, p.y, p.z);
		mu = mu + neighbor;
		M = M + neighbor * neighbor.transpose();
	}

	mu = mu / ((double)nneighbors);
	M = 1. / ((double)nneighbors) * M - mu * mu.transpose();

	// get local frame (reuses dynamically allocated buffers internally)
	eig.compute(M);

	Eigen::Vector3d t1 = eig.eigenvectors().col(2);
	Eigen::Vector3d t2 = eig.eigenvectors().col(1);
	Eigen::Vector3d n = eig.eigenvectors().col(0);

    A.resize(nneighbors, 6);
    B.resize(nneighbors);

	// build linear system
	for (int i = 0; i < nneighbors; ++i) {
		double xglob = refpoints[indices[i]].x - origin.x;
		double yglob = refpoints[indices[i]].y - origin.y;
		double zglob = refpoints[indices[i]].z - origin.z;
		Vector3d v(xglob, yglob, zglob);
		double x = v.transpose() * t1;
		double y = v.transpose() * t2;
		double z = v.transpose() * n;

		A(i, 0) = x * x;
		A(i, 1) = y * y;
		A(i, 2) = x * y;
		A(i, 3) = x;
		A(i, 4) = y;
		A(i, 5) = 1;

		B(i) = z;
	}

    // Solve using pre-allocated workspace
    qr.compute(A);
	VectorXd coeffs = qr.solve(B);

	// corresponding point:
	Vector3d delta = coeffs(5) * n;
	proj = origin + delta;

	// corresponding curvature
	double fxx = 2 * coeffs(0);
	double fyy = 2 * coeffs(1);
	double fxy = coeffs(2);
	double fx = coeffs(3);
	double fy = coeffs(4);

	H = 0.5 * ((1 + fx * fx) * fyy + (1 + fy * fy) * fxx - 2 * fxy * fx * fy) / pow(1 + fx * fx + fy * fy, 1.5);
}

/**
* \brief Compute the Euclidean distance between two points.
*/
inline double compute_distance(const Point& a, const Point& b) {
	return std::sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
}

inline double F(double input) 
{
	if (input > 0.008856)
		return std::cbrt(input); 
	else
		return ((double(841.0) / 108.0) * input + double(4.0) / 29.0); 
}

/**
* \brief Compute the transformation from RGB colorspace to XYZ colorspace
*/
inline void RGBtoXYZ(int _R, int _G, int _B, double& X, double& Y, double& Z) {
	double R = _R / 255.0;
	double G = _G / 255.0;
	double B = _B / 255.0;

	R = ((R > 0.0404482362771076) ? std::pow((R + 0.055) / 1.055, 2.4) : (R / 12.92));
	G = ((G > 0.0404482362771076) ? std::pow((G + 0.055) / 1.055, 2.4) : (G / 12.92));
	B = ((B > 0.0404482362771076) ? std::pow((B + 0.055) / 1.055, 2.4) : (B / 12.92));

	X = 0.412396 * R + 0.357583 * G + 0.180493 * B;
	Y = 0.212586 * R + 0.71517 * G + 0.0722005 * B;
	Z = 0.0192972 * R + 0.119184 * G + 0.950497 * B;
}

/**
* \brief Compute the transformation from XYZ colorspace to Lab colorspace
*/
inline void XYZtoLab(double Y, double& L) {
	const double Yo = 1.000000;
	L = 116.0 * F(Y / Yo) - 16.0; 
}

/**
* \brief Compute the transformation from RGB colorspace to Lab colorspace
*/
inline void RGBtoLab(double R, double G, double B, double& L) {
	double X, Y, Z;
	RGBtoXYZ(R, G, B, X, Y, Z);

	const double Yo = 1.000000;
	L = 116.0 * F(Y / Yo) - 16.0; 
}

/**
* \brief Compute the transformation from Lab colorspace to LCH colorspace
*/
inline void LabtoLCH(double _L, double _a, double _b, double& L, double& C, double& H) {
	L = _L;
	C = std::sqrt(_a * _a + _b * _b);
	double temp_h = std::atan2(_b, _a); 

	if (temp_h >= 0) {
		H = temp_h;
	}
	else {
		H = temp_h + 360.0;
	}
}

/**
* \brief Compute the transformation from RGB colorspace to LCH colorspace
*/
inline void RGBtoLCH(double R, double G, double B, double& L, double& C, double& H) {
	double _L, _a, _b;
	RGBtoLab(R, G, B, _L);
	LabtoLCH(_L, _a, _b, L, C, H);
}

/**
* \brief Remove the extension of an input filename
*/
std::string remove_extension(const std::string& filename) {
	size_t lastdot = filename.find_last_of(".");
	if (lastdot == std::string::npos) return filename;
	return filename.substr(0, lastdot);
}

/**
* \brief Write the mean of each feature and PCQM to CSV
*/
bool write_mean_features(const string regfile, const string reffile, double PCQM, const string destination) {
	int length;
	ifstream filestr;
	filestr.open(destination, ios::binary);
	filestr.seekg(0, ios::end); 
	length = filestr.tellg(); 
	filestr.close(); 

	std::ofstream PCQM_out_f(destination, std::ios::app);

	if (PCQM_out_f.is_open()) {
		if (length == -1) 
		{
			PCQM_out_f << "reffile" << ";";
			PCQM_out_f << "regfile" << ";";
			PCQM_out_f << "PCQM";
		}
		
		PCQM_out_f << "\n";
		PCQM_out_f << regfile << ";";
		PCQM_out_f << reffile << ";";
		PCQM_out_f << PCQM << ";";
		PCQM_out_f.close();

		return true;
	}
	else
		cout << "Unable to open " << destination << " for writting";
	PCQM_out_f.close();
	return false;
}

/**
* \brief Compute local feature depending on the case_number
*/
double compute_color_feature(int index_array, const size_t nMatches_Reg, const std::vector<double>& data_proj,
	const std::vector<double>& data_me, const std::vector<double>& ret_weight_Reg,
	const std::vector<double>& ret_weight_Ref,
	const std::vector<std::pair<size_t, double>>& ret_matches_Reg, double sum_distances_me,
	double sum_distances_proj, int case_number, double constant_value) {

	double mu_me_gaussian = 0.0;
	double mu_proj_gaussian = 0.0;
	double mu_me_unweighted = 0.0;
	double mu_proj_unweighted = 0.0;
	double nb_elem = 0.0;

	for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
		size_t index_reg = ret_matches_Reg[cpt_neigh].first;

		mu_me_gaussian += data_me[index_reg] * ret_weight_Reg[cpt_neigh];
		mu_proj_gaussian += data_proj[index_reg] * ret_weight_Ref[cpt_neigh];

		mu_me_unweighted += data_me[index_reg];
		mu_proj_unweighted += data_proj[index_reg];

		nb_elem += 1.0;
	}

	mu_me_gaussian /= sum_distances_me;
	mu_proj_gaussian /= sum_distances_proj;

	mu_me_unweighted /= nb_elem;
	mu_proj_unweighted /= nb_elem;

	switch (case_number) {
	case 1: { // LCH
		return 1.0 / (constant_value * (mu_me_gaussian - mu_proj_gaussian) * (mu_me_gaussian - mu_proj_gaussian) + 1.0);
	}

	case 2: { // Contrast IDF2
		double standard_dev_me = 0.0;
		double standard_dev_proj = 0.0;

		for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
			size_t index_reg = ret_matches_Reg[cpt_neigh].first;

			standard_dev_me += pow(data_me[index_reg] - mu_me_gaussian, 2.0) * ret_weight_Reg[cpt_neigh];
			standard_dev_proj += pow(data_proj[index_reg] - mu_proj_gaussian, 2.0) * ret_weight_Ref[cpt_neigh];
		}

		if (standard_dev_me < 0) standard_dev_me = 0.0;
		if (standard_dev_proj < 0) standard_dev_proj = 0.0;

		standard_dev_me /= sum_distances_me;
		standard_dev_proj /= sum_distances_proj;

		return (2.0 * std::sqrt(standard_dev_me) * std::sqrt(standard_dev_proj) + constant_value) /
			(standard_dev_me + standard_dev_proj + constant_value);
	}

	case 3: { // Structure IDF3
		double standard_dev_me = 0.0;
		double standard_dev_proj = 0.0;
		double covariance = 0.0;

		for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
			size_t index_reg = ret_matches_Reg[cpt_neigh].first;
			standard_dev_me += pow(data_me[index_reg] - mu_me_unweighted, 2.0);
			standard_dev_proj += pow(data_proj[index_reg] - mu_proj_unweighted, 2.0);
			covariance += ((data_me[index_reg] - mu_me_unweighted) * (data_proj[index_reg] - mu_proj_unweighted));
		}
		standard_dev_me /= nb_elem;
		standard_dev_proj /= nb_elem;

		if (standard_dev_me < 0) standard_dev_me = 0.0;
		if (standard_dev_proj < 0) standard_dev_proj = 0.0;

		covariance /= nb_elem;

		return (covariance + constant_value) / (std::sqrt(standard_dev_me) * std::sqrt(standard_dev_proj) + constant_value);
	}
	}
	return 0;
}

void compute_geometric_feature(int index_array, const size_t nMatches_Reg, const std::vector<double>& data_proj,
	const std::vector<double>& data_me, const std::vector<double>& ret_weight_Reg,
	const std::vector<double>& ret_weight_Ref,  std::vector<double>& structure_field,
	const std::vector<std::pair<size_t, double>>& ret_matches_Reg, double sum_distances_me,
	double sum_distances_proj, double constant_value) {

	double mu_me_gaussian = 0.0;
	double mu_proj_gaussian = 0.0;
	double mu_me_unweighted = 0.0;
	double mu_proj_unweighted = 0.0;
	double nb_elem = 0.0;

	for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
		size_t index_reg = ret_matches_Reg[cpt_neigh].first;

		mu_me_gaussian += data_me[index_reg] * ret_weight_Reg[cpt_neigh];
		mu_proj_gaussian += data_proj[index_reg] * ret_weight_Ref[cpt_neigh];

		mu_me_unweighted += data_me[index_reg];
		mu_proj_unweighted += data_proj[index_reg];

		nb_elem += 1.0;
	}

	mu_me_gaussian /= sum_distances_me;
	mu_proj_gaussian /= sum_distances_proj;

	mu_me_unweighted /= nb_elem;
	mu_proj_unweighted /= nb_elem;

	double structure = 0.0;
	double variance_me = 0.0;
	double variance_proj = 0.0;
	double covariance = 0.0;

	for (unsigned int cpt_neigh = 0; cpt_neigh < nMatches_Reg; cpt_neigh++) {
		size_t index_reg = ret_matches_Reg[cpt_neigh].first;

		variance_me += pow((std::abs(data_me[index_reg]) - mu_me_gaussian), 2.0) * ret_weight_Reg[cpt_neigh];
		variance_proj += pow((std::abs(data_proj[index_reg]) - mu_proj_gaussian), 2.0) * ret_weight_Ref[cpt_neigh];
		covariance +=
			((std::abs(data_me[index_reg]) - mu_me_gaussian) * (std::abs(data_proj[index_reg]) - mu_proj_gaussian)) *
			ret_weight_Reg[cpt_neigh];
	}
	
	double standard_dev_me = std::sqrt(variance_me / sum_distances_me);
	double standard_dev_proj = std::sqrt(variance_proj / sum_distances_proj);
	covariance = covariance / sum_distances_me;

	structure = std::abs(standard_dev_me * standard_dev_proj - covariance) / ((standard_dev_me * standard_dev_proj) + constant_value);
	if (structure > 1.0) structure = 1.0; 

	structure_field[index_array] = structure;
}

void compute_statistics(double radius, const double maxDim, PointSet& regptset, KdTree& m_kdtree2,
	const std::vector<Point>& projectedpointsOnRef,
	const std::vector<double>& meancurvaturesProj, const std::vector<double>& meancurvaturesMe,
	std::vector<double>& geom_structure_field, double& PCQM, const double radius_factor,
	const std::vector<double>& tab_lstar_me,
	const std::vector<double>& tab_lstar_proj,
	std::vector<double>& color_lightness_field,
	std::vector<double>& color_structure_field,
	int threshold_knnsearch, std::string outfile)
{
	std::cout << "Computing PCQM" << std::endl;
	double f3 = 0.0;
	double f4 = 0.0;
	double f6 = 0.0;

#pragma omp parallel 
    {
        nanoflann::SearchParams params;
        params.sorted = false;

        std::vector<std::pair<size_t, double>> ret_matches_Reg;
        std::vector<double> ret_weight_Reg;
        std::vector<double> ret_weight_Ref;

        //FEATURES COMPUTATION
#pragma omp for reduction(+:f3, f4, f6)
        for (int i = 0; i < regptset.npts(); i++) {

            double search_radius_neighborhood = static_cast<double>(radius * radius_factor);
            Point origin = regptset.pts[i];
            
            ret_weight_Reg.clear();
            ret_weight_Ref.clear();
            
            double sum_distances_me = 0.0;
            double sum_distances_proj = 0.0;
            double query_pt[3] = { origin.x, origin.y, origin.z };

            const size_t nMatches_Reg =
                m_kdtree2.radiusSearch(&query_pt[0], search_radius_neighborhood * search_radius_neighborhood, ret_matches_Reg, params);

            double debug_variance = search_radius_neighborhood / 2.0;

            for (size_t cpt_reg = 0; cpt_reg < nMatches_Reg; cpt_reg++) {
                double d_reg = std::sqrt(ret_matches_Reg[cpt_reg].second);
                Point p_orig_proj = projectedpointsOnRef[i];
                Point p_neigh_proj = projectedpointsOnRef[ret_matches_Reg[cpt_reg].first];
                double d_ref = compute_distance(p_orig_proj, p_neigh_proj);

                double wi1 =
                    1.0 / debug_variance / sqrt(2 * 3.141592) *
                    exp(-(d_reg * d_reg) / 2.0 / debug_variance / debug_variance);
                double wi2 =
                    1.0 / debug_variance / sqrt(2 * 3.141592) *
                    exp(-(d_ref * d_ref) / 2.0 / debug_variance / debug_variance);

                ret_weight_Reg.push_back(wi1);
                ret_weight_Ref.push_back(wi2);

                sum_distances_me += wi1;
                sum_distances_proj += wi2;
            }

            double constant_curvature = 1.0;
            double constant_1 = 0.002;
            double constant_3 = 0.1;

            compute_geometric_feature(i, nMatches_Reg, meancurvaturesProj, meancurvaturesMe,
                ret_weight_Reg, ret_weight_Ref,
                geom_structure_field,
                ret_matches_Reg, sum_distances_me, sum_distances_proj, constant_curvature);

            color_lightness_field[i] = compute_color_feature(i, nMatches_Reg, tab_lstar_proj, tab_lstar_me, ret_weight_Reg, ret_weight_Ref,
                ret_matches_Reg, sum_distances_me, sum_distances_proj, 1, constant_1);

            color_structure_field[i] =
                compute_color_feature(i, nMatches_Reg, tab_lstar_proj, tab_lstar_me, ret_weight_Reg, ret_weight_Ref,
                    ret_matches_Reg, sum_distances_me, sum_distances_proj, 3, constant_3);

            f3 += geom_structure_field[i];
            f4 += color_lightness_field[i];
            f6 += color_structure_field[i];
        }
    }

	double size_tab = (double)color_lightness_field.size();

	//PCQM Formula
	PCQM = f3/ size_tab * 0.0057 + (1.0 - f4/ size_tab) * 0.9771 + (1.0 - f6/ size_tab) * 0.0172;

	//Write features and PCQM to CSV
	write_mean_features( global_regfile, global_reffile, PCQM, outfile);
}

/**
* \fn main(int argc, char** argv)
* \brief Entry point of the program
* \return EXIT_SUCCESS if the code executed successfuly.
*/
int main(int argc, char** argv) {
	bool fast_quit = false;

	double RadiusCurvature = 0.004;
	int threshold_knnsearch = 20;
	double radius_factor = 2.0;

	if (argc < 5) {
		std::cerr << "Usage: " << "REFERENCE.ply DISTORTED.ply PCQM_OUT PCQM_WEIGHTS (Options) (--fastquit || -fq) (-r radius)(-rx factor) (-knn nb_point)" << std::endl;
		return -1;
	}

	std::string reffile = argv[2];
	std::string regfile = argv[1];
	std::string outfile = argv[3];
	std::string weight_prefix = argv[4];

	for (int i = 5; i < argc; ++i) {
		if (std::string(argv[i]) == "--fastquit" || std::string(argv[i]) == "-fq") {
			fast_quit = true;
			std::cout << "fast_quit set to : " << fast_quit << std::endl;
		}

		if (std::string(argv[i]) == "-r") {
			if (i + 1 < argc) { 
				i++;              
				RadiusCurvature = std::stod(std::string(argv[i]));
				std::cout << "Radius set to : " << RadiusCurvature << std::endl;
			}
			else {
				std::cerr << "-r option requires one argument." << std::endl;
				return -1;
			}
		}

		if (std::string(argv[i]) == "-knn") {
			if (i + 1 < argc) { 
				i++;
				threshold_knnsearch = std::stoi(std::string(argv[i])); 
				std::cout << "threshold_knnsearch set to : " << threshold_knnsearch << std::endl;
			}
			else {
				std::cerr << "-knn option requires one argument." << std::endl;
				return -1;
			}
		}

		if (std::string(argv[i]) == "-rx") {
			if (i + 1 < argc) { 
				i++;
				radius_factor = std::stod(std::string(argv[i])); 
				std::cout << "radius_factor set to : " << radius_factor << std::endl;
			}
			else {
				std::cerr << "-rx option requires one argument." << std::endl;
				return -1;
			}
		}
	}

	global_regfile = remove_extension(regfile);
	global_reffile = remove_extension(reffile);

	std::cout << "Input reference point set file:  " << reffile << std::endl;
	std::cout << "Input registered point set file:  " << regfile << std::endl;
	std::cout << "Output file:  " << outfile << std::endl;

	PointSet refptset;
	PointSet regptset;

	refptset.read_ply_file(reffile, true);
	regptset.read_ply_file(regfile, true);

	KdTree m_kdtree(3, refptset, KDTreeSingleIndexAdaptorParams(10));
	m_kdtree.buildIndex();
	KdTree m_kdtree2(3, regptset, KDTreeSingleIndexAdaptorParams(10));
	m_kdtree2.buildIndex();

	std::vector<double> init_grid_L;
	std::vector<double> grid_L;

	initMatLABCH(weight_prefix, init_grid_L, grid_L);

	std::vector<Point> projectedpointsOnRef;
	projectedpointsOnRef.assign(regptset.npts(), Point());

	std::vector<double> meancurvaturesProj;
	std::vector<double> meancurvaturesMe;
	std::vector<double> geom_structure_field;

	meancurvaturesProj.assign(regptset.npts(), 0.0);
	meancurvaturesMe.assign(regptset.npts(), 0.0);
	geom_structure_field.assign(regptset.npts(), 0.0);

	std::vector<double> color_lightness_field, color_structure_field;
	color_lightness_field.assign(regptset.npts(), 0.0);
	color_structure_field.assign(regptset.npts(), 0.0);

	std::vector<double> tab_lstar_me, tab_lstar_proj;
	tab_lstar_me.assign(regptset.npts(), 0.0);
	tab_lstar_proj.assign(regptset.npts(), 0.0);

	double rangeX = std::abs(regptset.xmin) + std::abs(regptset.xmax);
	double rangeY = std::abs(regptset.ymin) + std::abs(regptset.ymax);
	double rangeZ = std::abs(regptset.zmin) + std::abs(regptset.zmax);

	double maxDim = std::max(rangeX, std::max(rangeY, rangeZ));
	double radius = RadiusCurvature * maxDim;
	
	std::cout << "Start curvature computation and color projection" << std::endl;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel
    {
        std::vector<size_t> ret_index_ref_init(threshold_knnsearch);
        std::vector<size_t> ret_index_reg_init(threshold_knnsearch);
        std::vector<double> out_dist_sqr_ref(threshold_knnsearch);
        std::vector<double> out_dist_sqr_reg(threshold_knnsearch);
        
        std::vector<std::pair<size_t, double>> ret_matches_Ref;
        std::vector<std::pair<size_t, double>> ret_matches_Me;
        std::vector<size_t> ret_index_ref;
        std::vector<size_t> ret_index_reg;
        
        nanoflann::SearchParams params;
        
        MatrixXd A;
        VectorXd B;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;

#pragma omp for
        for (int i = 0; i < regptset.npts(); ++i) {
            Point origin = regptset.pts[i];

            double H = 0;
            double K = 0;
            Point proj;

            double query_pt[3] = { origin.x, origin.y, origin.z };

            m_kdtree.knnSearch(&query_pt[0], threshold_knnsearch, &ret_index_ref_init[0], &out_dist_sqr_ref[0]);
            m_kdtree2.knnSearch(&query_pt[0], threshold_knnsearch, &ret_index_reg_init[0], &out_dist_sqr_reg[0]);
            
            params.sorted = true; 

            size_t nMatches_Ref = m_kdtree.radiusSearch(&query_pt[0], radius * radius, ret_matches_Ref, params);
            size_t nMatches_Reg = m_kdtree2.radiusSearch(&query_pt[0], radius * radius, ret_matches_Me, params);

            ret_index_ref.resize(nMatches_Ref);
            ret_index_reg.resize(nMatches_Reg);

            for (size_t cpt_ref = 0; cpt_ref < nMatches_Ref; cpt_ref++) {
                ret_index_ref[cpt_ref] = ret_matches_Ref[cpt_ref].first;
            }

            for (size_t cpt_reg = 0; cpt_reg < nMatches_Reg; cpt_reg++) {
                ret_index_reg[cpt_reg] = ret_matches_Me[cpt_reg].first;
            }

            double dummy_double;
            computeProjectionAndCurvature(origin, refptset.pts, ret_index_ref_init, proj, dummy_double, A, B, eig, qr);

            Point dummy_point;
            computeProjectionAndCurvature(origin, refptset.pts, ret_index_ref, dummy_point, H, A, B, eig, qr);
            computeProjectionAndCurvature(origin, regptset.pts, ret_index_reg, dummy_point, K, A, B, eig, qr);

            Point closest_point_from_origin_ref = refptset.pts[ret_index_ref_init[0]];

            double distance_ori_proj = compute_distance(origin, proj);
            double distance_ori_closest = compute_distance(origin, closest_point_from_origin_ref);

            meancurvaturesProj[i] = std::abs(H) * maxDim;
            meancurvaturesMe[i] = std::abs(K) * maxDim;
            projectedpointsOnRef[i] = (distance_ori_proj <= distance_ori_closest ? proj : closest_point_from_origin_ref);

            double L_star = 0.0;
            RGBtoLab(regptset.pts[i].r, regptset.pts[i].g, regptset.pts[i].b, L_star);
            L_star = interpolate1_process(init_grid_L, grid_L, L_star);
            tab_lstar_me[i] = L_star;
            RGBtoLab(refptset.pts[ret_index_ref_init[0]].r, refptset.pts[ret_index_ref_init[0]].g,
                refptset.pts[ret_index_ref_init[0]].b, L_star); 
            L_star = interpolate1_process(init_grid_L, grid_L, L_star);
            tab_lstar_proj[i] = L_star;
        }
    }

	std::chrono::steady_clock::time_point end_projection = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_projection - begin).count() / 1000000.0 << " sec elpased" << std::endl;

	std::cout << "Start compute_statistics" << std::endl;

	double PCQM = 0.0;

	compute_statistics(radius, maxDim, regptset, m_kdtree2, projectedpointsOnRef,
		meancurvaturesProj, meancurvaturesMe,  geom_structure_field, PCQM, radius_factor,
		tab_lstar_me, 
		tab_lstar_proj,
		color_lightness_field,
		color_structure_field, threshold_knnsearch, outfile);

	std::cout << "PCQM value is : " << PCQM << std::endl;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0 << " sec elpased" << std::endl;

	if (!fast_quit) {
		std::cout << "Press enter to exit" << std::endl;
		std::getchar();
	}
	return EXIT_SUCCESS;
}