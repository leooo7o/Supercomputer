#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <omp.h>
using namespace std;
#define DL -2.
#define DD -2.
#define DR 2.
#define DU 2.
#define M 40
#define N 40
#define delta 1e-6
vector<vector <double>> w(M + 1, vector<double>(N + 1, 0));
const double h_1 = (DR - DL) / M;
const double h_2 = (DU - DD) / N;
const double eps = max(h_1, h_2) * max(h_1, h_2);
double isinarea(double x, double y) {
	if (y * y <= x && x <= 1) {
		return 1;
	}
	else {
		return 0;
	}
}
void nodetype(vector<vector<int>>& nodeType) {
#pragma omp parallel for collapse(2)
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (isinarea(DL + (j + 0.5) * h_1, DD + (i + 0.5) * h_2) > 0) {
				nodeType[i][j] = 1;
			}
			else
			{
				nodeType[i][j] = 0;
			}
		}
	}
}
void cal_F(vector<vector<double>>& Fij, const vector<vector<int>>& nodeType) {
#pragma omp parallel for collapse(2)
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			double L = DL + (j - 0.5) * h_1;
			double R = DL + (j + 0.5) * h_1;
			double U = DD + (i + 0.5) * h_2;
			double D = DD + (i - 0.5) * h_2;
			int ij = nodeType[i][j], im1j = nodeType[i - 1][j], ijm1 = nodeType[i][j - 1], im1jm1 = nodeType[i - 1][j - 1];
			if (R <= 0 || L >= 1 || U <= -1 || D >= 1 || D * D >= R || U * U >= R) {
				Fij[i][j] = 0;
			}
			else if (R <= 1)
			{
				if (im1j == 0 && ij == 0 && im1jm1 == 0 && ijm1 == 0)
				{
					Fij[i][j] = 0;
				}
				if (im1j == 1 && ij == 0 && im1jm1 == 0 && ijm1 == 0)
				{
					Fij[i][j] = (R - U * U) * (U - sqrt(R)) / (h_1 * h_2);
				}
				if (im1j == 0 && ij == 1 && im1jm1 == 0 && ijm1 == 0)
				{
					Fij[i][j] = (R - D * D) * (sqrt(R) - D) / (h_1 * h_2);
				}
				if (im1j == 1 && ij == 1 && im1jm1 == 0 && ijm1 == 0)
				{
					Fij[i][j] = R / h_1;
				}
				if (im1j == 1 && ij == 1 && im1jm1 == 1 && ijm1 == 0)
				{
					Fij[i][j] = 0;
				}
				if (im1j == 1 && ij == 1 && im1jm1 == 0 && ijm1 == 1)
				{
					Fij[i][j] = 1;
				}
				if (im1j == 1 && ij == 1 && im1jm1 == 1 && ijm1 == 1)
				{
					Fij[i][j] = 1;
				}
			}
			else if (R > 1)
			{
				(1 - L) / h_1;
			}
		}
	}
}
double cal_uv(const vector<vector<double> >& u, const vector<vector<double> >& v, double h_1, double h_2) {
	double res = 0;
#pragma omp parallel for reduction(+:res) collapse(2)
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res = res + h_1 * h_2 * u[i][j] * v[i][j];
		}
	}
	return res;
}
double cal_norm(const vector<vector<double>>& u, double h_1, double h_2) {
	return sqrt(cal_uv(u, u, h_1, h_2));
}
vector<vector<double>> cal_difference(const vector<vector<double>>& u, const vector<vector <double>>& v) {
	vector<vector<double>> res(M + 1, vector<double>(N + 1, 0));
#pragma omp parallel for collapse(2)
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i][j] = u[i][j] - v[i][j];
		}
	}
	return res;
}
vector<vector<double>> multiconst(const vector<vector<double>>& u, double c) {
	vector<vector<double>> res(M + 1, vector<double>(N + 1, 0));
#pragma omp parallel for collapse(2)
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i][j] = c * u[i][j];
		}
	}
	return res;
}
double cal_A(int i, int j, const vector<vector<int>>& nodeType) {
	double x = DL + (j - 0.5) * h_1;
	double yU = DD + (i + 0.5) * h_2;
	double yD = DD + (i - 0.5) * h_2;
	int type = nodeType[i - 1][j - 1] + nodeType[i][j - 1];
	if (type == 0) {
		if (x < 0 || x > 1 || yU < -1 || yD > 1 || (abs(yU) > abs(yD) && yD * yD > x) || (abs(yU) < abs(yD) && yU * yU > x)) {
			return 1 / eps;
		}
		else {
			return (2 * sqrt(x)) / h_2 + (1 - (2 * sqrt(x)) / h_2) / eps;
		}
	}
	else if (type == 1) {
		if (yD * yD > x) {
			return (yU + sqrt(x)) / h_2 + (1 - (yU + sqrt(x)) / h_2) / eps;
		}
		else {
			return (sqrt(x) - yD) / h_2 + (1 - (sqrt(x) - yD) / h_2) / eps;
		}
	}
	else {
		return 1;
	}
}
double cal_B(int i, int j, const vector<vector<int>>& nodeType) {

	double y = DD + (i - 0.5) * h_2;
	double xR = DL + (j + 0.5) * h_1;
	double xL = DL + (j - 0.5) * h_1;
	int type = nodeType[i - 1][j] + nodeType[i - 1][j - 1];
	if (type == 0) {
		if (y < -1 || y > 1 || xL > 1 || xR < 0 || xR < y * y) {
			return 1 / eps;
		}
		else {
			return (1 - y * y) / h_1 + (1 - (1 - y * y) / h_1) / eps;
		}
	}
	else if (type == 1) {
		if (xR > 1) {
			return (1 - xL) / h_1 + (1 - (1 - xL) / h_1) / eps;
		}
		else {
			return (xR - y * y) / h_1 + (1 - (xR - y * y) / h_1) / eps;
		}
	}
	else {
		return 1;
	}
}
vector<vector<double>> operatorA(const vector<vector<double>>& w, const vector<vector<int>>& nodeType) {
	vector<vector<double>> res(M + 1, vector<double>(N + 1, 0));
#pragma omp parallel for collapse(2)
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			double term1 = (cal_A(i, j + 1, nodeType) * (w[i][j + 1] - w[i][j]) - cal_A(i, j, nodeType) * (w[i][j] - w[i][j - 1])) / (h_1 * h_1);
			double term2 = (cal_B(i + 1, j, nodeType) * (w[i + 1][j] - w[i][j]) - cal_B(i, j, nodeType) * (w[i][j] - w[i - 1][j])) / (h_2 * h_2);
			res[i][j] = -(term1 + term2);
		}
	}
	return res;
}
int main() {
	vector<vector<int>> nodeType(M, vector<int>(N, 0));
	nodetype(nodeType);
	vector<vector<double>> Fij(M + 1, vector<double>(N + 1, 0.0)), wn = w, residual, Ar;
	cal_F(Fij, nodeType);
	double tau;
	int itimes = 0;
	auto start = chrono::steady_clock::now();
	while (1) {
		w = wn;
		residual = cal_difference(operatorA(w, nodeType), Fij);
		Ar = operatorA(residual, nodeType);
		tau = cal_uv(residual, residual, h_1, h_2) / cal_uv(Ar, residual, h_1, h_2);
		wn = cal_difference(w, multiconst(residual, tau));
		itimes++;
		if (cal_norm(cal_difference(wn, w), h_1, h_2) < delta)
		{
			break;
		}
	}
	auto end = chrono::steady_clock::now();
	cout << "Iteration Times: " << itimes << endl << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;
	ofstream FILE("result " + to_string(M) + " " + to_string(N) + ".csv");
	for (int i = M; i >= 0; i--) {
		for (int j = 0; j < N + 1; j++) {
			if (j == N) {
				FILE << wn[M - i][j] << "\n";
			}
			else {
				FILE << wn[M - i][j] << ",";
			}
		}
	}
	FILE.close();
	return 0;
}
