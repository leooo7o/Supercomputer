#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <string>
using namespace std;
const int M = 40;
const int N = 40; 
const double xmin = -2.0, xmax = 2.0;
const double ymin = -2.0, ymax = 2.0;
const double epsilon = 0.01;
const double tolerance = 1e-6;
const double hx = (xmax - xmin) / (M - 1);
const double hy = (ymax - ymin) / (N - 1);
void save_to_csv(const std::vector<std::vector<double>>& u) {
    ofstream FILE("result " + to_string(M) + " " + to_string(N) + ".csv");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (j == M - 1) {
                FILE << u[j][N - 1 - i] << "\n";
            }
            else {
                FILE << u[j][N - 1 - i] << ",";
            }
        }
    }
    FILE.close();
}
int inRegionD(double x, double y) {
    return (y * y <= x && x <= 1) ? 1 : 0;
}
double cal_F(int i, int j) {
    double L = xmin + (i - 0.5) * hx,R = xmin + (i + 0.5) * hx,D = ymin + (j - 0.5) * hy,U = ymin + (j + 0.5) * hy;
    int ij = inRegionD(R, U), im1j = inRegionD(R, D), ijm1 = inRegionD(L, U), im1jm1 = inRegionD(L, D);
    if (R <= 0 || L >= 1 || U <= -1 || D >= 1 || D * D >= R || U * U >= R) {
        return 0;
    }
    else if (R <= 1)
    {
        if (im1j == 0 && ij == 0 && im1jm1 == 0 && ijm1 == 0)
        {
            return 0;
        }
        if (im1j == 1 && ij == 0 && im1jm1 == 0 && ijm1 == 0)
        {
            return (R - U * U) * (U - sqrt(R)) / (hx * hy);
        }
        if (im1j == 0 && ij == 1 && im1jm1 == 0 && ijm1 == 0)
        {
            return (R - D * D) * (sqrt(R) - D) / (hx * hy);
        }
        if (im1j == 1 && ij == 1 && im1jm1 == 0 && ijm1 == 0)
        {
            return R / hx;
        }
        if (im1j == 1 && ij == 1 && im1jm1 == 1 && ijm1 == 0)
        {
            return 0;
        }
        if (im1j == 1 && ij == 1 && im1jm1 == 0 && ijm1 == 1)
        {
            return 1;
        }
        if (im1j == 1 && ij == 1 && im1jm1 == 1 && ijm1 == 1)
        {
            return 1;
        }
    }
    else if (R > 1)
    {
        return (1 - L) / hx;
    }
}
double cal_A(int i, int j) {
    double x = xmin + (i - 0.5) * hx,yD = ymin + (j - 0.5) * hy,yU = ymin + (j + 0.5) * hy; // Pij+1
    int type = inRegionD(x, yD) + inRegionD(x, yU);
    if (type == 0) {
        if (x < 0 || x > 1 || yU < -1 || yD > 1 || (abs(yU) > abs(yD) && yD * yD > x) || (abs(yU) < abs(yD) && yU * yU > x)) {
            return 1 / epsilon;
        }
        else {
            return (2 * sqrt(x)) / hy + (1 - (2 * sqrt(x)) / hy) / epsilon;
        }
    }
    else if (type == 1) {
        if (yD * yD > x) {
            return (yU + sqrt(x)) / hy + (1 - (yU + sqrt(x)) / hy) / epsilon;
        }
        else {
            return (sqrt(x) - yD) / hy + (1 - (sqrt(x) - yD) / hy) / epsilon;
        }
    }
    else {
        return 1;
    }
}
double cal_B(int i, int j) {
    double xL = xmin + (i - 0.5) * hx,y = ymin + (j - 0.5) * hy,xR = xmin + (i + 0.5) * hx; // Pij+1
    int type = inRegionD(xR, y) + inRegionD(xL, y);
    if (type == 0) {
        if (y < -1 || y > 1 || xL > 1 || xR < 0 || xR < y * y) {
            return 1 / epsilon;
        }
        else {
            return (1 - y * y) / hx + (1 - (1 - y * y) / hx) / epsilon;
        }
    }
    else if (type == 1) {
        if (xR > 1) {
            return (1 - xL) / hx + (1 - (1 - xL) / hx) / epsilon;
        }
        else {
            return (xR - y * y) / hx + (1 - (xR - y * y) / hx) / epsilon;
        }
    }
    else {
        return 1;
    }
}
void computeUij(vector<vector<double>>& u,vector<vector<double>>& A,vector<vector<double>>& B,vector<vector<double>>& F,int& iterations,int rank,int size,
    int Px, int Py,
    int coords_x, int coords_y,
    int local_M, int local_N,
    int start_i, int start_j,
    MPI_Comm comm2d) {
    double norm_global = 1.0;
    while (norm_global > tolerance) {
        int north, south, east, west;
        MPI_Cart_shift(comm2d, 0, 1, &west, &east);
        MPI_Cart_shift(comm2d, 1, 1, &north, &south); 
        vector<double> send_top(local_N, 0.0);
        vector<double> recv_bottom(local_N, 0.0);
        vector<double> send_bottom(local_N, 0.0);
        vector<double> recv_top(local_N, 0.0);
        vector<double> send_left(local_M, 0.0);
        vector<double> recv_right(local_M, 0.0);
        vector<double> send_right(local_M, 0.0);
        vector<double> recv_left(local_M, 0.0);
        for (int j = 1; j <= local_N; ++j) {
            send_top[j - 1] = u[1][j];
        }
        for (int j = 1; j <= local_N; ++j) {
            send_bottom[j - 1] = u[local_M][j];
        }
        for (int i = 1; i <= local_M; ++i) {
            send_left[i - 1] = u[i][1];
        }
        for (int i = 1; i <= local_M; ++i) {
            send_right[i - 1] = u[i][local_N];
        }
        MPI_Request requests[8];
        int req_count = 0;
        if (north != MPI_PROC_NULL) {
            MPI_Irecv(&recv_top[0], local_N, MPI_DOUBLE, north, 0, comm2d, &requests[req_count++]);
            MPI_Isend(&send_top[0], local_N, MPI_DOUBLE, north, 1, comm2d, &requests[req_count++]);
        }
        if (south != MPI_PROC_NULL) {
            MPI_Irecv(&recv_bottom[0], local_N, MPI_DOUBLE, south, 1, comm2d, &requests[req_count++]);
            MPI_Isend(&send_bottom[0], local_N, MPI_DOUBLE, south, 0, comm2d, &requests[req_count++]);
        }
        if (west != MPI_PROC_NULL) {
            MPI_Irecv(&recv_left[0], local_M, MPI_DOUBLE, west, 2, comm2d, &requests[req_count++]);
            MPI_Isend(&send_left[0], local_M, MPI_DOUBLE, west, 3, comm2d, &requests[req_count++]);
        }
        if (east != MPI_PROC_NULL) {
            MPI_Irecv(&recv_right[0], local_M, MPI_DOUBLE, east, 3, comm2d, &requests[req_count++]);
            MPI_Isend(&send_right[0], local_M, MPI_DOUBLE, east, 2, comm2d, &requests[req_count++]);
        }
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        if (north != MPI_PROC_NULL) {
            for (int j = 1; j <= local_N; ++j) {
                u[0][j] = recv_top[j - 1];
            }
        }
        if (south != MPI_PROC_NULL) {
            for (int j = 1; j <= local_N; ++j) {
                u[local_M + 1][j] = recv_bottom[j - 1];
            }
        }
        if (west != MPI_PROC_NULL) {
            for (int i = 1; i <= local_M; ++i) {
                u[i][0] = recv_left[i - 1];
            }
        }
        if (east != MPI_PROC_NULL) {
            for (int i = 1; i <= local_M; ++i) {
                u[i][local_N + 1] = recv_right[i - 1];
            }
        }
        double local_R = 0.0, local_AR = 0.0;
        double alpha;
        vector<vector<double>> r(local_M + 2, vector<double>(local_N + 2, 0.0));
        vector<vector<double>> Ar(local_M + 2, vector<double>(local_N + 2, 0.0));
        for (int i = 1; i <= local_M; ++i) {
            for (int j = 1; j <= local_N; ++j) {
                r[i][j] = -F[i][j]
                    - (1.0 / (hx * hx)) * (A[i + 1][j] * (u[i + 1][j] - u[i][j])
                        - A[i][j] * (u[i][j] - u[i - 1][j]))
                    - (1.0 / (hy * hy)) * (B[i][j + 1] * (u[i][j + 1] - u[i][j])
                        - B[i][j] * (u[i][j] - u[i][j - 1]));
            }
        }
        for (int i = 1; i <= local_M; ++i) {
            for (int j = 1; j <= local_N; ++j) {
                Ar[i][j] = -(1.0 / (hx * hx)) * (A[i + 1][j] * (r[i + 1][j] - r[i][j])
                    - A[i][j] * (r[i][j] - r[i - 1][j]))
                    - (1.0 / (hy * hy)) * (B[i][j + 1] * (r[i][j + 1] - r[i][j])
                        - B[i][j] * (r[i][j] - r[i][j - 1]));
            }
        }
        for (int i = 1; i <= local_M; ++i) {
            for (int j = 1; j <= local_N; ++j) {
                local_R += r[i][j] * r[i][j];
                local_AR += Ar[i][j] * r[i][j];
            }
        }
        double R, AR;
        MPI_Allreduce(&local_R, &R, 1, MPI_DOUBLE, MPI_SUM, comm2d);
        MPI_Allreduce(&local_AR, &AR, 1, MPI_DOUBLE, MPI_SUM, comm2d);
        alpha = R / AR;
        double local_norm = 0.0;
        for (int i = 1; i <= local_M; ++i) {
            for (int j = 1; j <= local_N; ++j) {
                u[i][j] = u[i][j] - alpha * r[i][j];
                local_norm += r[i][j] * r[i][j];
            }
        }
        double norm_sq;
        MPI_Allreduce(&local_norm, &norm_sq, 1, MPI_DOUBLE, MPI_SUM, comm2d);
        norm_global = sqrt(norm_sq);
        if (norm_global < tolerance) {
            break;
        }
        iterations++;
    }
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int Px = 1, Py = 1;
    double min_ratio_diff = 1e9;
    for (int i = 1; i <= sqrt(size); i++) {
        if (size % i == 0) {
            int j = size / i;
            double ratio = double(i) / double(j);
            if (ratio < 0.5) ratio = 1.0 / ratio;
            if (ratio >= 1.0 && ratio <= 2.0) {
                if (abs(ratio - 1.0) < min_ratio_diff) {
                    min_ratio_diff = abs(ratio - 1.0);
                    Px = i;
                    Py = j;
                }
            }
        }
    }
    if (Px * Py != size) {
        Py = 2;
    }
    int dims[2] = { Px, Py };
    int periods[2] = { 0, 0 };
    MPI_Comm comm2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2d);
    int coords[2];
    MPI_Cart_coords(comm2d, rank, 2, coords);
    int coords_x = coords[0];
    int coords_y = coords[1];
    int local_M = M / Px;
    int remainder_M = M % Px;
    int start_i = coords_x * local_M + min(coords_x, remainder_M);
    local_M += (coords_x < remainder_M) ? 1 : 0;
    int local_N = N / Py;
    int remainder_N = N % Py;
    int start_j = coords_y * local_N + min(coords_y, remainder_N);
    local_N += (coords_y < remainder_N) ? 1 : 0;
    vector<vector<double>> u(local_M + 2, vector<double>(local_N + 2, 0.0));
    vector<vector<double>> A(local_M + 2, vector<double>(local_N + 2, 0.0));
    vector<vector<double>> B(local_M + 2, vector<double>(local_N + 2, 0.0));
    vector<vector<double>> F(local_M + 2, vector<double>(local_N + 2, 0.0));
    for (int i = 1; i <= local_M; ++i) {
        for (int j = 1; j <= local_N; ++j) {
            int global_i = start_i + i;
            int global_j = start_j + j;
            A[i][j] = cal_A(global_i, global_j);
            B[i][j] = cal_B(global_i, global_j);
            F[i][j] = cal_F(global_i, global_j);
        }
    }
    int iter = 0;
    MPI_Barrier(comm2d);
    auto start0 = chrono::high_resolution_clock::now();
    computeUij(u, A, B, F, iter, rank, size, Px, Py, coords_x, coords_y, local_M, local_N, start_i, start_j, comm2d);
    MPI_Barrier(comm2d);
    auto end0 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end0 - start0;
    if (rank == 0) {
        cout << "M = " << M << " and N = " << N << endl<<"time: " << duration.count() << " ms" << endl<<"iterations: " << iter << endl;
    }
    vector<vector<double>> u_local(local_M, vector<double>(local_N, 0.0));
    for (int i = 0; i < local_M; i++) {
        for (int j = 0; j < local_N; j++) {
            u_local[i][j] = u[i + 1][j + 1];
        }
    }
    int* recvcounts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
    }
    int local_size = local_M * local_N;
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm2d);
    if (rank == 0) {
        displs = new int[size];
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    vector<double> u_flat(local_M * local_N, 0.0);
    for (int i = 0; i < local_M; i++) {
        for (int j = 0; j < local_N; j++) {
            u_flat[i * local_N + j] = u_local[i][j];
        }
    }
    vector<double> u_total;
    if (rank == 0) {
        int total_size = 0;
        for (int i = 0; i < size; i++) total_size += recvcounts[i];
        u_total.resize(total_size, 0.0);
    }
    MPI_Gatherv(u_flat.data(), local_size, MPI_DOUBLE,
        u_total.data(), recvcounts, displs, MPI_DOUBLE,
        0, comm2d);
    if (rank == 0) {
        vector<vector<double>> u_global(M, vector<double>(N, 0.0));
        int idx = 0;
        for (int p = 0; p < size; p++) {
            int proc_coords[2];
            MPI_Cart_coords(comm2d, p, 2, proc_coords);
            int proc_coords_x = proc_coords[0];
            int proc_coords_y = proc_coords[1];
            int proc_local_M = M / Px;
            int proc_remainder_M = M % Px;
            int proc_start_i = proc_coords_x * proc_local_M + min(proc_coords_x, proc_remainder_M);
            proc_local_M += (proc_coords_x < proc_remainder_M) ? 1 : 0;
            int proc_local_N = N / Py;
            int proc_remainder_N = N % Py;
            int proc_start_j = proc_coords_y * proc_local_N + min(proc_coords_y, proc_remainder_N);
            proc_local_N += (proc_coords_y < proc_remainder_N) ? 1 : 0;
            for (int i = 0; i < proc_local_M; i++) {
                for (int j = 0; j < proc_local_N; j++) {
                    int global_i = proc_start_i + i;
                    int global_j = proc_start_j + j;
                    u_global[global_i][global_j] = u_total[idx++];
                }
            }
        }
        save_to_csv(u_global);
    }
    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
    }
    MPI_Finalize();
    return 0;
}
