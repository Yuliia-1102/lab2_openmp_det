#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include "/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/aarch64-apple-darwin24/14/include/omp.h"
#include <chrono>
using namespace std;

int main() {
    omp_set_num_threads(8);

    int n = 2520;
    cout << "Розмір матриці: " << n << "." << endl;

    vector<double> A(n * n);
    mt19937 gen(11);
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    double det = 1.0;
    int total_swaps = 0;

    auto start = chrono::high_resolution_clock::now();

    for (int k = 0; k < n; ++k) {
        double max_val = fabs(A[k * n + k]);
        int pivot = k;

        int nthreads = omp_get_max_threads();
        vector<double> t_max(nthreads, max_val);
        vector<int> t_piv(nthreads, pivot);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double local_max = max_val;
            int local_pivot = pivot;

            #pragma omp for
            for (int i = k + 1; i < n; ++i) {
                double val = fabs(A[i * n + k]);
                if (val > local_max) {
                    local_max = val;
                    local_pivot = i;
                }
            }

            t_max[tid] = local_max;
            t_piv[tid] = local_pivot;
        }

        max_val = t_max[0];
        pivot = t_piv[0];
        for (int t = 1; t < nthreads; ++t) {
            if (t_max[t] > max_val) {
                max_val = t_max[t];
                pivot = t_piv[t];
            }
        }

        if (fabs(max_val) < 1e-12) {
            cout << "Матриця сингулярна. Визначник = 0.0." << endl;
            return 0;
        }

        if (pivot != k) {
            for (int j = 0; j < n; ++j)
                swap(A[k * n + j], A[pivot * n + j]);
            total_swaps++;
        }

        #pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
        }
    }

    for (int i = 0; i < n; ++i)
        det *= A[i * n + i];

    int sign = (total_swaps % 2 == 0) ? 1 : -1;
    det *= sign;

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Визначник матриці з OpenMP-алгоритму: " << setprecision(10) << det << '.' << endl;
    cout << "Час виконання OpenMP-алгоритму: " << elapsed.count() << " секунд." << endl;
    cout << '\n';

    return 0;
}