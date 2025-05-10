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
    omp_set_num_threads(4);

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
        int pivot = k;
        double max_val = fabs(A[k * n + k]);

        #pragma omp parallel
        {
            int local_pivot = pivot;
            double local_max = max_val;

            #pragma omp for
            for (int i = k + 1; i < n; ++i) {
                double val = fabs(A[i * n + k]);
                if (val > local_max) {
                    local_max = val;
                    local_pivot = i;
                }
            }

            #pragma omp critical
            {
                if (local_max > max_val) {
                    max_val = local_max;
                    pivot = local_pivot;
                }
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
}