//
//  Created by Adrian Phoulady on 12/19/19.
//  © 2019 Adrian Phoulady
//

#include <cstdint>
#include <cmath>

// omitting inline for not requiring C++17
/*inline*/ static uint64_t mcg_state = 0xcafef00dd15ea5e5u;
/*inline*/ static uint32_t constexpr fastrand_max = UINT32_MAX;
/*inline*/ static double constexpr fastrand_max_inverse = 1. / fastrand_max;
/*inline*/ static double constexpr two_pi = 2 * 3.14159265358979323846;

// fast pcg32
// https://en.wikipedia.org/wiki/Permuted_congruential_generator
__host__ __device__ inline static uint32_t fastrand(uint64_t &state = mcg_state) {
    auto x = state;
    state *= 6364136223846793005u;
    return (x ^ x >> 22) >> (22 + (x >> 61)); // 22 = 32 - 3 - 7, 61 = 64 - 3
}

// random from 0. to 1.
__device__ inline static double fastrandom(uint64_t &state) {
    return fastrand_max_inverse * fastrand(state);
}

// a biased one suffices here
__host__ __device__ inline static uint32_t fastrandrange(uint32_t n, uint64_t &state = mcg_state) {
    return (uint64_t) n * fastrand(state) >> 32;
}

// fastrand's seed
__device__ inline static void fastsrand(uint64_t &state) {
    state = state << 1 | 1;
    fastrand(state);
}

// fastrand's seed
inline static void fastsrand(uint64_t &&state) {
    mcg_state = state << 1 | 1;
    fastrand();
}

// Box–Muller transform
__device__ inline static double normal(double mean, double variance, uint64_t &state) {
    // it won't be an issue if the first random number is .0
    return mean + sqrt(-2 * log(fastrandom(state)) * variance) * sin(two_pi * fastrandom(state));
}

// approximating binomial with normal
__device__ inline static int binomial(double p, int n, uint64_t &state) {
    double b = normal(n * p, n * p * (1 - p), state) + .5; // "+ .5" for rounding to the nearest integer
    return b <= 0? 0: b >= n? n: (int) b;
}

// Fisher-Yates random shuffle
inline static void shuffle(int *a, int n) {
    for (int i = n; i; --i)
        std::swap(a[i - 1], a[fastrandrange(i)]);
}
