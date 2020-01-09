//
//  Created by Adrian Phoulady on 12/19/19.
//  © 2019 Adrian Phoulady
//

#include "fastrand.cuh"

/*inline*/ static int constexpr threads_per_block = 1024;
/*inline*/ static int constexpr blocks_per_grid = 80;

typedef uint64_t word; // the memory block type for bitwise implementation
/*inline*/ static int constexpr word_bits = sizeof(word) << 3;

__global__ static void init(word *state, double *weight, int clauses, int literals, int states, uint64_t offset);
__global__ static void prevalue(word *state, int *clause, bool *active, int clauses, int literals, int states, word actmask);
__global__ static void value(word *state, int *clause, bool *active, double *weight, double *share, int clauses, int literals, int states, word actmask, word const *x, bool training);
__global__ static void postvalue(int *clause, bool *active, double *weight, double *share, int clauses, bool training);
__global__ static void accumulate(double const *ain, int insize, double *aout);
__global__ static void clause_mask(int *clause, int clauses, double diversion, int y);
__global__ static void setter(word *state, int *clause, double *weight, int clauses, int literals, int states, double p, double onepgamma, word const *x);
__global__ static void clearer(word *state, int *clause, double *weight, int clauses, int literals, int states, double onepgammainv, word const *x);

template<typename T>
inline static T *allocate(T * const &dummy, int size) {
    T *data;
    cudaMalloc(&data, size * sizeof(T));
    return data;
}

class weightm {

    int const clauses;     // number of clauses
    double const p;        // setter feedback probability
    double const gamma;    // weight learning rate
    int const threshold;   // weighted sum threshold for learning toward
    int const states;      // number of bits of states
    int const literals;    // number of words containing all literals
    word const actmask;    // mask for zeroing the remaining action bits in literal words
    word * const state;    // state of all clauses in [clause, literals, bit] order
    int * const clause;    // value of the clause and the mask for receiving setter/clearer feedback
    bool * const active;   // if clause has at list one set action bit (clause is not empty)
    double * const weight; // weight associated to each clause
    double * const share;  // clause share contributed to weighted sum

public:

    // constructor
    weightm(int features, int clauses, double p, double gamma, int threshold, int states = 8)
    : clauses{clauses},
    p{p},
    gamma{gamma},
    threshold{threshold},
    states{states},
    literals{(2 * features - 1) / word_bits + 1},
    actmask{~(bool(2 * features % word_bits) * ~word(0) << 2 * features % word_bits)},
    state{allocate(state, clauses * literals * states)},
    clause{allocate(clause, clauses)},
    active{allocate(active, clauses)},
    weight{allocate(weight, clauses)},
    share{allocate(share, clauses)} {
        init<<<blocks_per_grid, threads_per_block>>>(state, weight, clauses, literals, states, fastrand());
        cudaDeviceSynchronize();
    }

    // destructor
    ~weightm() {
        cudaFree(share);
        cudaFree(weight);
        cudaFree(active);
        cudaFree(clause);
        cudaFree(state);
    }

    // the weighted sum of clauses for an input
    double infer(word const *x, bool training = false) {
        prevalue<<<blocks_per_grid, threads_per_block>>>(state, clause, active, clauses, literals, states, actmask);
        cudaDeviceSynchronize();
        value<<<blocks_per_grid, threads_per_block>>>(state, clause, active, weight, share, clauses, literals, states, actmask, x, training);
        cudaDeviceSynchronize();
        postvalue<<<blocks_per_grid, threads_per_block>>>(clause, active, weight, share, clauses, training);
        cudaDeviceSynchronize();

        static double inference, *dinference = allocate(dinference, blocks_per_grid);
        accumulate<<<blocks_per_grid, threads_per_block>>>(share, clauses, dinference); // sum all the elements in block i of all grids to dinference(i)
        cudaDeviceSynchronize();
        accumulate<<<1, threads_per_block>>>(dinference, blocks_per_grid, dinference); // sum the sums of all blocks in a single variable dinference(0)
        cudaDeviceSynchronize();
        cudaMemcpy(&inference, dinference, sizeof(inference), cudaMemcpyDeviceToHost);

        return inference;
    }

    // training the machine for a single input
    void train(word const *x, int y) {
        double const diversion = .5 + (.5 - y) * infer(x, true) / threshold;
        clause_mask<<<blocks_per_grid, threads_per_block>>>(clause, clauses, diversion, y);
        cudaDeviceSynchronize();

        setter<<<blocks_per_grid, threads_per_block>>>(state, clause, weight, clauses, literals, states, p, 1 + gamma, x);
        clearer<<<blocks_per_grid, threads_per_block>>>(state, clause, weight, clauses, literals, states, 1 / (1 + gamma), x);
        cudaDeviceSynchronize();
    }

    // fitting the machine on a dataset for a number of epochs
    void fit(word const *x, int const *y, int samples, int epochs) {
        while (epochs--)
            for (int i = 0; i < samples; ++i)
                train(x + i * literals, y[i]);
    }

    // predicting the 0-1 output for the input
    int predict(word const *x) {
        return infer(x) >= 0;
    }

    // evaluating the accuracy of the machine
    double evaluate(word const *x, int const *y, int samples) {
        int correct = 0;
        for (int i = 0; i < samples; ++i)
            correct += predict(x + i * literals) == y[i];
        return (double) correct / samples;
    }

};

// increase the states of the automata
__device__ inline static void add(word *state_word, int states, word addend) {
    for (int b = 0; addend && b < states; ++b) {
        state_word[b] ^= addend;
        addend &= state_word[b] ^ addend;
    }
    if (addend)
        for (int b = 0; b < states; ++b)
            state_word[b] ^= addend;
}

// decrease the states of the automata
__device__ inline static void subtract(word *state_word, int states, word subtrahend) {
    for (int b = 0; subtrahend && b < states; ++b) {
        state_word[b] ^= subtrahend;
        subtrahend &= ~(state_word[b] ^ subtrahend);
    }
    if (subtrahend)
        for (int b = 0; b < states; ++b)
            state_word[b] ^= subtrahend;
}

__device__ inline static word literal_mask(uint64_t &rs, double p) {
    int flips = binomial(p, word_bits, rs);
    bool const target = flips <= word_bits / 2;
    // if flips are more than half, do it the other way; make 0s in an all-1 sequence
    if (!target)
        flips = word_bits - flips;
    word mask = 0;
    while (flips) {
        auto b = fastrandrange(word_bits, rs);
        // it might be one instruction less if we had done it in two separate loops for target 0 and target 1
        if ((mask >> b & 1) == target)
            continue;
        mask ^= (word) 1 << b;
        --flips;
    }
    return mask;
}

#define index (blockIdx.x * blockDim.x + threadIdx.x)
#define stride (gridDim.x * blockDim.x)

__device__ /*inline*/ static uint64_t rand_state[threads_per_block * blocks_per_grid];

// initialize the states, weights, and random seeds
__global__ static void init(word *state, double *weight, int clauses, int literals, int states, uint64_t offset) {
    // initialize all the states to 2^states - 1
    for (int cl = index; cl < clauses * literals; cl += stride) {
        auto state_word = state + cl * states;
        for (int b = 0; b < states - 1; ++b)
            state_word[b] = ~((word) 0);
        state_word[states - 1] = 0;
    }
    // even clauses are positive and and odds are negative
    for (int c = index; c < clauses; c += stride)
        weight[c] = c & 1? -1: +1;
    // random state initialization
    fastsrand(rand_state[index] = index + offset);
}

// prepare the prerequisites for running value
__global__ static void prevalue(word *state, int *clause, bool *active, int clauses, int literals, int states, word actmask) {
    for (int c = index; c < clauses; c += stride) {
        state[(c + 1) * literals * states - 1] &= actmask;
        clause[c] = 1;
        active[c] = false;
    }
}

// calculate values of clauses for an input
__global__ static void value(word *state, int *clause, bool *active, double *weight, double *share, int clauses, int literals, int states, word actmask, word const *x, bool training) {
    for (int cl = index; cl < clauses * literals; cl += stride) {
        auto c = cl / literals;
        if (!clause[c])
            continue;
        auto l = cl - c * literals; // == cl % literals;
        auto action = state[(cl + 1) * states - 1];
        if ((action & x[l]) != action) {
            clause[c] = 0;
            continue;
        }
        if (action)
            active[c] = true;
    }
}

// evaluate values of shares
__global__ static void postvalue(int *clause, bool *active, double *weight, double *share, int clauses, bool training) {
    for (int c = index; c < clauses; c += stride)
        share[c] = clause[c] && (training || active[c])? weight[c]: 0;
}

// sum the shares in ain to aout
__global__ static void accumulate(double const *ain, int insize, double *aout) {
    __shared__ double block_share[threads_per_block];

    double grid_share = 0;
    for (int i = index; i < insize; i += stride)
        grid_share += ain[i];
    block_share[threadIdx.x] = grid_share;
    __syncthreads();

    for (int accumulator_size = threads_per_block >> 1; accumulator_size; accumulator_size >>= 1) {
        if (threadIdx.x < accumulator_size)
            block_share[threadIdx.x] += block_share[threadIdx.x + accumulator_size];
        __syncthreads();
    }

    if (!threadIdx.x)
        aout[blockIdx.x] = block_share[0];
}

// deciding on sending feedback and also its type to the clause
__global__ static void clause_mask(int *clause, int clauses, double diversion, int y) {
    auto rs = rand_state[index];
    for (int c = index; c < clauses; c += stride)
        clause[c] |= ((fastrandom(rs) < diversion) << 1) | ((y != (c & 1)) << 2); // 0b(setter-feedback-value)
    rand_state[index] = rs;
}

// setter feedback or feedback type I
__global__ static void setter(word *state, int *clause, double *weight, int clauses, int literals, int states, double p, double onepgamma, word const *x) {
    auto rs = rand_state[index];

    for (int cl = index; cl < clauses * literals; cl += stride) {
        auto c = cl / literals, sfv = clause[c];

        if ((sfv & 0b110) != 0b110) // 0b(setter-feedback-value)
            continue;

        auto l = cl - c * literals; // == cl % literals;
        word mask = literal_mask(rs, p);
        auto state_word = state + cl * states;
        if (sfv & 1) {
            add(state_word, states, x[l]);
            subtract(state_word, states, mask & ~x[l]);
            if (!l)
                weight[c] *= onepgamma;
        }
        else
            subtract(state_word, states, mask);
    }

    rand_state[index] = rs;
}

// clearer feedback or feedback type II
__global__ static void clearer(word *state, int *clause, double *weight, int clauses, int literals, int states, double onepgammainv, word const *x) {
    for (int cl = index; cl < clauses * literals; cl += stride) {
        auto c = cl / literals, sfv = clause[c];
        if ((sfv & 0b111) != 0b011) // 0b(setter-feedback-value)
            continue;
        auto l = cl - c * literals; // == cl % literals;
        if (!l)
            weight[c] *= onepgammainv;
        auto state_word = state + cl * states;
        add(state_word, states, ~state_word[states - 1] & ~x[l]);
    }
}
