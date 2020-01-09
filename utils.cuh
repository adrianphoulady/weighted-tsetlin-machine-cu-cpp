//
//  Created by Adrian Phoulady on 12/20/19.
//  © 2019 Adrian Phoulady
//

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include "multiweightm.cuh"

#define literals ((2 * features - 1) / word_bits + 1)

// get the next option from command line arguments
char getopt(int argc, char * const argv[], char const *optstr, char *&optarg) {
    static int optind = 1;
    if (optind >= argc || *argv[optind] != '-')
        return -1;
    auto opt = *(argv[optind++] + 1);
    auto p = strchr(optstr, opt);
    if (!p)
        return '?';
    if (p[1] == ':') {
        if (optind >= argc)
            return ':';
        optarg = argv[optind++];
    }
    return opt;
}

// read hyper-parameters from command line arguments
void update(int argc, char * const argv[], std::string &experiment, int &clauses, double &p, int &threshold, double &gamma, int &epochs, bool &shuffle, int &verbose) {
    int opt;
    static char *optarg = nullptr;
    while ((opt = getopt(argc, argv, "x:c:p:t:g:e:n:s:v:h", optarg)) != -1)
        switch (opt) {
            case 'h':
                printf("-x experiment\n-c clauses\n-p p\n-t threshold\n-g gamma\n-e epochs\n-n new rand\n-s shuffle\n-v verbose\n");
                break;
            case 'x':
                experiment = optarg;
                break;
            case 'c':
                clauses = atoi(optarg);
                break;
            case 'p':
                p = atof(optarg);
                break;
            case 't':
                threshold = atoi(optarg);
                break;
            case 'g':
                gamma = atof(optarg);
                break;
            case 'e':
                epochs = atoi(optarg);
                break;
            case 'n':
                fastsrand(strcmp(optarg, "0")? atoi(optarg): time(nullptr));
                break;
            case 's':
                shuffle = strcmp(optarg, "0") && strcmp(optarg, "false");
                break;
            case 'v':
                verbose = atoi(optarg);
        }
}

// load the file into arrays x and y, and get the number of features
int load_file(std::string const &fname, word *&x, int *&y, int &samples) {
    std::ifstream fin(fname);
    if (!fin) {
        printf("File %s is missing!\n", fname.c_str());
        exit(1);
    }
    std::vector<std::vector<int>> data;
    for (std::string line; std::getline(fin, line); ) {
        std::istringstream iss{line};
        data.emplace_back(std::vector<int>{});
        std::copy(std::istream_iterator<int>(iss), std::istream_iterator<int>(), std::back_inserter(data.back()));
    }
    fin.close();

    int features = data[0].size() - 1;
    samples = data.size();
    x = new word[samples * literals];
    y = new int[samples];
    std::fill(x, x + samples * literals, 0);
    for (int s = 0; s < samples; ++s) {
        if (data[s].size() != features + 1u) {
            printf("Inconsistent sample at line %d of %s!", s + 1, fname.c_str());
            exit(2);
        }
        for (int f = 0; f < features; ++f) {
            int l = f + !data[s][f] * features;
            x[s * literals + l / word_bits] |= (word) 1 << l % word_bits;
        }
        y[s] = data[s].back();
    }

    return features;
}

// determine features and classes, and load test and train data
void load_data(std::string const &experiment, int &features, int &classes, word *&x_train, int *&y_train, int &trains, word *&x_test, int *&y_test, int &tests) {
    load_file("data/" + experiment + "-train.data", x_train, y_train, trains);
    features = load_file("data/" + experiment + "-test.data", x_test, y_test, tests);
    classes = std::max(*std::max_element(y_train, y_train + trains), *std::max_element(y_test, y_test + tests)) + 1;
}

// sample train data for faster evaluations in training, assuming size < x.rows
void sample_data(word const *x, int const *y, int size, word *&xs, int *&ys, int ssize, int features) {
    int *idx = new int[size];
    for (int i = 0; i < size; ++i)
        idx[i] = i;
    xs = new word[ssize * literals];
    ys = new int[ssize];
    for (int i = 0; i < ssize; ++i) {
        std::swap(idx[i], idx[i + fastrandrange(size - i)]);
        std::copy(x + idx[i] * literals, x + (idx[i] + 1) * literals, xs + i * literals);
        ys[i] = y[idx[i]];
    }
}

// migrate it to device
template<typename T>
void host_to_device(T *&a, int size) {
    T *da = allocate(da, size);
    cudaMemcpy(da, a, size * sizeof(T), cudaMemcpyHostToDevice);
    delete[] a;
    a = da;
}

// fit a machine on the dataset for the given hyper-parameters.
void fit(std::string const &experiment, int clauses, double p, double gamma, int threshold, int epochs, bool shuffle = false, int verbose = 1) {
    clock_t tc0 = clock();

    int features, classes;
    word *x_train, *x_test, *x_tray;
    int *y_train, *y_test, *y_tray;
    int trains, tests, trays;
    load_data(experiment, features, classes, x_train, y_train, trains, x_test, y_test, tests);
    sample_data(x_train, y_train, trains, x_tray, y_tray, trays = tests / 4, features);

    multiweightm wtm{classes, features, clauses, p, gamma, threshold, 8};

    printf("%s - samples=%dK, features=%d, classes=%d - clauses=%d, p=%.4f, gamma=%.5f, threshold=%d - tpb=%d, bpg=%d\n", experiment.c_str(), trains / 1000, features, classes, clauses, p, gamma, threshold, threads_per_block, blocks_per_grid);

    host_to_device(x_train, trains * literals);
    host_to_device(x_test, tests * literals);
    host_to_device(x_tray, trays * literals);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        clock_t c0 = clock();
        wtm.fit(x_train, y_train, trains, 1, shuffle);
        clock_t c1 = clock();
        if (epoch % verbose == 0 || epoch == epochs) {
            double e1 = wtm.evaluate(x_test, y_test, tests);
            clock_t c2 = clock();
            double e2 = wtm.evaluate(x_tray, y_tray, trays);

            printf("epoch %03d of training and testing -", epoch);
            printf(" %04lus and %04lus -", (c1 - c0) / CLOCKS_PER_SEC, (c2 - c1) / CLOCKS_PER_SEC);
            printf(" %6.2f%%  and %6.2f%%\n", 100 * e2, 100 * e1);
        }
        else
            printf("epoch %03d of training - %04lus\n", epoch, (c1 - c0) / CLOCKS_PER_SEC);
    }

    int ss = (clock() - tc0) / CLOCKS_PER_SEC, mm = ss / 60, hh = mm / 60;
    printf("total time: %02d:%02d:%02d\n", hh, mm % 60, ss % 60);
}
