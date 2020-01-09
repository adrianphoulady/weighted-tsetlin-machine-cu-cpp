//
//  Created by Adrian Phoulady on 12/19/19.
//  © 2019 Adrian Phoulady
//

#include <iostream>
#include "weightm.cuh"

class multiweightm {

    int const classes;
    int const literals;
    weightm * const machine;

public:

    // constructor
    multiweightm(int classes, int features, int clauses, double p, double gamma, int threshold, int state_bits = 8)
    : classes{classes},
    literals{(2 * features - 1) / word_bits + 1},
    machine{(weightm *) new char[classes * sizeof(weightm)]} {
        while (classes--)
            new (machine + classes) weightm(features, clauses, p, gamma, threshold, state_bits);
    };

    ~multiweightm() {
        delete[] (char *) machine;
    }

    // training for a single input
    void train(word const *x, int y) {
        int zero = fastrandrange(classes - 1), one = y;
        zero += zero >= one;
        machine[zero].train(x, 0);
        machine[one].train(x, 1);
    };

    // fitting on the input dataset
    void fit(word const *x, int const *y, int samples, int epochs, bool mix = true) {
        int *idx = new int[samples];
        for (int i = 0; i < samples; ++i)
            idx[i] = i;
        while (epochs--) {
            if (mix)
                shuffle(idx, samples);
            for (int i = 0; i < samples; ++i)
                train(x + idx[i] * literals, y[idx[i]]);
        }
    };

    // predicting the class of a single input
    int predict(word const *input) {
        int mxi = 0;
        double mxv = machine[0].infer(input), v;
        for (int m = 1; m < classes; ++m)
            if (mxv < (v = machine[m].infer(input))) {
                mxv = v;
                mxi = m;
            }
        return mxi;
    };

    // evaluating the machine on a dataset
    double evaluate(word const *x, int const *y, int samples) {
        int correct = 0;
        for (int i = 0; i < samples; ++i)
            correct += predict(x + i * literals) == y[i];
        return (double) correct / samples;
    };

};
