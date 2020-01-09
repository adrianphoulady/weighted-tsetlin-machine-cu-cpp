//
//  Created by Adrian Phoulady on 12/20/19.
//  © 2019 Adrian Phoulady
//

#include "utils.cuh"

int main(int argc, char * const argv[]) {
    std::string experiment = "imdb";
    int clauses = 100000, threshold = 130, epochs = 35, verbose = 1;
    double p = .006, gamma = .006;
    bool shuffle = true;

    update(argc, argv, experiment, clauses, p, threshold, gamma, epochs, shuffle, verbose);
    fit(experiment, clauses, p, gamma, threshold, epochs, shuffle, verbose);

    return 0;
}
