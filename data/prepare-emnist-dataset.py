#
#  Created by Adrian Phoulady on 12/31/19.
#  (c) 2019 Adrian Phoulady.
#

import numpy as np
import emnist


def savexy(x, y, threshold, filename):
    xy = np.concatenate((x.reshape(-1, 28 * 28) > int(threshold * 255), y.reshape(-1, 1)), 1)
    np.savetxt(filename, xy, '%d')


t = .3
x_train, y_train = emnist.extract_training_samples('byclass')
x_test, y_test = emnist.extract_test_samples('byclass')
print(x_train.shape)
savexy(x_train, y_train, t, 'emnist-train.data')
savexy(x_test, y_test, t, 'emnist-test.data')
