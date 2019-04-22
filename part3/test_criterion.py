from TensorGraph.Operation import add, matmul, softmax, negative, reduce_sum, multiply, log
from TensorGraph.Variable import Variable
from TensorGraph.Placeholder import Placeholder
from TensorGraph.Session import Session
import numpy as np

if __name__ == '__main__':
    # Create red points centered at (-2, -2)
    np.random.seed(0)
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

    # Create blue points centered at (2, 2)
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

    X = Placeholder()
    c = Placeholder()

    W = Variable([
        [1, -1],
        [1, -1]
    ])
    b = Variable([0, 0])

    p = softmax(add(matmul(X, W), b))

    # Cross-entropy loss
    J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

    session = Session()
    print(session.run(J, {
        X: np.concatenate((blue_points, red_points)),
        c:
            [[1, 0]] * len(blue_points)
            + [[0, 1]] * len(red_points)

    }))
