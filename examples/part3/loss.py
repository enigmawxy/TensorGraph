import tensorgraph as tg
import numpy as np

if __name__ == '__main__':
    # Create red points centered at (-2, -2)
    np.random.seed(0)
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

    # Create blue points centered at (2, 2)
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

    tg.Graph().as_default()

    X = tg.placeholder()
    c = tg.placeholder()

    W = tg.Variable([
        [1, -1],
        [1, -1]
    ])
    b = tg.Variable([0, 0])

    p = tg.softmax(tg.add(tg.matmul(X, W), b))

    # Cross-entropy loss
    J = tg.negative(tg.reduce_sum(tg.reduce_sum(tg.multiply(c, tg.log(p)), axis=1)))

    session = tg.Session()
    print(session.run(J, {
        X: np.concatenate((blue_points, red_points)),
        c:
            [[1, 0]] * len(blue_points)
            + [[0, 1]] * len(red_points)

    }))
