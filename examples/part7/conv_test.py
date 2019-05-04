import numpy as np
import tensorgraph as tg

if __name__ == '__main__':
    np.random.seed(10)

    input_x = np.random.randn(3, 1, 28, 28)

    # Create a new graph
    tg.Graph().as_default()

    # Create training input placeholder
    X = tg.Placeholder()

    # Build a hidden layer
    W1 = tg.Variable(0.01 * np.random.randn(30, 1, 5, 5))
    b1 = tg.Variable(np.zeros(30))

    out = tg.nn.con2d(X, W1, b1)

    # Build placeholder inputs
    feed_dict = {
        X: input_x
    }

    # Create session
    session = tg.Session()
    res = session.run(out, feed_dict)
    print(res)

