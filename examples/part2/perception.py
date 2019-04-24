import tensorgraph as tg
import numpy as np

if __name__ == '__main__':
    # Create red points centered at (-2, -2)
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

    # Create blue points centered at (2, 2)
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

    tg.Graph().as_default()

    # x = Placeholder()
    # w = Variable([1, 1])
    # b = Variable(0)
    # p = sigmoid(add(matmul(w, x), b))
    #
    # session = Session()
    # print(session.run(p, {x: [3, 2]}))

    X = tg.placeholder(name='X')

    # Create a weight matrix for 2 output classes:
    # One with a weight vector (1, 1) for blue and one with a weight vector (-1, -1) for red
    W = tg.Variable([[1, -1], [1, -1]], 'W')
    b = tg.Variable([0, 0], 'b')
    p = tg.softmax(tg.add(tg.matmul(X, W, 'matmul'), b, 'add'), 'softmax')

    # Create a session and run the perceptron on our blue/red points
    session = tg.Session()
    output_probabilities = session.run(p, {X: np.concatenate((blue_points, red_points))})

    # Print the first 10 lines, corresponding to the probabilities of the first 10 points
    print(output_probabilities[:10])

