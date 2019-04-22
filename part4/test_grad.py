from TensorGraph.Session import *
from TensorGraph.train import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = Placeholder()
    c = Placeholder()

    # Create red points centered at (-2, -2)
    np.random.seed(0)
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

    # Create blue points centered at (2, 2)
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

    # Initialize weights randomly
    W = Variable(np.random.randn(2, 2))
    b = Variable(np.random.randn(2))

    # Build perceptron
    p = softmax(add(matmul(X, W), b))

    # Build cross-entropy loss
    J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

    # Build minimization op
    minimization_op = GradientDescentOptimizer(learning_rate=0.01).minimize(J)

    # Build placeholder inputs
    feed_dict = {
        X: np.concatenate((blue_points, red_points)),
        c:
            [[1, 0]] * len(blue_points)
            + [[0, 1]] * len(red_points)

    }

    # Create session
    session = Session()

    # Perform 100 gradient descent steps
    for step in range(100):
        J_value = session.run(J, feed_dict)
        if step % 10 == 0:
            print("Step:", step, " Loss:", J_value)
        session.run(minimization_op, feed_dict)

    # Print final result
    W_value = session.run(W)
    print("Weight matrix:\n", W_value)
    b_value = session.run(b)
    print("Bias:\n", b_value)

    # W_value = np.array([[1.27496197 - 1.77251219], [1.11820232 - 2.01586474]])
    # b_value = np.array([-0.45274057 - 0.39071841])

    # Plot a line y = -x
    x_axis = np.linspace(-4, 4, 100)
    y_axis = -W_value[0][0] / W_value[1][0] * x_axis - b_value[0] / W_value[1][0]
    plt.plot(x_axis, y_axis)

    # Add the red and blue points
    plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
    plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')
    plt.show()
