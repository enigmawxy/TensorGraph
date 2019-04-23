import numpy as np
import tensorgraph as tg
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tg.Graph().as_default()

    X = tg.placeholder()
    c = tg.placeholder()

    # Create red points centered at (-2, -2)
    np.random.seed(0)
    red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

    # Create blue points centered at (2, 2)
    blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

    # Initialize weights randomly
    W = tg.Variable(np.random.randn(2, 2))
    b = tg.Variable(np.random.randn(2))

    # Build perception
    p = tg.softmax(tg.add(tg.matmul(X, W), b))

    # Build cross-entropy loss
    J = tg.negative(tg.reduce_sum(tg.reduce_sum(tg.multiply(c, tg.log(p)), axis=1)))

    # Build minimization op
    minimization_op = tg.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

    # Build placeholder inputs
    feed_dict = {
        X: np.concatenate((blue_points, red_points)),
        c:
            [[1, 0]] * len(blue_points)
            + [[0, 1]] * len(red_points)

    }

    # Create session
    session = tg.Session()

    # 设置动态绘制，不是训练完了才绘制结果
    plt.ion()

    # Perform 100 gradient descent steps
    for step in range(100):
        J_value = session.run(J, feed_dict)
        if step % 10 == 0:
            print("Step:", step, " Loss:", J_value)

            # 动态绘制结果图，你可以看到训练过程如何慢慢的拟合数据点
            ax = plt.gca()
            ax.set_title('Classifier')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
            plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')

            W_value = session.run(W)
            b_value = session.run(b)
            x_axis = np.linspace(-4, 4, 100)
            y_axis = -W_value[0][0] / W_value[1][0] * x_axis - b_value[0] / W_value[1][0]
            plt.plot(x_axis, y_axis, 'r')
            plt.pause(0.1)

        session.run(minimization_op, feed_dict)

    # 关闭动态绘制模式
    plt.ioff()
    plt.show()
