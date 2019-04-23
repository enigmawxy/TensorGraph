import tensorgraph as tg

if __name__ == '__main__':
    tg.Graph().as_default()

    # Create variables
    A = tg.Variable([[1, 0], [0, -1]], 'A')
    b = tg.Variable([1, 1], 'b')

    # Create placeholder
    x = tg.placeholder(name='x')

    # Create hidden node y
    y = tg.matmul(A, x, 'y')

    # Create output node z
    z = tg.add(y, b, 'z')

    session = tg.Session()
    output = session.run(z, {
        x: [1, 2]
    })
    print(output)
