import tensorgraph as tg

if __name__ == '__main__':
    tg.Graph().as_default()

    # Create variables
    A = tg.Variable([[1, 0], [0, -1]])
    b = tg.Variable([1, 1])

    # Create placeholder
    x = tg.placeholder()

    # Create hidden node y
    y = tg.matmul(A, x)

    # Create output node z
    z = tg.add(y, b)

    session = tg.Session()
    output = session.run(z, {
        x: [1, 2]
    })
    print(output)
