from TensorGraph.Operation import add, matmul
from TensorGraph.Variable import Variable
from TensorGraph.Placeholder import Placeholder
from TensorGraph.Session import Session

if __name__ == '__main__':
    # Create variables
    A = Variable([[1, 0], [0, -1]])
    b = Variable([1, 1])

    # Create placeholder
    x = Placeholder()

    # Create hidden node y
    y = matmul(A, x)

    # Create output node z
    z = add(y, b)

    session = Session()
    output = session.run(z, {
        x: [1, 2]
    })
    print(output)
