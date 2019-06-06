# coding: utf-8
import numpy as np
from tensorgraph.graph import Operation, Variable


class AND(Operation):
    def __init__(self, x1, x2):
        super(self.__class__, self).__init__(x1, x2)

    def compute(self):
        x1, x2 = self.input_nodes
        if x1.output_value is None:
            a1 = x1.value
        else:
            a1 = x1.output_value
        if x2.output_value is None:
            a2 = x2.value
        else:
            a2 = x2.output_value

        x = np.array([a1, a2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            self.output_value = 0
        else:
            self.output_value = 1

        return self.output_value


class NAND(Operation):
    def __init__(self, x1, x2):
        super(self.__class__, self).__init__(x1, x2)

    def compute(self):
        x1, x2 = self.input_nodes
        if x1.output_value is None:
            a1 = x1.value
        else:
            a1 = x1.output_value
        if x2.output_value is None:
            a2 = x2.value
        else:
            a2 = x2.output_value

        x = np.array([a1, a2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            self.output_value = 0
        else:
            self.output_value = 1

        return self.output_value


class OR(Operation):
    def __init__(self, x1, x2):
        super(self.__class__, self).__init__(x1, x2)

    def compute(self):
        x1, x2 = self.input_nodes
        if x1.output_value is None:
            a1 = x1.value
        else:
            a1 = x1.output_value
        if x2.output_value is None:
            a2 = x2.value
        else:
            a2 = x2.output_value

        x = np.array([a1, a2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            self.output_value = 0
        else:
            self.output_value = 1

        return self.output_value


class XOR(Operation):
    def __init__(self, x1, x2):
        super(self.__class__, self).__init__(x1, x2)

    def compute(self):
        x1, x2 = self.input_nodes
        s1 = NAND(x1, x2).compute()
        s2 = OR(x1, x2).compute()
        y = AND(Variable(s1), Variable(s2))
        self.output_value = y.compute()

        return self.output_value
