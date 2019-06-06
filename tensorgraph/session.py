from tensorgraph.graph import Operation
from tensorgraph.graph import Placeholder, Variable


class Session:
    """Represents a particular execution of a computational graph.
    """

    @staticmethod
    def run(operation, feed_dict={}):
        """Computes the output of an operation

        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        nodes_post_order = traverse_post_order(operation)

        for node in nodes_post_order:

            if type(node) == Placeholder:
                node.output_value = feed_dict[node]
            elif type(node) == Variable:
                node.output_value = node.value
            else:
                node.inputs = [input_node.output_value for input_node in node.input_nodes]
                node.compute()

        return operation.output_value


def traverse_post_order(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_post_order = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)

        # if node.name is not None:
        #     print(node.name)

        nodes_post_order.append(node)

    recurse(operation)

    return nodes_post_order
