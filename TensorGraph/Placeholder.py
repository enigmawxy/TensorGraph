

class Placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
        self.consumers = []

        # Append this placeholder to the list of placeholders in the currently active default graph
        DEFAULT_GRAPH.placeholders.append(self)
