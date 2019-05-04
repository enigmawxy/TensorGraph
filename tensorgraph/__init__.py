#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorgraph.graph import *
from tensorgraph.operations import *
from tensorgraph.gradients import RegisterGradient
from tensorgraph.session import Session
import tensorgraph.train
from tensorgraph.nn import *

# Create a default graph.
# import builtins
# DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
