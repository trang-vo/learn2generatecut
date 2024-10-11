import sys
sys.path.append(r"/opt/ibm/ILOG/CPLEX_Studio2211/cplex/python/3.8/x86-64_linux/")

import cplex
from cplex.callbacks import UserCutCallback, BranchCallback, LazyConstraintCallback, NodeCallback
