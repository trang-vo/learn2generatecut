from .maxcut import MaxcutSolver
from .tsp import TSPSolver

SOLVER_NAME = {
    "tsp": TSPSolver,
    "maxcut": MaxcutSolver,
}
