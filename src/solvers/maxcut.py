from time import time

from .base import Solver, CALLBACK_NAME
from problems.maxcut import MaxcutProblem
from .callbacks.base import BaseLazyCallback
from .callbacks.separators.separator_name import SEPARATOR_NAME


class MaxcutSolver(Solver):
    def __init__(self, problem: MaxcutProblem, cut_type: str, **kwargs) -> None:
        super().__init__(problem, **kwargs)
        self.separator = SEPARATOR_NAME[cut_type](self.edge2idx, origin_graph=self.graph)

        lazy_constraint = self.register_callback(BaseLazyCallback)
        lazy_constraint.set_attribute(self.separator)

    def create_mip_formulation(self):
        self.objective.set_sense(self.objective.sense.maximize)

        for edge in self.graph.edges:
            var_name = "x.{}.{}".format(*edge)
            self.edge2idx[edge] = self.variables.add(
                obj=[self.graph.edges[edge]["weight"]],
                lb=[0.0],
                ub=[1.0],
                types=["B"],
                names=[var_name],
            )[0]

    def basic_solve(self, *args, **kwargs):
        if "user_callback" in kwargs:
            self.user_callback = self.register_callback(
                CALLBACK_NAME[kwargs["user_callback"]]
            )
            self.user_callback.set_attribute(self.separator, **kwargs["user_cb_kwargs"])

        s = time()
        self.solve()
        t = time() - s
        print("Time to solve model", t)
        print("The objective value is", self.solution.get_objective_value())

        return t, self.solution.get_objective_value()
