import sys

from problems.base import Problem
from .callbacks.callback_name import CALLBACK_NAME
from .cplex_api import cplex


class Solver(cplex.Cplex):
    def __init__(self, problem: Problem, **kwargs) -> None:
        super().__init__()
        self.parameters.preprocessing.presolve.set(
            self.parameters.preprocessing.presolve.values.off
        )
        self.parameters.threads.set(1)
        self.parameters.mip.strategy.search.set(
            self.parameters.mip.strategy.search.values.traditional
        )
        self.parameters.mip.interval.set(1)
        self.parameters.mip.limits.cutsfactor.set(0)
        self.parameters.mip.display.set(4)

        self.results_logger = sys.stdout
        self.warning_logger = sys.stdout
        self.error_logger = sys.stdout
        self.logger = sys.stdout

        if "display_log" in kwargs:
            if not kwargs["display_log"]:
                logfile = None
                if "log_path" in kwargs:
                    if kwargs["log_path"] != "":
                        logfile = open(kwargs["log_path"], "w")
                    else:
                        logfile = None
                self.results_logger = self.set_results_stream(logfile)
                self.warning_logger = self.set_warning_stream(logfile)
                self.error_logger = self.set_error_stream(logfile)
                self.logger = self.set_log_stream(logfile)

        if "time_limit" in kwargs:
            self.parameters.timelimit.set(kwargs["time_limit"])

        self.edge2idx = {}
        self.graph = problem.graph
        self.problem = problem
        self.create_mip_formulation()
        self.user_callback = None

    def create_mip_formulation(self):
        raise NotImplementedError

    def basic_solve(self, *args, **kwargs):
        raise NotImplementedError
