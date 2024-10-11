class StateRecorder:
    def __init__(self, *args, **kwargs):
        self.nb_cuts = None
        self.gap = None
        self.time = None
        self.obj = None
        self.cutoff = None
        self.depth = 0
        self.solution = None
        self.node_id = 0
        self.id = None
        self.cut_distances = []
        for kw, value in kwargs.items():
            setattr(self, kw, value)


class NodeRecorder:
    def __init__(self, **kwargs):
        self.nb_separation: int = 0
        self.nb_cuts: int = 0
        self.id: int = -1
        self.depth: int = 0
        # Compute the average cut's quality measures
        self.cut_distances = []
        # Compute the cut's quality measures at the last round of cut generation at the node
        self.last_cut_distances = []
        self.obj_improvements = []
        self.nb_cut_rounds = 0
        self.nb_visited = 0
