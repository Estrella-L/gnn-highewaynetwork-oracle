from filtering import Filtering
from preprocess import SampleSubgraph


class Coarsen:
    def __init__(self, query_graph, data_graph):
        self.query_graph = query_graph
        self.data_graph = data_graph
        self.filter_model = Filtering(self.query_graph, self.data_graph)
        self.sampler = SampleSubgraph(self.query_graph, self.data_graph)
    
    def build_coarse_data_graph():
        candidates, candidate_count = self.filter_model.GQL_filter()
        if 0 in candidate_count:
            return None
        starting_vertex = candidate_count.index(min(candidate_count))
        new_vertices, new_v_label, new_degree, new_edges, new_e_label, new_v_neigh = self.sampler.find_subgraph(starting_vertex, candidates)

