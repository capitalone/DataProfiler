from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import scipy.stats as st

from . import BaseColumnProfiler, utils
from .profiler_options import TextProfilerOptions


class GraphProfile(object):
    def __init__(self, name, options=None):
        """
        Initialization of Graph Profiler.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Graph Profiler
        :type options: GraphOptions
        """

        self.name = name
        self.sample_size = 0
        self.times = defaultdict(float)

        """
        Properties
        """
        self._attributes = None
        self._num_nodes = None
        self._num_edges = None
        self._categorical_attributes = None
        self._continuous_attributes = None
        self._avg_node_degree = None
        self._global_max_component_size = None
        self._continuous_distribution = None
        self._categorical_distribution = None
        self.metadata = dict()
        self.times = dict()

        self.__calculations = {
            "num_nodes": GraphProfile._get_num_nodes,
            "num_edges": GraphProfile._get_num_edges,
            "categorical_attributes": GraphProfile._get_categorical_attributes,
            "continuous_attributes": GraphProfile._get_continuous_attributes,
            "avg_node_degree": GraphProfile._get_avg_node_degree,
            "global_max_component_size": GraphProfile._get_global_max_component_size,
            "continuous_distribution": GraphProfile._get_continuous_distribution,
            "categorical_distribution": GraphProfile._get_categorical_distribution,
        }

    @property
    def profile(self):
        """
        Returns the profile of the graph
        :return: the profile of the graph in data
        """
        profile = dict(
            num_nodes=self._num_nodes,
            avg_node_degree=self._avg_node_degree,
            global_max_component_size=self._global_max_component_size,
            continuous_distribution=self._continuous_distribution,
            categorical_distribution=self._categorical_distribution,
            times=self.times,
        )
        return profile

    def diff(self, other_profile, options=None):
        """
        Finds the differences for two unstructured text profiles

        :param other_profile: profile to find the difference with
        :type other_profile: GraphProfiler
        :param options: options for diff output
        :type options: dict
        :return: the difference between profiles
        :rtype: dict
        """
        cls = self.__class__
        if not isinstance(other_profile, cls):
            raise TypeError(
                "Unsupported operand type(s) for diff: '{}' "
                "and '{}'".format(cls.__name__, other_profile.__class__.__name__)
            )
        raise NotImplementedError("Function not yet implemented.")

    def report(self, remove_disabled_flag=False):
        """
        Report on profile attribute of the class and pop value from self.profile if key not in self.__calculations
        """
        calcs_dict_keys = self.__calculations.keys()
        profile = self.profile

        if remove_disabled_flag:
            profile_keys = list(profile.keys())
            for profile_key in profile_keys:
                # need to add props
                if profile_key == "<PROP>":
                    if "<PROP>" in calcs_dict_keys:
                        continue
                profile.pop(profile_key)
        return profile

    def _update_helper(self, data, profile):
        """
        Method for updating the graph profile properties with a cleaned
        dataset and the known null parameters of the dataset.

        :param data: networkx graph
        :type data: NetworkX Graph
        :param profile: graph profile dictionary
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def update(self, graph):
        """
        Updates the graph profile.

        :param data: networkx graph
        :type data: NetworkX Graph
        :return: None
        """
        graph_size = graph.size()
        if graph_size == 0:
            return self
        profile = dict(sample_size=graph_size)

        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=graph_size,
            prev_dependent_properties={},
            subset_properties={},
        )

        self._update_helper(graph_size, profile)

        return self

    """
    Update functions to update props with get functions
    """

    def _update_num_nodes(self, graph):
        return self._get_num_nodes(graph)

    def _update_num_edges(self, graph):
        return self._get_num_edges(graph)

    def _update_avg_node_degree(self, graph):
        return self._get_avg_node_degree(graph)

    def _update_global_max_component_size(self, graph):
        return self._get_global_max_component_size(graph)

    def _update_categorical_attributes(self, graph):
        return self._get_categorical_attributes(graph)

    def _update_continuous_attributes(self, graph):
        return self._get_continuous_attributes(graph)

    def _update_continuous_distribution(self, graph):
        return self._get_continuous_distribution(graph)

    def _update_categorical_distribution(self, graph):
        return self._get_categorical_distribution(graph)

    """
    Get functions to calculate props
    """

    @staticmethod
    def _get_num_nodes(graph):
        """
        Compute the number of nodes
        """
        return graph.number_of_nodes()

    @staticmethod
    def _get_num_edges(graph):
        """
        Compute the number of edges
        """
        return graph.number_of_edges()

    @staticmethod
    def _get_categorical_attributes(graph):
        return GraphProfile._get_categorical_and_continuous_attributes(graph)[0]

    @staticmethod
    def _get_continuous_attributes(graph):
        return GraphProfile._get_categorical_and_continuous_attributes(graph)[1]

    @staticmethod
    def _get_avg_node_degree(graph):
        """
        Compute average node degree of nodes in graph
        """
        total_degree = 0
        for node in graph:
            total_degree += graph.degree[node]
        return total_degree / GraphProfile._get_num_nodes(graph)

    @staticmethod
    def _get_global_max_component_size(graph):
        """
        Compute largest subgraph component of the graph
        """
        graph_connected_components = sorted(
            nx.connected_components(graph), key=len, reverse=True
        )
        largest_component = graph.subgraph(graph_connected_components[0])
        return largest_component.size()

    @staticmethod
    def _get_continuous_distribution(self, graph):
        """
        Compute the continuous distribution of graph edge continuous attributes
        """
        attributes = GraphProfile.find_all_attributes(graph)
        continuous_attributes = GraphProfile._get_continuous_attributes(graph)

        distribution_candidates = [st.bernoulli, st.binom, st.geom, st.poisson]
        continuous_distributions = dict()

        for attribute in attributes:
            if attribute in continuous_attributes:
                data_as_list = self._attribute_data_as_list(graph, attribute)

                distribution_candidates = [
                    st.norm,
                    st.uniform,
                    st.expon,
                    st.logistic,
                    st.lognorm,
                    st.gamma,
                ]
                best_mle = ("initial value", 1000)
                for distribution in distribution_candidates:
                    # compute fit, mle, kolmogorov-smirnov test to test fit, and pdf
                    fit = distribution.fit(data_as_list)
                    mle = distribution.nnlf(fit, data_as_list)
                    ktest = st.kstest(data_as_list, "norm", fit)

                    test_points = np.linspace(
                        min(data_as_list), max(data_as_list), 1000
                    )
                    pdf = distribution.pdf(test_points, fit)

                    if mle <= best_mle[0]:
                        best_mle = (distribution.name, mle, pdf, ktest)
                continuous_distributions[attribute] = best_mle
            else:
                continuous_distributions[attribute] = None

        return continuous_distributions

    @staticmethod
    def _get_categorical_distribution(self, graph):
        """
        Compute categorical probabilities of graph edge categorical attributes
        """
        attributes = GraphProfile.find_all_attributes(graph)
        categorical_attributes = GraphProfile._get_categorical_attributes(graph)

        distribution_candidates = [st.bernoulli, st.binom, st.geom, st.poisson]
        categorical_distributions = dict()

        for attribute in attributes:
            if attribute in categorical_attributes:
                best_mle = ("initial value", 1000)
                data_as_list = self._attribute_data_as_list(graph, attribute)

                for distribution in distribution_candidates:
                    # compute fit, mle, kolmogorov-smirnov test to test fit, and pdf
                    fit = distribution.fit(data_as_list)
                    mle = distribution.nnlf(fit, data_as_list)
                    ktest = st.kstest(data_as_list, "norm", fit)

                    test_points = np.linspace(
                        min(data_as_list), max(data_as_list), 1000
                    )
                    pdf = distribution.pdf(test_points, fit)

                    if mle <= best_mle[0]:
                        best_mle = (distribution.name, mle, pdf, ktest)
                categorical_distributions[attribute] = best_mle
            else:
                categorical_distributions[attribute] = None

        return categorical_distributions

    @staticmethod
    def _get_categorical_and_continuous_attributes(graph):
        """
        Finds and lists categorical and continuous attributes
        """
        categorical_attributes = []
        continuous_attributes = []
        attributes = GraphProfile._find_all_attributes(graph)
        for attribute in attributes:
            is_categorical = False
            for u, v in graph.edges():
                attribute_value = graph[u][v][attribute]
                if float(attribute_value).is_integer():
                    is_categorical = True
                    break
            if is_categorical:
                categorical_attributes.append(attribute)
            else:
                continuous_attributes.append(attribute)
        return (categorical_attributes, continuous_attributes)

    """
    Helper functions
    """

    @staticmethod
    def _find_all_attributes(graph):
        """
        Compute the number of attributes for each edge
        """
        attribute_list = set(
            np.array([list(graph.edges[n].keys()) for n in graph.edges()]).flatten()
        )
        return list(attribute_list)

    def _attribute_data_as_list(self, graph, attribute):
        """
        Fetches graph attribute data and converts it to a conveniently readable list
        """
        data_as_list = []
        for u, v in list(graph.edges):
            value = graph[u][v][attribute]
            data_as_list.append(value)
        return data_as_list
