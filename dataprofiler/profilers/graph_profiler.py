
import networkx as nx
import numpy as np
import scipy.stats as st

from collections import Counter, defaultdict

from . import BaseColumnProfiler, utils
from .profiler_options import TextProfilerOptions


class GraphProfile(object):

    def __init__(self, name, data=None, options=None):
        """
        Initialization of Graph Profiler.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Graph Profiler
        :type options: GraphOptions
        """
        if data is None:
            raise NotImplementedError("Data cannot be empty.")
        if not isinstance(data, nx.Graph):
            raise NotImplementedError("Data must be a valid NetworkX graph.")

        
        self.name = name
        self.sample_size = 0
        self.times = defaultdict(float)
        
        """
        Properties
        """
        self._graph = data
        self._attributes = self._find_all_attributes()

        self._num_nodes = self._compute_num_nodes()
        self._num_edges = self._compute_num_edges()
        
        attribute_tuple = self._find_categorical_and_continuous_attributes()
        self._categorical_attributes = attribute_tuple[0]
        self._continuous_attributes = attribute_tuple[1]

        # generic graph information
        self._avg_node_degree = self._compute_avg_node_degree()

        if not isinstance(data, nx.DiGraph):
            self._global_max_component_size = self._compute_global_max_component_size()
        else:
            self._global_max_component_size = None
            
        self._continuous_distributions = None #self._compute_continuous_distribution()
        self._categorical_probabilities = None #self._compute_categorical_probabilities()

        # case specific
        self._activation_probability = self._compute_activation_probabilities()
        self._deactivation_probability = self._compute_deactivation_probabilities()
        self._end_state_probabilities = self._compute_end_state_probabilities()
        self._link_probability = self._compute_link_probabilities()
        self._conditional_probabilities = self._compute_conditional_probabilities()

        self.metadata = dict()

        # self.line_length = {'max': None, 'min': None,...} #numeric stats mixin?

        if options and not isinstance(options, GraphOptions):
            raise ValueError(
                "Graphrofiler parameter 'options' must be of type"
                " GraphOptions."
            )
        """
        self._<PROP_option> = None
        if options:
            self.<PROP_option> = options.<PROP_option>

        self.__calculations = {
            "<PROP>": TextProfiler._update_<PROP>,
        }
        BaseColumnProfiler._filter_properties_w_options(self.__calculations, options)
        """

    @property
    def profile(self):
        """
        Returns the profile of the graph
        :return: the profile of the graph in data
        """
        profile = dict(
            num_nodes = self._num_nodes,
            avg_node_degree = self._avg_node_degree,
            global_max_component_size = self._global_max_component_size,
            continuous_distributions = self._continuous_distributions,
            categorical_probabilities = self._categorical_probabilities,
            activation_probability = self._activation_probability,
            deactivation_probability = self._deactivation_probability,
            end_state_probabilities = self._end_state_probabilities,
            link_probability = self._link_probability,
            conditional_probabilities = self._conditional_probabilities,
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
        calcs_dict_keys = self._GraphProfiler__calculations.keys()
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

    """
    Computation functions
    """
    def _find_all_attributes(self):
        """
        Compute the number of attributes for each edge
        """
        attribute_list = set(np.array([list(self._graph.edges[n].keys()) for n in self._graph.edges()]).flatten())
        return list(attribute_list)
    
    def _find_categorical_and_continuous_attributes(self):
        """
        Finds and lists categorical and continuous attributes
        """
        categorical_attributes = []
        continuous_attributes = []
        for attribute in self._attributes:
            is_categorical = False
            for u,v in self._graph.edges():
                attribute_value = self._graph[u][v][attribute]
                if float(attribute_value).is_integer():
                    is_categorical = True
                    break
            if is_categorical:
                categorical_attributes.append(attribute)
            else:
                continuous_attributes.append(attribute)
        return (categorical_attributes, continuous_attributes)
    
    def _attribute_data_as_list(self, attribute):
        """
        Fetches graph attribute data and converts it to a conveniently readable list
        """
        data_as_list = []
        for u,v in list(self._graph.edges):
            value = self._graph[u][v][attribute]
            data_as_list.append(value)
        return data_as_list

    """
    The following functions compute generic properties for the graph profile
    """
    def _compute_num_nodes(self):
        """
        Compute the number of nodes
        """
        return self._graph.number_of_nodes()

    def _compute_num_edges(self):
        """
        Compute the number of edges
        """
        return self._graph.number_of_edges()

    def _compute_avg_node_degree(self):
        """
        Compute average node degree of nodes in graph
        """
        total_degree = 0
        for node in self._graph:
            total_degree += self._graph.degree[node]
        return total_degree/self._num_nodes

    def _compute_global_max_component_size(self):
        """
        Compute largest subgraph component of the graph
        """
        graph_connected_components = sorted(nx.connected_components(self._graph), key=len, reverse=True)
        largest_component = self._graph.subgraph(graph_connected_components[0])
        return largest_component.size()

    def _compute_continuous_distribution(self, attribute):
        """
        Compute the continuous distribution of graph edge continuous attribute
        """
        # create a list compiling all data for that attribute
        data_as_list = self._attribute_data_as_list(attribute)

        distribution_candidates = [
            st.norm, 
            st.uniform, 
            st.expon, 
            st.logistic, 
            st.lognorm, 
            st.gamma
        ]
        best_mle = ("initial value", 1000)
        for distribution in distribution_candidates:
            # compute fit, mle, kolmogorov-smirnov test to test fit, and pdf
            fit = distribution.fit(data_as_list)
            mle = distribution.nnlf(fit, data_as_list)
            ktest = st.kstest(data_as_list , 'norm' , fit)

            test_points = np.linspace(min(data_as_list), max(data_as_list), 1000)
            pdf = distribution.pdf(test_points, fit)

            if mle <= best_mle[0]:
                best_mle = (distribution.name, mle, pdf, ktest)
        return best_mle

    def _compute_categorical_distribution(self, attribute):
        """
        Compute categorical probabilities of graph edge categorical attribute
        """
        # create a list compiling all data for that attribute
        data_as_list = self._attribute_data_as_list(attribute)

        distribution_candidates = [
            st.bernoulli, 
            st.binom, 
            st.geom, 
            st.poisson
        ]
        best_mle = ("initial value", 1000)
        for distribution in distribution_candidates:
            # compute fit, mle, kolmogorov-smirnov test to test fit, and pdf
            fit = distribution.fit(data_as_list)
            mle = distribution.nnlf(fit, data_as_list)
            ktest = st.kstest(data_as_list, 'norm' , fit)

            test_points = np.linspace(min(data_as_list), max(data_as_list), 1000)
            pdf = distribution.pdf(test_points, fit)

            if mle <= best_mle[0]:
                best_mle = (distribution.name, mle, pdf, ktest)
        return best_mle

    """
    The following functions compute case-specific properties for the graph profile
    """
    def _compute_conditional_probabilities(self):
        return
    def _compute_end_state_probabilities(self):
        return
    def _compute_link_probabilities(self):
        return
    def _compute_activation_probabilities(self):
        return
    def _compute_deactivation_probabilities(self):
        return