"""Class and functions to calculate and profile properties of graph data."""
from __future__ import annotations

import pickle
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, cast

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st

from ..data_readers.graph_data import GraphData
from . import BaseColumnProfiler, utils
from .profiler_options import ProfilerOptions


class GraphProfiler(object):
    """
    GraphProfiler class.

    Creates a profile describing a graph dataset
    Statistical properties of graph
    """

    def __init__(
        self, data: Union[nx.Graph, GraphData], options: ProfilerOptions = None
    ) -> None:
        """
        Initialize Graph Profiler.

        :param data: data
        :type data: Union[networkx.Graph, data_readers.graph_data.GraphData]
        :param options: Options for the Graph Profiler
        :type options: GraphOptions
        """
        self.sample_size = 0
        self.times: Dict[str, float] = defaultdict(float)

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
        self.metadata: Dict = dict()

        self.__calculations = {
            "num_nodes": GraphProfiler._update_num_nodes,
            "num_edges": GraphProfiler._update_num_edges,
            "categorical_attributes": GraphProfiler._update_categorical_attributes,
            "continuous_attributes": GraphProfiler._update_continuous_attributes,
            "avg_node_degree": GraphProfiler._update_avg_node_degree,
            "global_max_component_size": GraphProfiler._update_global_max_comp_size,
            "continuous_distribution": GraphProfiler._update_continuous_distribution,
            "categorical_distribution": GraphProfiler._update_categorical_distribution,
        }

    def __add__(self, other: GraphProfiler) -> GraphProfiler:
        """
        Merge two Graph profiles together overriding the `+` operator.

        :param other: graph profile being added to this one.
        :type other: GraphProfiler
        :return: merger of the two profiles
        :rtype: GraphProfiler
        """
        raise NotImplementedError(
            "profile adding is not currently supported for the GraphProfiler"
        )

    @property
    def profile(self) -> Dict:
        """
        Return the profile of the graph.

        :return: the profile of the graph in data
        """
        profile = dict(
            num_nodes=self._num_nodes,
            num_edges=self._num_edges,
            categorical_attributes=self._categorical_attributes,
            continuous_attributes=self._continuous_attributes,
            avg_node_degree=self._avg_node_degree,
            global_max_component_size=self._global_max_component_size,
            continuous_distribution=self._continuous_distribution,
            categorical_distribution=self._categorical_distribution,
            times=self.times,
        )
        return profile

    def diff(self, other_profile: GraphProfiler, options: Dict = None) -> Dict:
        """
        Find the differences for two graph profiles.

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

        diff_profile = {
            "num_nodes": utils.find_diff_of_numbers(
                self._num_nodes, other_profile._num_nodes
            ),
            "num_edges": utils.find_diff_of_numbers(
                self._num_edges, other_profile._num_edges
            ),
            "categorical_attributes": utils.find_diff_of_lists_and_sets(
                self._categorical_attributes, other_profile._categorical_attributes
            ),
            "continuous_attributes": utils.find_diff_of_lists_and_sets(
                self._continuous_attributes, other_profile._continuous_attributes
            ),
            "avg_node_degree": utils.find_diff_of_numbers(
                self._avg_node_degree, other_profile._avg_node_degree
            ),
            "global_max_component_size": utils.find_diff_of_numbers(
                self._global_max_component_size,
                other_profile._global_max_component_size,
            ),
            "continuous_distribution": utils.find_diff_of_dicts_with_diff_keys(
                self._continuous_distribution,
                other_profile._continuous_distribution,
            ),
            "categorical_distribution": utils.find_diff_of_dicts_with_diff_keys(
                self._categorical_distribution,
                other_profile._categorical_distribution,
            ),
            "times": utils.find_diff_of_dicts(self.times, other_profile.times),
        }

        return diff_profile

    def report(self, remove_disabled_flag: bool = False) -> Dict:
        """
        Report on profile attribute of the class.

        Pop value from self.profile if key not in self.__calculations
        """
        calcs_dict_keys = self.__calculations.keys()
        profile = self.profile
        list_keys = [
            "num_nodes",
            "num_edges",
            "categorical_attributes",
            "continuous_attributes",
            "avg_node_degree",
            "global_max_component_size",
            "continuous_distribution",
            "categorical_distribution",
        ]

        if remove_disabled_flag:
            profile_keys = list(profile.keys())
            for profile_key in profile_keys:
                # need to add props
                if profile_key in list_keys:
                    if profile_key in calcs_dict_keys:
                        continue
                profile.pop(profile_key)
        return profile

    def _update_helper(self, profile: Dict) -> None:
        """
        Update the graph profile properties with a cleaned dataset.

        :param data: networkx graph
        :type data: NetworkX Graph
        :param profile: graph profile dictionary
        :type profile: dict
        :return: None
        """
        self.sample_size += profile.pop("sample_size")
        self.metadata = profile

    def update(self, graph: nx.Graph) -> GraphProfiler:
        """
        Update the graph profile.

        :param data: networkx graph
        :type data: NetworkX Graph
        :return: None
        """
        if not isinstance(graph, nx.Graph) and not isinstance(graph, GraphData):
            raise NotImplementedError(
                "Profiler only takes GraphData objects or NetworkXGraph"
            )
        if isinstance(graph, GraphData):
            graph = graph.data
        graph_size = graph.size()
        if graph_size == 0 and graph.number_of_nodes() == 0:
            return self
        profile = dict(sample_size=graph_size)

        BaseColumnProfiler._perform_property_calcs(
            self,  # type: ignore
            self.__calculations,
            df_series=graph,
            prev_dependent_properties={},
            subset_properties={},
        )

        self._update_helper(profile)

        return self

    """
    Update functions to update props with get functions
    """

    def _update_num_nodes(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update num_nodes for profile."""
        if subset_properties is None:
            subset_properties = {}

        self._num_nodes = self._get_num_nodes(graph)
        subset_properties["num_nodes"] = self._num_nodes

    def _update_num_edges(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update num_edges for profile."""
        self._num_edges = self._get_num_edges(graph)

    def _update_avg_node_degree(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update avg_node_degree for profile."""
        if subset_properties is None:
            subset_properties = {}

        self._avg_node_degree = self._get_avg_node_degree(
            graph, subset_properties.get("num_nodes", 0)
        )

    def _update_global_max_comp_size(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update global_max_component_size for profile."""
        self._global_max_component_size = self._get_global_max_component_size(graph)

    def _update_categorical_attributes(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update categorical_attributes for profile."""
        if subset_properties is None:
            subset_properties = {}

        self._categorical_attributes = self._get_categorical_attributes(graph)
        subset_properties["categorical_attributes"] = self._categorical_attributes

    def _update_continuous_attributes(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update continuous_attributes for profile."""
        if subset_properties is None:
            subset_properties = {}

        self._continuous_attributes = self._get_continuous_attributes(graph)
        subset_properties["continuous_attributes"] = self._continuous_attributes

    def _update_continuous_distribution(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update continuous_distribution for profile."""
        if subset_properties is None:
            subset_properties = {}

        self._continuous_distribution = self._get_continuous_distribution(
            graph, subset_properties.get("continuous_attributes", [])
        )

    def _update_categorical_distribution(
        self,
        graph: nx.Graph,
        prev_dependent_properties: Dict = None,
        subset_properties: Dict = None,
    ) -> None:
        """Update categorical_distribution for profile."""
        if subset_properties is None:
            subset_properties = {}

        self._categorical_distribution = self._get_categorical_distribution(
            graph, subset_properties.get("categorical_attributes", [])
        )

    """
    Get functions to calculate props
    """

    @BaseColumnProfiler._timeit(name="num_nodes")
    def _get_num_nodes(self, graph: nx.Graph) -> int:
        """Compute the number of nodes."""
        return cast(int, graph.number_of_nodes())

    @BaseColumnProfiler._timeit(name="num_edges")
    def _get_num_edges(self, graph: nx.Graph) -> int:
        """Compute the number of edges."""
        return cast(int, graph.number_of_edges())

    @BaseColumnProfiler._timeit(name="categorical_attributes")
    def _get_categorical_attributes(self, graph: nx.Graph) -> List[str]:
        """Fetch list of categorical attributes."""
        return self._get_categorical_and_continuous_attributes(graph)[0]

    @BaseColumnProfiler._timeit(name="continuous_attributes")
    def _get_continuous_attributes(self, graph: nx.Graph) -> List[str]:
        """Fetch list of continuous attributes."""
        return self._get_categorical_and_continuous_attributes(graph)[1]

    @BaseColumnProfiler._timeit(name="avg_node_degree")
    def _get_avg_node_degree(self, graph: nx.Graph, num_nodes: int) -> float:
        """Compute average node degree of nodes in graph."""
        total_degree = 0
        for node in graph:
            total_degree += graph.degree[node]
        return total_degree / num_nodes

    @BaseColumnProfiler._timeit(name="global_max_component_size")
    def _get_global_max_component_size(self, graph: nx.Graph) -> int:
        """Compute largest subgraph component of the graph."""
        graph_connected_components = sorted(
            nx.connected_components(graph), key=len, reverse=True
        )
        largest_component: nx.Graph = graph.subgraph(graph_connected_components[0])
        return cast(int, largest_component.size())

    @BaseColumnProfiler._timeit(name="continuous_distribution")
    def _get_continuous_distribution(
        self, graph: nx.Graph, continuous_attributes: List[str]
    ) -> Dict:
        """
        Compute the continuous distribution of graph edge continuous attributes.

        Returns properties array in the profile:
        [optional: shape, loc, scale, mean, variance, skew, kurtosis]

        - 6-property length: norm, uniform, expon, logistic
        - 7-property length: gamma, lognorm
            - gamma: shape=a
            - lognorm: shape=s
        """
        attributes = self._find_all_attributes(graph)
        continuous_distributions: Dict = dict()

        distribution_candidates: List[st.rv_continuous] = [
            st.norm,
            st.uniform,
            st.expon,
            st.logistic,
            st.lognorm,
            st.gamma,
        ]
        for attribute in attributes:
            if attribute in continuous_attributes:
                data_as_list = self._attribute_data_as_list(graph, attribute)
                df = pd.Series(data_as_list)
                best_fit: str = None  # type: ignore[assignment]
                best_mle: float = 1000
                best_fit_properties: Tuple = None  # type: ignore[assignment]

                for distribution in distribution_candidates:
                    # compute fit, mle, kolmogorov-smirnov test to test fit, and pdf
                    fit = distribution.fit(df)
                    mle = distribution.nnlf(fit, df)

                    if mle <= best_mle:
                        best_distrib = distribution
                        best_fit = distribution.name
                        best_mle = mle
                        best_fit_properties = fit

                mean, variance, skew, kurtosis = best_distrib.stats(
                    best_fit_properties, moments="mvsk"
                )
                properties: Dict[str, List[np.ndarray]] = {
                    "best_fit_properties": list(best_fit_properties),
                    "mean": list(mean),
                    "variance": list(variance),
                    "skew": list(skew),
                    "kurtosis": list(kurtosis),
                }
                continuous_distributions[attribute] = {
                    "name": best_fit,
                    "scale": best_mle,
                    "properties": properties,
                }
            else:
                continuous_distributions[attribute] = None
        return continuous_distributions

    @BaseColumnProfiler._timeit(name="categorical_distribution")
    def _get_categorical_distribution(
        self, graph: nx.Graph, categorical_attributes: List[str]
    ) -> Dict:
        """Compute histogram of graph edge categorical attributes."""
        attributes = GraphProfiler._find_all_attributes(graph)

        categorical_distributions: Dict = dict()

        for attribute in attributes:
            if attribute in categorical_attributes:
                data_as_list = self._attribute_data_as_list(graph, attribute)
                hist, edges = np.histogram(data_as_list, bins="auto", density=False)
                categorical_distributions[attribute] = {
                    "bin_counts": list(hist),
                    "bin_edges": list(edges),
                }
            else:
                categorical_distributions[attribute] = None
        return categorical_distributions

    @staticmethod
    def _get_categorical_and_continuous_attributes(
        graph: nx.Graph,
    ) -> Tuple[List[str], List[str]]:
        """Find and list categorical and continuous attributes."""
        categorical_attributes = []
        continuous_attributes = []
        attributes = GraphProfiler._find_all_attributes(graph)
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
    def _find_all_attributes(graph: nx.Graph) -> List[str]:
        """Compute the number of attributes for each edge."""
        attribute_list = set(
            np.array([list(graph.edges[n].keys()) for n in graph.edges()]).flatten()
        )
        return list(attribute_list)

    def _attribute_data_as_list(self, graph: nx.Graph, attribute: str) -> List:
        """Fetch graph attribute data and convert it to a readable list."""
        data_as_list = []
        for u, v in list(graph.edges):
            value = graph[u][v][attribute]
            data_as_list.append(value)
        return data_as_list

    def _save_helper(self, filepath: Optional[str], data_dict: Dict) -> None:
        """
        Save profiler to disk.

        :param filepath: Path of file to save to
        :type filepath: String
        :param data_dict: profile data to be saved
        :type data_dict: dict
        :return: None
        """
        # Set Default filepath
        if filepath is None:
            filepath = "profile-{}.pkl".format(
                datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
            )

        # add profiler class to data_dict
        data_dict["profiler_class"] = self.__class__.__name__

        # Pickle and save profile to disk
        with open(filepath, "wb") as outfile:
            pickle.dump(data_dict, outfile)

    def save(self, filepath: str = None) -> None:
        """
        Save profiler to disk.

        :param filepath: Path of file to save to
        :type filepath: String
        :return: None
        """
        # Create dictionary for all metadata, options, and profile
        data_dict = {
            "sample_size": self.sample_size,
            "times": self.times,
            "_attributes": self._attributes,
            "_num_nodes": self._num_nodes,
            "_num_edges": self._num_edges,
            "_categorical_attributes": self._categorical_attributes,
            "_continuous_attributes": self._continuous_attributes,
            "_avg_node_degree": self._avg_node_degree,
            "_global_max_component_size": self._global_max_component_size,
            "_continuous_distribution": self._continuous_distribution,
            "_categorical_distribution": self._categorical_distribution,
            "metadata": self.metadata,
        }

        self._save_helper(filepath, data_dict)

    @classmethod
    def load(cls, filepath: str) -> GraphProfiler:
        """
        Load profiler from disk.

        :param filepath: Path of file to load from
        :type filepath: String
        :return: GraphProfiler being loaded
        :rtype: GraphProfiler
        """
        # Load profile from disk
        with open(filepath, "rb") as infile:
            data = pickle.load(infile)

        profiler = cls(data=None, options=None)

        for key in data:
            setattr(profiler, key, data[key])

        return profiler
