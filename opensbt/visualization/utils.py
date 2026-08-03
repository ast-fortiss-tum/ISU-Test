import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from typing import List, Tuple
from collections import defaultdict
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from opensbt.analysis.statistics.wilcoxon import run_wilcoxon_and_delaney
import logging
import sys
import math
import tqdm
import matplotlib.pyplot as plt
import random
from collections import Counter


logging.getLogger("matplotlib").setLevel(logging.WARNING)


def close_loggers() -> None:
    # Remove all handlers associated with the root logger object. Needed to call logging.basicConfig multiple times
    # such that for different experiment runs a new log file is written
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


class Log:
    def __init__(self, logger_prefix: str, static_logger: bool = True) -> None:
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0 or not static_logger:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)

            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)




class AnalysisException(Exception):
    pass


class AnalysisPlots:
    plots_on_ax = defaultdict(int)
    label_colors = dict()

    @staticmethod
    def interpolate(x: np.ndarray, target_len: int, noise: bool = False):
        x = x.reshape((-1))
        x = [v for v in x]
        while len(x) < target_len:
            new_value = 2 * x[-1] - x[-2]
            if noise:
                new_value *= (random.random() * 0.04 + 0.98)
            x.append(new_value)
        return np.array(x)
   
    @classmethod
    def plot_with_std(cls, ax, dfs: List[pd.DataFrame], label, metric: str = 'failures', color=None, target_time=None, **kwargs):
        """
        Plot mean ± std of multiple runs on the same Axes.
        Parameters:
            ax: matplotlib Axes
            dfs: list of DataFrames (run histories) to plot
            metric: column name to plot
            color: optional, fixed color to use
            kwargs: additional keyword args passed to ax.plot (e.g., label)
        """
        # Initialize counter for this Axes
        if ax not in cls.plots_on_ax:
            cls.plots_on_ax[ax] = 0

        # Use fixed color if provided, else auto
        if color is None:
            if label in cls.label_colors:
                color = cls.label_colors[label]
            else:
                color = f"C{cls.plots_on_ax[ax]}"
                cls.label_colors[label] = color
            cls.plots_on_ax[ax] += 1

        # Prepare data
        xs = [df[metric].to_numpy() for df in dfs]
        if target_time is not None:
            xs = [AnalysisPlots.interpolate(x, target_time+1) for x in xs]
        min_len = min(target_time + 1, min(x.shape[0] for x in xs))
        xs = np.concatenate([x[:min_len].reshape(-1, 1) for x in xs], axis=1)
        avg = xs.mean(1)
        std = np.std(xs, 1)

        # Convert time index to minutes
        if target_time is None:
            time = dfs[0].index[:min_len].total_seconds() / 60  # assuming TimedeltaIndex
        else:
            time = np.arange(target_time + 1)
        # Plot mean
        ax.plot(time, avg, color=color, **kwargs)

        # Fill ± std
        ax.fill_between(x=time, y1=avg-std, y2=avg+std, color=color, alpha=0.5)

    @classmethod
    def boxplot(cls, ax, dfs_lists: List[List[pd.DataFrame]], labels, metric: str = 'failures',**kwargs):
        # Prepare data
        xs = [[df[metric].iloc[-1] for df in dfs] for dfs in dfs_lists]
        # Plot 
        bplot = ax.boxplot(xs, patch_artist=True, **kwargs)
        for patch, label in zip(bplot['boxes'], labels):
            if label in cls.label_colors:
                color = cls.label_colors[label]
            else:
                color = f"C{len(cls.label_colors)}"
                cls.label_colors[label] = color
            patch.set_facecolor(color)

    @staticmethod
    def plot_clusters(
        to_cluster,
        centers,
        filename,
        names,
        env_configurations_names,
        visualize_name=None,
        seed=0,
    ):
        colors = [
            "black",
            "green",
            "blue",
            "purple",
            "magenta",
            "pink",
            "grey",
            "cyan",
            "lime",
            "orange",
            "peru",
        ]
        markers = ["o", "^", "2", "*", "s", "+", "x", "D", "h", "8", ">"]
        for c in centers:
            # add center of cluster as a new row
            to_cluster = np.append(to_cluster, c.reshape(1, -1), axis=0)

        for per in [2, 5, 10, 20, 30]:
            # assuming TSNE
            n_iterations = 5000
            tsne = TSNE(
                n_components=2,
                perplexity=per,
                n_jobs=-1,
                n_iter=n_iterations,
                random_state=seed,
            )
            embeddings = tsne.fit_transform(to_cluster)

            _ = plt.figure()
            ax = plt.gca()
            ax.tick_params(left=False)
            ax.tick_params(bottom=False)
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticklabels([])

            count_configs = 0
            name_idx = 0

            embeddings_names = dict()
            for i in range(len(embeddings) - len(centers)):
                if (
                    count_configs
                    > env_configurations_names[names[name_idx]] - 1
                ):
                    count_configs = 0
                    name_idx += 1

                if names[name_idx] not in embeddings_names:
                    embeddings_names[names[name_idx]] = ([], [])
                embeddings_names[names[name_idx]][0].append(embeddings[i][0])
                embeddings_names[names[name_idx]][1].append(embeddings[i][1])

                count_configs += 1

            for nm, ems in embeddings_names.items():
                assert len(ems[0]) == env_configurations_names[nm], "Number of embeddings {} != Number of failures: {} for {}".format(
                    len(ems[0]), env_configurations_names[nm], nm
                )

            for i, name_embeddings in enumerate(embeddings_names.items()):
                nm, embeddings_ = name_embeddings

                if visualize_name is None or visualize_name == nm:
                    plt.scatter(
                        embeddings_[0],
                        embeddings_[1],
                        s=80,
                        color=colors[i],
                        marker=markers[i],
                        label=nm,
                        alpha=0.6
                    )

            embeddings_centroids = ([], [])
            for i in range(
                len(embeddings) - 1, len(embeddings) - 1 - len(centers), -1
            ):
                embeddings_centroids[0].append(embeddings[i][0])
                embeddings_centroids[1].append(embeddings[i][1])

            plt.scatter(
                embeddings_centroids[0],
                embeddings_centroids[1],
                s=50,
                color="red",
                marker="$c$",
                label="centroid",
            )

            plt.legend(prop={"size": 6})

            plt.savefig(f"{filename}_per{per}.pdf", format="pdf")
            plt.clf()
            plt.close()


class AnalysisTables:
    @staticmethod
    def statistics(values: List[List], algo_names: List[str]):
        result = defaultdict(lambda: defaultdict(tuple))
        for i in range(len(algo_names)):
            print("i:", i)
            for j in range(i + 1, len(algo_names)):
                print("j:", j)
                algo1 = algo_names[i]
                algo2 = algo_names[j]
                data1 = values[i]
                data2 = values[j]
                print(f"Comparing {algo1} vs {algo2}")
                print(f"Data1: {data1}")
                print(f"Data2: {data2}")
                if len(data1) == len(data2) and len(data1) > 2:
                    stats = run_wilcoxon_and_delaney(data1, data2)
                    result[algo1][algo2] = stats
                    stats = run_wilcoxon_and_delaney(data2, data1)
                    result[algo2][algo1] = stats
            print("<Results>", result)
            print("----")
        return result


class AnalysisDiversity:
    @staticmethod
    def average_max_distance(embeddings: np.ndarray):
        distances = distance_matrix(embeddings, embeddings)
        return distances.max(0).mean()
    
    @staticmethod
    def average_distance(embeddings: np.ndarray):
        distances = distance_matrix(embeddings, embeddings)
        return distances.mean(0).mean()    

    @staticmethod
    def cluster_data(
        data: np.ndarray,
        n_clusters_interval: Tuple,
        seed: int,
        silhouette_threshold: int,
    ) -> Tuple:
        logger = Log("cluster_data")
        optimal_n_clusters = 1
        optimal_score = -1
        if n_clusters_interval[0] != 1:
            range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
            optimal_score = -1
            optimal_n_clusters = -1
            for n_clusters in tqdm.tqdm(range_n_clusters):
                clusterer = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
                cluster_labels = clusterer.fit_predict(data)
                try:
                    silhouette_avg = silhouette_score(
                        data, cluster_labels, random_state=seed
                    )
                    logger.debug(
                        "For n_clusters = {}, the average silhouette score is: {}".format(
                            n_clusters, silhouette_avg
                        )
                    )
                    if silhouette_avg > optimal_score:
                        if silhouette_threshold < 0 or optimal_score == -1:
                            optimal_score = silhouette_avg
                            optimal_n_clusters = n_clusters
                            logger.debug(
                                "New optimal silhouette score: {}. Num clusters: {}".format(
                                    silhouette_avg, n_clusters
                                )
                            )
                        else:
                            assert (
                                0 < silhouette_threshold <= 100
                            ), "Silhouette threshold needs to be in (0, 100]. Found {}".format(
                                silhouette_threshold
                            )
                            percentage_increase = round(
                                100 * (silhouette_avg - optimal_score) / optimal_score, 2
                            )
                            if percentage_increase >= silhouette_threshold:
                                optimal_score = silhouette_avg
                                optimal_n_clusters = n_clusters
                                logger.debug(
                                    "New optimal silhouette score: {} > {}% of previous score {}. Num clusters: {}".format(
                                        silhouette_avg,
                                        silhouette_threshold,
                                        percentage_increase,
                                        n_clusters,
                                    )
                                )
                except ValueError:
                    logger.warn("Clustering ValueError exception")
                    break

            assert optimal_n_clusters != -1, "Error in silhouette analysis"
            logger.debug(
                "Best score is {} for n_cluster = {}".format(
                    optimal_score, optimal_n_clusters
                )
            )

        clusterer = KMeans(n_clusters=optimal_n_clusters, n_init="auto").fit(data)
        labels = clusterer.labels_
        centers = clusterer.cluster_centers_
        return clusterer, labels, centers, optimal_score
    
    @staticmethod
    def compute_coverage_entropy(
        labels,
        centers,
        names,
        env_configurations_names,
    ):
        coverage_names = dict()
        entropy_names = dict()
        logger = Log("coverage_entropy")
        num_clusters = len(centers)
        coverage_set_names = dict()
        number_points_names = dict()
        name_idx = 0
        count_labels = 0
        for label in labels:
            if count_labels > env_configurations_names[names[name_idx]] - 1:
                count_labels = 0
                name_idx += 1

            nm = names[name_idx]
            if nm not in coverage_set_names:
                coverage_set_names[nm] = set()
            coverage_set_names[nm].add(label)

            if nm not in number_points_names:
                number_points_names[nm] = []
            number_points_names[nm].append(label)

            count_labels += 1

        for nm, ls in number_points_names.items():
            assert len(ls) ==  env_configurations_names[nm] , "Number of labels {} != Number of failures: {} for {}".format(
                len(ls), env_configurations_names[nm], nm
            )

        logger.info("Number of clusters: {}".format(len(centers)))
        ideal_entropy = np.log2(len(centers))
        num_classes = len(env_configurations_names)
        logger.info(f"Num classes: {num_classes}")

        gini_dict = dict()
        counter_labels = Counter(labels)

        for nm, set_clusters in coverage_set_names.items():

            # TODO: computing gini coefficient only if with two classes for now
            if len(set_clusters) > 0 and num_classes == 2:
                number_points_name_array = np.asarray(number_points_names[nm])
                gini_dict[nm] = [
                    len(
                        number_points_name_array[
                            number_points_name_array == cluster_label
                        ]
                    )
                    / counter_labels[cluster_label]
                    for cluster_label in counter_labels
                ]

            coverage = 100 * len(set_clusters) / len(centers)
            coverage_names[nm] = coverage
            logger.info("Coverage for method {}: {}%".format(nm, coverage))
            counter = Counter(number_points_names[nm])
            distribution = [
                round(
                    100
                    * number_of_points_in_cluster
                    / env_configurations_names[nm],
                    2,
                )
                for number_of_points_in_cluster in counter.values()
            ]

            logger.info(
                "Distribution across clusters for method {}: {}".format(
                    nm, distribution
                )
            )
            entropy_names[nm] = 0.0
            if len(set_clusters) > 0:
                frequency_list = [
                    number_of_points_in_cluster / env_configurations_names[nm]
                    for number_of_points_in_cluster in counter.values()
                ]

                entropy = 0.0
                for freq in frequency_list:
                    entropy += freq * np.log2(freq)
                if not math.isclose(entropy, 0.0):
                    entropy *= -1
                logger.info("Entropy: {}, Ideal: {}".format(entropy, ideal_entropy))
                entropy = round(100 * entropy / ideal_entropy, 2)
                entropy_names[nm] = entropy
                logger.info("Entropy: {}%".format(entropy))

        # TODO: computing gini coefficient only if with two classes for now
        if num_classes == 2:
            same_length = all(
                len(probabilities) == len(list(gini_dict.values())[0])
                for probabilities in gini_dict.values()
            )
            assert (
                same_length
            ), f"All the probability scores are not of the same length: {gini_dict}"

            logger.info(f"Gini impurity coefficient dictionary: {gini_dict}")

            gini_impurity_coeff = 0
            for i in range(num_clusters):

                if sum([gini_dict[nm][i] for nm in gini_dict.keys()]) != 1.0:
                    raise RuntimeError(
                        f"Error in computing the gini coefficient: {gini_dict}"
                    )

                gini_impurity_coeff += 1 - sum(
                    [gini_dict[nm][i] ** 2 for nm in gini_dict.keys()]
                )

            gini_impurity_coeff /= num_clusters

            if gini_impurity_coeff > 1.0:
                raise RuntimeError(
                    f"The gini coefficient cannot be > 1.0: {gini_impurity_coeff}"
                )

            logger.info(f"Gini impurity coefficient: {gini_impurity_coeff}")
            logger.info(f"Gini purity coefficient: {1 - gini_impurity_coeff}")
        
        return coverage_names, entropy_names, num_clusters
