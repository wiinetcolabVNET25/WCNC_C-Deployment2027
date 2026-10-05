import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .plot_utils import Plot

from itertools import combinations

class Preprocessing:

    @staticmethod
    def remove_outliers(df: pd.DataFrame, features: list[str],
                        method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove rows containing outliers in any of `features`.

        Parameters
        ----------
        method:
            "iqr" or "zscore".

        threshold:
            IQR multiplier or absolute z-score threshold.

        Returns
        -------
        DataFrame with outlier rows removed.
        """

        if df.empty:
            return df.copy()

        data = df[features]

        if method == "iqr":

            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1

            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr

            mask = (
                (data >= lower) & 
                (data <= upper)
            ).all(axis=1)

        elif method == "zscore":

            mean, std = data.mean(), data.std()

            # Avoid division by zero for constant features.
            std = std.replace(0, 1)

            z = (data - mean) / std

            mask = (z.abs() <= threshold).all(axis=1)

        else:

            raise ValueError(f"Unknown outlier method: {method}")

        return df.loc[mask].copy()

    @staticmethod
    def do_standard_scale(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
        """
        Standardize the selected features.

        Returns
        -------
        scaled_df:
            Copy of df with the selected features standardized.

        scaler:
            Fitted StandardScaler.
        """

        result = df.copy()

        scaler = StandardScaler()

        result[features] = scaler.fit_transform(result[features])

        return result, scaler

class Clustering:

    @staticmethod
    def evaluate_cluster(X, labels) -> dict:
        """
        Evaluate a clustering.

        DBSCAN noise points (-1) are excluded from the
        clustering quality metrics.

        Returns
        -------

        metrics:
            n_clusters, n_noise, noise_ratio, silhouette, calinski_harabasz, davies_bouldin
        """

        X = np.asarray(X)
        labels = np.asarray(labels)

        # Remove DBSCAN noise.
        mask = labels != -1

        X_eval = X[mask]
        labels_eval = labels[mask]

        n_clusters = len(np.unique(labels_eval))

        # Silhouette, CH and DB require at least 2 clusters.
        if n_clusters < 2 or len(X_eval) <= n_clusters:
            return {
                "n_clusters": n_clusters,
                "n_noise": np.sum(~mask),
                "noise_ratio": np.mean(~mask),
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
            }
             
        return {
            "n_clusters": n_clusters,
            "n_noise": np.sum(~mask),
            "noise_ratio": np.mean(~mask),

            "silhouette": silhouette_score(X_eval, labels_eval),
            "calinski_harabasz": calinski_harabasz_score(X_eval, labels_eval),
            "davies_bouldin": davies_bouldin_score(X_eval, labels_eval),
        }

    @staticmethod
    def get_clusters_silhouette_scores(X, labels) -> tuple[dict[int, list[float]], dict]:
        """
        Evaluate clustering and calculate per-sample silhouette scores.

        Returns
        -------
        silhouette_by_cluster:
            Dictionary mapping cluster ID to the silhouette scores
            of its samples, ordered in descending order.

            DBSCAN noise (-1) is excluded.

        metrics:
            Dictionary containing:
                n_clusters
                n_noise
                noise_ratio
                silhouette
        """

        X = np.asarray(X)
        labels = np.asarray(labels)

        # Remove DBSCAN noise.
        mask = labels != -1

        X_eval = X[mask]
        labels_eval = labels[mask]

        cluster_ids = sorted(np.unique(labels_eval))
        n_clusters = len(cluster_ids)

        # Silhouette requires at least 2 clusters.
        if n_clusters < 2 or len(X_eval) <= n_clusters:

            return (
                {
                    cluster_id: []
                    for cluster_id in cluster_ids
                },
                {
                    "n_clusters": n_clusters,
                    "n_noise": np.sum(~mask),
                    "noise_ratio": np.mean(~mask),
                    "silhouette": np.nan,
                },
            )

        sample_scores = silhouette_samples(X_eval, labels_eval)

        silhouette_by_cluster = {
            cluster_id: sorted(sample_scores[labels_eval == cluster_id], reverse=True)
            for cluster_id in cluster_ids
        }

        silhouette = float(sample_scores.mean())

        return (
            silhouette_by_cluster,
            {
                "n_clusters": n_clusters,
                "n_noise": np.sum(~mask),
                "noise_ratio": np.mean(~mask),
                "silhouette": silhouette,
            },
        )

    @staticmethod
    def run_kmeans(df: pd.DataFrame, features: list[str],
                   k: int, n_runs: int = 1, random_state: int = 42) \
        -> tuple[pd.DataFrame, KMeans, float, int]:
        """
        Run KMeans multiple times and return the run with the best
        silhouette score.

        Parameters
        ----------
        n_runs:
            Number of KMeans executions.

        random_state:
            Seed used to generate the seeds for the individual runs.

        Returns
        -------
        clustered_df:
            Original dataframe with an additional `cluster` column,
            corresponding to the best run.

        model:
            Fitted KMeans model from the best run.

        score:
            Best run score.

        seed:
            Best run seed.

        """

        X = df[features].to_numpy() 

        rng = np.random.default_rng(random_state)

        best_score = -np.inf
        best_model = None
        best_seed = None
        best_labels = None

        for _ in range(n_runs):

            seed = int(rng.integers(0, 2**32 - 1))

            model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels = model.fit_predict(X)

            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_model = model
                best_seed = seed
                best_labels = labels

        result = df[features].copy()
        result["cluster"] = best_labels

        cluster_sizes = result["cluster"].value_counts().sort_values()
        cluster_mapping = { old_id: new_id
            for new_id, old_id in enumerate(cluster_sizes.index)}
        result["cluster"] = result["cluster"].map(cluster_mapping)

        return result, best_model, best_score, best_seed

    @staticmethod
    def run_minibatch_kmeans(df: pd.DataFrame, features: list[str],
        k: int, n_runs: int = 10, random_state: int = 42,
        max_iter: int = 100, batch_size: int = 1024) \
            -> tuple[pd.DataFrame, MiniBatchKMeans, float, int]:
        """
        Run MiniBatchKMeans multiple times and return the run
        with the best silhouette score.

        Parameters
        ----------
        n_runs:
            Number of MiniBatchKMeans executions.

        random_state:
            Seed used to generate the seeds for the individual runs.

        Returns
        -------
        clustered_df:
            Original dataframe with an additional `cluster` column,
            corresponding to the best run.

        model:
            Fitted MiniBatchKMeans model from the best run.

        score:
            Best run score.

        seed:
            Best run seed.

        """
        
        X = df[features].to_numpy()

        rng = np.random.default_rng(random_state)

        best_score = -np.inf
        best_model = None
        best_seed = None
        best_labels = None

        for _ in range(n_runs):

            seed = int(rng.integers(0, 2**32 - 1))

            model = MiniBatchKMeans(n_clusters=k, random_state=seed,
                max_iter=max_iter, batch_size=batch_size, n_init="auto")
            labels = model.fit_predict(X)

            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_model = model
                best_seed = seed
                best_labels = labels

        result = df[features].copy()
        result["cluster"] = best_labels

        cluster_sizes = result["cluster"].value_counts().sort_values()
        cluster_mapping = { old_id: new_id
            for new_id, old_id in enumerate(cluster_sizes.index)}
        result["cluster"] = result["cluster"].map(cluster_mapping)

        return result, best_model, best_score, best_seed

    @staticmethod
    def run_dbscan(df: pd.DataFrame, features: list[str],
                   eps: float = 0.5, min_samples: int = 5,
                   metric: str = "euclidean", p: float = 2, n_jobs: int = None) -> tuple[pd.DataFrame, DBSCAN]:
        """
        Run DBSCAN.

        Input data is expected to already be preprocessed/scaled.

        Parameters
        ----------
        eps:
            Maximum distance between two samples for them to be
            considered neighbors.

        min_samples:
            Minimum number of samples required to form a dense region.

        metric:
            Distance metric.

        p:
            Power parameter for Minkowski distance.

        Returns
        -------
        clustered_df:
            Copy of df with the selected features and an additional column "cluster".

        dbscan:
            DBSCAN model fit.
        """

        X = df[features].to_numpy()

        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p, n_jobs=n_jobs)

        labels = model.fit_predict(X)

        result = df.copy()
        result["cluster"] = labels

        cluster_sizes = result["cluster"].value_counts().sort_values()
        cluster_mapping = { old_id: new_id
            for new_id, old_id in enumerate(cluster_sizes.index)}
        result["cluster"] = result["cluster"].map(cluster_mapping)

        return result, model  

    @staticmethod
    def clustered_celldf_to_dict(df: pd.DataFrame) -> dict:
        """
        Convert a cell dataframe indexed as 'x:y' into
        a dictionary mapping cluster IDs
        to lists of cell coordinates.
        """

        d = {}
        for cluster_id, cluster_df in df.groupby("cluster"):

            x = cluster_df.index.str.split(":").str[0].astype(int)
            y = cluster_df.index.str.split(":").str[1].astype(int)

            d[cluster_id] = list(zip(x, y))

        return d

    @staticmethod
    def clustered_celldf_to_list(df: pd.DataFrame) -> list[list[tuple]]:
        """
        Convert a cell dataframe indexed as 'x:y' into
        a list of clusters.

        DBSCAN noise (-1) is included as a cluster unless
        explicitly removed beforehand.
        """

        l = []
        for _, cluster_df in df.groupby("cluster"):

            x = cluster_df.index.str.split(":").str[0].astype(int)
            y = cluster_df.index.str.split(":").str[1].astype(int)
            l.append(list(zip(x, y)))

        return l

    @staticmethod
    def plot_celldf_best_kmeans_results_for_each_feature_combination_and_k(
        df: pd.DataFrame, features_combs: list[list[str]],
        ks: list[int], n_runs: int = 20, figsize: tuple[float, float] | None = None, nrows=1):
        """
        Run KMeans for several (features, k) combinations and visualize
        the best run according to silhouette score.

        Each combination gets two subplots:
            - spatial cell clusters
            - silhouette scores by cluster
        """

        if len(features_combs) != len(ks):
            raise ValueError("features_combs and ks must have the same length.")

        n = len(features_combs)
        if figsize is None:
            figsize = (12 * n, 5 * nrows)
        ncols = n // nrows + (1 if (n % nrows != 0) else 0)

        fig, axes = plt.subplots(nrows=nrows, ncols=2*ncols,
                                 figsize=figsize, squeeze=False)
        row = 0
        col = 0
        for features, k in zip(features_combs, ks):

            cluster_ax = axes[row, 2 * col]
            silhouette_ax = axes[row, 2 * col + 1]

            result, _, _, _ = Clustering.run_kmeans(df=df, features=features, k=k, n_runs=n_runs)

            eval_result = Clustering.evaluate_cluster(X=result[features], labels=result["cluster"])

            cluster_ids_sorted_by_size = result["cluster"].value_counts(ascending=True).index.tolist()
            cmap = plt.get_cmap("tab10")
            colors = {
                cluster_id: cmap(i)
                for i, cluster_id in enumerate(cluster_ids_sorted_by_size)
            }
            clusters_dict = Clustering.clustered_celldf_to_dict(df=result)

            Plot.plot_cell_clusters(
                clusters_dict=clusters_dict,
                ax=cluster_ax,
                marker_size=5,
                colors=colors
            )

            silhouette_by_cluster, silhouette_metrics = Clustering.get_clusters_silhouette_scores(
                X=result[features], labels=result["cluster"])

            y_lower = 10

            for i, cluster_id in enumerate(cluster_ids_sorted_by_size):

                scores = np.asarray(silhouette_by_cluster[cluster_id])

                cluster_size = len(scores)
                y_upper = y_lower + cluster_size

                y_values = np.arange(y_lower, y_upper)

                silhouette_ax.fill_betweenx(
                    y_values,
                    0,
                    scores,
                    color=cmap(i)
                )

                silhouette_ax.text(
                    -0.05,
                    y_lower + 0.5 * cluster_size,
                    str(cluster_id),
                    va="center",
                )

                y_lower = y_upper + 10

            silhouette_ax.axvline(silhouette_metrics["silhouette"], linestyle="--")

            title = (
                f"k={k}\n"
                f"features={features}\n"
                f"silhouette={eval_result['silhouette']:.3f}\n"
                f"CH={eval_result['calinski_harabasz']:.3f}\n"
                f"Davies={eval_result['davies_bouldin']:.3f}"
            )

            cluster_ax.set_title(title)
            silhouette_ax.set_title("Silhouette scores")

            col += 1

            if col == ncols:
                row += 1
                col = 0

        plt.tight_layout()

        return fig, axes

    @staticmethod
    def plot_celldf_clustering_result(
        result_df: pd.DataFrame, features: list[str],
        figsize: tuple[float, float] | None = None,
    ):
        """
        Plot a single clustering result.

        Parameters
        ----------
        result_df:
            Dataframe containing the original cell data, feature columns,
            and a `cluster` column.

        features:
            Features used for clustering.

        figsize:
            Figure size used when axes are not supplied.

        Returns
        -------
        fig, axes
            Matplotlib figure and the two axes.
        """

        fig, axes = plt.subplots(1, 2, figsize=figsize or (12, 5), squeeze=False)

        cluster_ax = axes[0, 0]
        silhouette_ax = axes[0, 1]

        # Clustering metrics
        eval_result = Clustering.evaluate_cluster(
            X=result_df[features], labels=result_df["cluster"])

        cluster_ids_sorted_by_size = result_df["cluster"].value_counts(ascending=True).index.tolist()
        cmap = plt.get_cmap("tab10")
        colors = {
            cluster_id: cmap(i)
            for i, cluster_id in enumerate(cluster_ids_sorted_by_size)
        }
        clusters_dict = Clustering.clustered_celldf_to_dict(df=result_df)

        # Spatial clusters
        Plot.plot_cell_clusters(
            clusters_dict=clusters_dict,
            ax=cluster_ax,
            marker_size=5,
            colors=colors
        )

        # Silhouette scores
        silhouette_by_cluster, silhouette_metrics = (
            Clustering.get_clusters_silhouette_scores(
                X=result_df[features],
                labels=result_df["cluster"],
            )
        )

        y_lower = 10

        for i, cluster_id in enumerate(cluster_ids_sorted_by_size):

            scores = np.asarray(silhouette_by_cluster[cluster_id])

            cluster_size = len(scores)
            y_upper = y_lower + cluster_size

            y_values = np.arange(y_lower, y_upper)

            silhouette_ax.fill_betweenx(
                y_values,
                0,
                scores,
                color=cmap(i)
            )

            silhouette_ax.text(
                -0.05,
                y_lower + 0.5 * cluster_size,
                str(cluster_id),
                va="center",
            )

            y_lower = y_upper + 10

        silhouette_ax.axvline(
            silhouette_metrics["silhouette"],
            linestyle="--",
        )

        k = result_df["cluster"].nunique()

        title = (
            f"k={k}\n"
            f"features={features}\n"
            f"silhouette={eval_result['silhouette']:.3f}\n"
            f"CH={eval_result['calinski_harabasz']:.3f}\n"
            f"Davies={eval_result['davies_bouldin']:.3f}"
        )

        cluster_ax.set_title(title)
        silhouette_ax.set_title("Silhouette scores")

        plt.tight_layout()

        return fig, axes

    @staticmethod
    def evaluate_kmeans_feature_combinations(
        df: pd.DataFrame, feature_columns: list[str] | None = None,
        min_features: int = 1, max_features: int | None = None,
        k_values: list[int] = (2, 3, 4), n_runs: int = 10) -> pd.DataFrame:
        """
        Evaluate KMeans for every combination of features.

        Parameters
        ----------
        df:
            Input dataframe.

        feature_columns:
            Columns to consider. If None, all dataframe columns are used.

        min_features:
            Minimum number of features in a combination.

        max_features:
            Maximum number of features in a combination.
            If None, all features are considered.

        k_values:
            Number of clusters to test.

        n_runs:
            Number of KMeans executions for each (features, k).

        Returns
        -------
        pd.DataFrame
            One row per feature-combination / k / run.
        """

        if feature_columns is None:
            feature_columns = df.columns.tolist()

        if max_features is None:
            max_features = len(feature_columns)

        if min_features < 1:
            raise ValueError("min_features must be >= 1")

        if max_features > len(feature_columns):
            raise ValueError("max_features cannot exceed the number of features")

        results = []

        for n_features in range(min_features, max_features + 1):

            for combination in combinations(feature_columns, r=n_features):

                features = list(combination)

                result = Clustering.evaluate_kmeans(
                    df=df, features=features, k_values=k_values, n_runs=n_runs)

                results.append(result)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def generate_feature_combinations(features: list[str]):
        """
        Returns all non-empty features combinations.
        """
        return [
            combo
            for r in range(1, len(features) + 1)
            for combo in combinations(features, r=r)
        ] 

    @staticmethod
    def get_dbscan_min_samples(n_features: int, offsets=(0, 2, 4, 8, 10)):
        base = 2 * n_features
        return [base + offset for offset in offsets]

    @staticmethod
    def calculate_k_distances(df: pd.DataFrame, features: list[str],
                              min_samples_values: list[int], metric="euclidean", p=2):
        """
        Calculate sorted k-distances for one feature space.

        Each returned row contains:
            - features
            - min_samples
            - rank
            - k_distance
        """

        X = df[features].to_numpy()

        results = []
        for min_samples in min_samples_values:

            nn = NearestNeighbors(n_neighbors=min_samples, metric=metric, p=p, n_jobs=8)
            nn.fit(X)

            distances, _ = nn.kneighbors(X)
            k_distances = distances[:, min_samples - 1]
            k_distances = np.sort(k_distances)[::-1]

            for rank, distance in enumerate(k_distances):
                results.append({
                    "features": tuple(features),
                    "n_features": len(features),
                    "min_samples": min_samples,
                    "rank": rank,
                    "k_distance": distance,
                })

        return pd.DataFrame(results)

    @staticmethod
    def dbscan_step_1_get_feature_combinations_k_distances(
        df: pd.DataFrame, features: list[str],
        metric="euclidean", p=2,
        min_samples_offsets=(0, 2, 4, 8, 10)):
        """
        Stage 1 of DBSCAN parameter selection.

        For every non-empty feature combination:
            1. choose min_samples sequence based on dimensionality;
            2. calculate sorted k-distances.

        Returns
        -------
        pd.DataFrame
            One row per (feature combination, min_samples, rank).
        """
        feature_sets = Clustering.generate_feature_combinations(features)

        all_results = []

        for feature_set in feature_sets:

            min_samples_values = Clustering.get_dbscan_min_samples(
                n_features=len(feature_set), offsets=min_samples_offsets)

            result = Clustering.calculate_k_distances(
                df=df, features=list(feature_set),
                min_samples_values=min_samples_values,
                metric=metric, p=p,
            )

            all_results.append(result)

        return pd.concat(
            all_results,
            ignore_index=True,
        )

    @staticmethod
    def dbscan_plot_feature_combinations_k_distances(
        k_distance_results: pd.DataFrame,
        ncols=3, figsize_per_plot=(5, 4),
        xlims_per_plot: tuple[float, float]=(-10, 1000),
        ylims_per_plot: tuple[float, float]=(0, 10)):
        """
        Plot k-distance curves for all feature combinations.

        Each subplot corresponds to one feature combination.
        Different min_samples values are shown as separate curves.
        """

        feature_sets = (
            k_distance_results[["features", "n_features"]]
            .drop_duplicates()
            .sort_values(["n_features", "features"])
            ["features"]
            .tolist()
        )

        n_plots = len(feature_sets)
        nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(nrows, ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            squeeze=False,)

        axes = axes.ravel()
        for ax, feature_set in zip(axes, feature_sets):

            subset = k_distance_results[k_distance_results["features"] == feature_set]

            for min_samples, group in subset.groupby("min_samples"):

                ax.plot(group["rank"], group["k_distance"],
                        label=f"min_samples={min_samples}")

            ax.set_title(f"Features: {', '.join(feature_set)}")

            ax.set_xlim(left=xlims_per_plot[0], right=xlims_per_plot[1])
            ax.set_ylim(bottom=ylims_per_plot[0], top=ylims_per_plot[1])
            ax.set_xlabel("Points sorted by k-distance")
            ax.set_ylabel("k-distance")
            ax.grid()
            ax.legend(fontsize=8)

        # Hide unused axes.
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        fig.tight_layout()

        return fig, axes

    @staticmethod
    def evaluate_kmeans(df: pd.DataFrame, features: list[str],
                    k_values: list[int], n_runs: int = 10) -> pd.DataFrame:
        """
        Evaluate KMeans for several values of k.

        Each k is executed `n_runs` times with different random seeds.

        Returns
        -------
        pd.DataFrame
            One row per (k, run), containing:

            - features
            - n_features
            - k
            - run
            - silhouette
            - calinski_harabasz
            - davies_bouldin
            - inertia
        """

        X = df[features].to_numpy()

        results = []

        for k in k_values:

            for run in range(n_runs):

                model = KMeans(n_clusters=k, random_state=run, n_init="auto")

                labels = model.fit_predict(X)

                results.append({
                    "features": tuple(features), "n_features": len(features),
                    "k": k, "run": run,
                    "silhouette": silhouette_score(X, labels),
                    "calinski_harabasz": calinski_harabasz_score(X, labels),
                    "davies_bouldin": davies_bouldin_score(X, labels),
                    "inertia": model.inertia_,
                })

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_minibatch_kmeans(df: pd.DataFrame, features: list[str],
        k_values: list[int], n_runs: int = 10,
        max_iter: int = 100, batch_size: int = 1024) -> pd.DataFrame:
        """
        Evaluate MiniBatchKMeans for several values of k.

        Each k is executed `n_runs` times with different random seeds.

        Returns
        -------
        pd.DataFrame
            One row per (k, run), containing:

            - features
            - n_features
            - k
            - run
            - silhouette
            - calinski_harabasz
            - davies_bouldin
            - inertia
        """

        X = df[features].to_numpy()

        results = []

        for k in k_values:

            for run in range(n_runs):

                model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size,
                    max_iter=max_iter, random_state=run, n_init="auto")
                labels = model.fit_predict(X)

                results.append({
                    "features": tuple(features), "n_features": len(features),
                    "k": k, "run": run,
                    "silhouette": silhouette_score(X, labels),
                    "calinski_harabasz": calinski_harabasz_score(X, labels),
                    "davies_bouldin": davies_bouldin_score(X, labels),
                    "inertia": model.inertia_,
                })

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_dbscan(df: pd.DataFrame, features: list[str],
                    eps_values, min_samples_values,
                    metrics=("euclidean",), p_values=(2,), n_jobs: int = None) \
                        -> pd.DataFrame:
        """
        Evaluate DBSCAN for several values of eps, min samples, p. values, and metrics.

        Returns
        -------
        pd.DataFrame
            One row per (eps, min. samples, metric, p), containing:

            - features
            - n_features
            - eps
            - min_samples
            - metric
            - p
            - silhouette
            - n_noise
            - n_clusters
            - noise_ratio
            - calinski_harabasz
            - davies_bouldin
            - largest and smallest cluster ratios
        """

        X = df[features].to_numpy()

        results = []

        for metric in metrics:
            for eps in eps_values:
                for min_samples in min_samples_values:
                    for p in p_values:

                        model = DBSCAN(eps=eps, min_samples=min_samples,
                                       metric=metric, p=p, n_jobs=n_jobs)
                        labels = model.fit_predict(X)

                        n_noise = np.sum(labels == -1)
                        noise_ratio = n_noise / len(labels)

                        # Ignore DBSCAN noise for silhouette.
                        valid = labels != -1

                        X_valid = X[valid]
                        labels_valid = labels[valid]

                        cluster_ids = np.unique(labels_valid)
                        n_clusters = len(cluster_ids)

                        if n_clusters >= 2:

                            clustered_ratio = valid.mean()

                            cluster_sizes = np.bincount(labels[valid])

                            largest_cluster_size = cluster_sizes.max()
                            largest_cluster_ratio = largest_cluster_size / len(labels)

                            smallest_cluster_size = cluster_sizes.min()
                            smallest_cluster_ratio = smallest_cluster_size / len(labels)

                            silhouette = silhouette_score(X_valid, labels_valid)
                            calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
                            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)

                        else:

                            clustered_ratio = 0.0

                            largest_cluster_size = 0
                            largest_cluster_ratio = 0.0

                            smallest_cluster_size = 0
                            smallest_cluster_ratio = 0.0

                            silhouette = np.nan
                            calinski_harabasz = np.nan
                            davies_bouldin = np.nan

                        results.append({

                            "features": tuple(features),
                            "n_features": len(features),

                            "eps": eps,
                            "min_samples": min_samples,
                            "metric": metric,
                            "p": p,

                            "n_clusters": n_clusters,

                            "n_noise": n_noise,
                            "noise_ratio": noise_ratio,
                            "clustered_ratio": clustered_ratio,

                            "largest_cluster_size": largest_cluster_size,
                            "largest_cluster_ratio": largest_cluster_ratio,

                            "smallest_cluster_size": smallest_cluster_size,
                            "smallest_cluster_ratio": smallest_cluster_ratio,

                            "silhouette": silhouette,
                            "calinski_harabasz": calinski_harabasz,
                            "davies_bouldin": davies_bouldin
                        })

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_hdbscan(df: pd.DataFrame, features: list[str],
        min_cluster_size_values, min_samples_values,
        metrics=("euclidean",),
        cluster_selection_methods=("eom",),
        cluster_selection_epsilons=(0.0,), n_jobs: int = None) -> pd.DataFrame:
        """
        Evaluate HDBSCAN for several values of min cluster size, min samples, etc.

        Returns
        -------
        pd.DataFrame
            One row per (metric, min cluster size, min samples, etc), containing:

            - features
            - n_features
            - metric
            - min_cluster_size
            - cluster_selection_method
            - cluster_selection_epsilon
            - silhouette
            - n_noise
            - n_clusters
            - noise_ratio
            - calinski_harabasz
            - davies_bouldin
            - largest and smallest cluster ratios
            - mean cluster probabilty
        """
        X = df[features].to_numpy()

        results = []

        for metric in metrics:
            for min_cluster_size in min_cluster_size_values:
                for min_samples in min_samples_values:
                    for cluster_selection_method in cluster_selection_methods:
                        for cluster_selection_epsilon in cluster_selection_epsilons:

                            model = HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples, metric=metric,
                                cluster_selection_method=cluster_selection_method,
                                cluster_selection_epsilon=cluster_selection_epsilon, n_jobs=n_jobs)
                            labels = model.fit_predict(X)
                            
                            invalid_labels = {-1, -2, -3}

                            noise_mask = np.isin(labels, list(invalid_labels))
                            n_noise = np.sum(noise_mask)
                            noise_ratio = n_noise / len(labels)

                            valid = ~noise_mask

                            X_valid = X[valid]
                            labels_valid = labels[valid]

                            cluster_ids, cluster_sizes = np.unique(labels_valid, return_counts=True)
                            n_clusters = len(cluster_ids)

                            if n_clusters > 2:

                                largest_cluster_size = cluster_sizes.max()
                                largest_cluster_ratio = largest_cluster_size / len(labels)

                                smallest_cluster_size = cluster_sizes.min()
                                smallest_cluster_ratio = smallest_cluster_size / len(labels)

                                median_cluster_size = np.median(cluster_sizes)

                                silhouette = silhouette_score(X_valid, labels_valid)
                                calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
                                davies_bouldin = davies_bouldin_score(X_valid, labels_valid)

                            else:

                                largest_cluster_size = 0
                                largest_cluster_ratio = 0.0

                                smallest_cluster_size = 0
                                smallest_cluster_ratio = 0.0

                                median_cluster_size = 0
                                
                                silhouette = np.nan
                                calinski_harabasz = np.nan
                                davies_bouldin = np.nan

                            # HDBSCAN membership probabilities
                            probabilities = model.probabilities_

                            mean_probability = np.mean(probabilities[valid]) if np.any(valid) else np.nan
                            median_probability = np.median(probabilities[valid]) if np.any(valid) else np.nan

                            results.append({

                                "features": tuple(features),
                                "n_features": len(features),

                                "metric": metric,
                                "min_cluster_size": min_cluster_size,
                                "min_samples": min_samples,
                                "cluster_selection_method": cluster_selection_method,
                                "cluster_selection_epsilon": cluster_selection_epsilon,

                                "n_clusters": n_clusters,

                                "n_noise": n_noise,
                                "noise_ratio": noise_ratio,
                                "clustered_ratio": 1 - noise_ratio,

                                "largest_cluster_size": largest_cluster_size,
                                "largest_cluster_ratio": largest_cluster_ratio,

                                "smallest_cluster_size": smallest_cluster_size,
                                "smallest_cluster_ratio": smallest_cluster_ratio,

                                "median_cluster_size": median_cluster_size,

                                "silhouette": silhouette,
                                "calinski_harabasz": calinski_harabasz,
                                "davies_bouldin": davies_bouldin,
                                
                                "mean_probability": mean_probability,
                                "median_probability": median_probability,
                            })

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_agglomerative_feature_combinations(
        df: pd.DataFrame, feature_columns: list[str] | None = None,
        min_features: int = 1, max_features: int | None = None,
        k_values: list[int] = (2, 3, 4),
        linkages: list[str] = ("ward", "complete", "average", "single")) \
            -> pd.DataFrame:
        """
        Evaluate Agglomerative Clustering for every combination of features.

        Parameters
        ----------
        df:
            Input dataframe.

        feature_columns:
            Columns to consider. If None, all dataframe columns are used.

        min_features:
            Minimum number of features in a combination.

        max_features:
            Maximum number of features in a combination.
            If None, all features are considered.

        k_values:
            Number of clusters to test.

        linkages:
            Linkage methods to evaluate.

        Returns
        -------
        pd.DataFrame
            One row per feature-combination / k / linkage.
        """

        if feature_columns is None:
            feature_columns = df.columns.tolist()

        if max_features is None:
            max_features = len(feature_columns)

        if min_features < 1:
            raise ValueError("min_features must be >= 1")

        if max_features > len(feature_columns):
            raise ValueError(
                "max_features cannot exceed the number of features"
            )

        results = []

        for n_features in range(min_features, max_features + 1):

            for combination in combinations(feature_columns, r=n_features):

                features = list(combination)

                result = Clustering.evaluate_agglomerative_clustering(
                    df=df, features=features, k_values=k_values, linkages=linkages)

                results.append(result)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def evaluate_agglomerative_clustering(df: pd.DataFrame, features: list[str],
        k_values: list[int], linkages: list[str] = ("ward", "complete", "average", "single")) \
            -> pd.DataFrame:
        """
        Evaluate Agglomerative Clustering for several values of k
        and linkage methods.

        Returns
        -------
        pd.DataFrame
            One row per (k, linkage), containing:

            - features
            - n_features
            - k
            - linkage
            - silhouette
            - calinski_harabasz
            - davies_bouldin
        """

        X = df[features].to_numpy()

        results = []

        for k in k_values:
            for linkage in linkages:

                # Ward linkage requires Euclidean distance.
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

                labels = model.fit_predict(X)

                results.append({
                    "features": tuple(features), "n_features": len(features),
                    "k": model.n_clusters_, "linkage": linkage,
                    "silhouette": silhouette_score(X, labels),
                    "calinski_harabasz": calinski_harabasz_score(X, labels),
                    "davies_bouldin": davies_bouldin_score(X, labels),
                })

        return pd.DataFrame(results)

    @staticmethod
    def run_agglomerative_clustering(df: pd.DataFrame, features: list[str],
        k: int, linkage: str = "ward") -> tuple[pd.DataFrame, AgglomerativeClustering, float]:
        """
        Run Agglomerative Clustering for a single parameter combination.

        Parameters
        ----------
        df:
            Input dataframe.

        features:
            Features used for clustering.

        k:
            Number of clusters.

        linkage:
            Linkage criterion to use. Supported values are:
            "ward", "complete", "average", or "single".

        Returns
        -------
        clustered_df:
            Original dataframe with an additional `cluster` column.

        model:
            Fitted AgglomerativeClustering model.

        score:
            Silhouette score of the clustering.
        """

        X = df[features].to_numpy()

        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        result = df[features].copy()
        result["cluster"] = labels

        cluster_sizes = result["cluster"].value_counts().sort_values()
        cluster_mapping = { old_id: new_id
            for new_id, old_id in enumerate(cluster_sizes.index)}
        result["cluster"] = result["cluster"].map(cluster_mapping)

        return result, model, score

    @staticmethod
    def map_vehicles_to_cell_clusters(trace: pd.DataFrame, clustered_celldf: pd.DataFrame) \
        -> pd.DataFrame:
        """
        Assign each vehicle to the cell cluster where it spent the most time.

        Parameters
        ----------
        trace:
            Original trace containing:
                vehicle_id, x, y, cell_dwell_time

        clustered_celldf:
            Cell dataframe indexed by "x:y" and containing a `cluster`
            column.

        Returns
        -------
        pd.DataFrame
            Indexed by vehicle_id with columns:
                - cluster
                - certainty

            `certainty` is the fraction of the vehicle's total dwell time
            spent in its assigned cluster.
        """

        # Map each trace record to its cell cluster
        cell_to_cluster = clustered_celldf["cluster"]

        trace = trace.copy()

        cell_ids = trace["x"].astype(str) + ":" + trace["y"].astype(str)

        trace["cluster"] = cell_ids.map(cell_to_cluster)

        if trace["cluster"].isna().any():
            raise ValueError("Some trace cells do not exist in clustered_celldf.")

        # Total dwell time per vehicle and cluster
        dwell_by_cluster = trace.groupby(["vehicle_id", "cluster"])["cell_dwell_time"].sum()

        # Cluster with the largest dwell time for each vehicle
        vehicle_clusters = dwell_by_cluster.groupby(level="vehicle_id").idxmax()

        # Extract vehicle_id and selected cluster
        result = pd.DataFrame(vehicle_clusters.tolist(), columns=["vehicle_id", "cluster"]).set_index("vehicle_id")

        # Total dwell time per vehicle
        total_dwell = trace.groupby("vehicle_id")["cell_dwell_time"].sum()

        # Dwell time in the selected cluster
        selected_dwell = pd.Series({vehicle_id: dwell_by_cluster.loc[ (vehicle_id, cluster) ]
                                    for vehicle_id, cluster in result["cluster"].items()})

        # Fraction of time spent in selected cluster
        result["certainty"] = selected_dwell / total_dwell

        return result

    @staticmethod
    def plot_k_distance_graph(X, min_samples_values, metric="euclidean", p=2, figsize: tuple[float, float]=(8, 5), ax=None):

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        for min_samples in min_samples_values:

            nn = NearestNeighbors(
                n_neighbors=min_samples,
                metric=metric,
                p=p,
                n_jobs=8
            )
            nn.fit(X)

            distances, _ = nn.kneighbors(X)

            k_distances = distances[:, min_samples - 1]

            ax.plot(np.sort(k_distances)[::-1], label=f"min_samples={min_samples}")

        ax.set_xlabel("Points sorted by k-distance")
        ax.set_ylabel("k-distance")
        ax.set_title("DBSCAN k-distance plots")
        ax.grid()
        ax.legend()

        return ax