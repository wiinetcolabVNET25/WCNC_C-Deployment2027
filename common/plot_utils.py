import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

class Plot:

    @staticmethod
    def plot_cell_histogram(trace: pd.DataFrame, use_dwell_time_weight: bool = False,
                            bins: int | tuple[int, int] = 100,
                            figsize: tuple[float, float] = (8, 8), ax=None):
        """
        Plot a 2D histogram of vehicle positions.

        Parameters
        ----------
        trace:
            DataFrame containing `x` and `y` columns.
        bins:
            Number of bins, or (x_bins, y_bins).
        figsize:
            Figure size when `ax` is not provided.
        ax:
            Optional matplotlib Axes.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.hist2d(trace["x"], trace["y"], bins=bins, weights=trace["cell_dwell_time"] if use_dwell_time_weight else None)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        return ax

    @staticmethod
    def plot_cell_feature_2d_histogram(cell_df: pd.DataFrame, feature: str,
                                       bins: int | tuple[int, int] = 100,
                                       figsize: tuple[float, float] = (8, 8), ax=None):
        """
        Plot a 2D histogram of a cell feature

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        x = cell_df.index.str.split(":").str[0].astype(int)
        y = cell_df.index.str.split(":").str[1].astype(int)

        ax.hist2d(x, y, bins=bins, weights=cell_df[feature])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        return ax

    @staticmethod
    def plot_vehicle_feature_2d_histogram( vehicle_df: pd.DataFrame, trace: pd.DataFrame,
        feature: str, trajectory: bool = True, weighted_dwell_time: bool = False,
        alpha: float = 0.7, bins: int | tuple[int, int] = 100, figsize: tuple[float, float] = (8, 8), ax=None):
        """
        Plot a 2D histogram of a vehicle feature over the zone.

        Parameters
        ----------
        vehicle_df:
            DataFrame indexed by vehicle_id containing vehicle features.

        trace:
            Vehicle mobility trace containing:
                vehicle_id, x, y, cell_dwell_time

        feature:
            Vehicle feature to plot.

        trajectory:
            If True, each vehicle contributes its feature value to every
            cell visited in its trajectory.

            If False, each vehicle contributes its feature value only
            at its initial position.

        weighted_dwell_time:
            If True, multiply each feature contribution by the
            cell dwell time.

        alpha:
            Histogram transparency.

        bins:
            Number of histogram bins, or (xbins, ybins).

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if trajectory:

            data = trace[["vehicle_id", "x", "y", "cell_dwell_time"]].join(
                vehicle_df[[feature]], on="vehicle_id", how="inner")

        else:

            positions = (
                trace
                .sort_values(["vehicle_id", "time_instant"])
                .groupby("vehicle_id", sort=False)
                [["x", "y"]]
                .first()
            )

            data = vehicle_df[[feature]].join(positions, how="inner")

            data["cell_dwell_time"] = 1

        data = data.dropna(subset=["x", "y", feature, "cell_dwell_time"])

        if weighted_dwell_time:
            weights = (data[feature] * data["cell_dwell_time"])
        else:
            weights = data[feature]

        ax.hist2d(data["x"], data["y"], bins=bins, weights=weights, alpha=alpha)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        weighting = (" (time weighted)" if weighted_dwell_time else "")
        location = ("traj." if trajectory else "initial pos")
        ax.set_title(f"{feature}\nby vehicle {location}{weighting}")

        return ax

    @staticmethod
    def plot_cell_clusters(
        clusters_dict, marker_style="o", marker_size = 20,
        figsize: tuple[float, float] = (8, 8), ax=None, colors=None):
        """
        Plot clusters of cells.

        Sorts clusters by number of samples in ascending order.

        Parameters
        ----------
        clusters_dict:
            Dictionary mapping cluster IDs to lists of `(x, y)` cell coordinates.

        figsize:
            Figure size when `ax` is not provided.
        ax:
            Optional matplotlib Axes.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        clusters_dict = dict(sorted(clusters_dict.items(), key=lambda item: len(item[1])))
        for cluster_id, cluster in clusters_dict.items():

            if not cluster:
                continue

            x, y = zip(*cluster)

            color = None if colors is None else colors[cluster_id]

            ax.scatter(x, y, marker=marker_style, s=marker_size, color=color, 
                       label=f"Cluster {cluster_id} ({len(cluster)})")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        ax.legend()

        return ax

    @staticmethod
    def plot_clusters_silhouette_scores(silhouette_by_cluster: dict[int, list[float]],
                                        figsize: tuple[float, float] = (10, 7), cluster_gap: int = 10, ax=None):
        """
        Plot per-sample silhouette scores grouped by cluster.

        Parameters
        ----------
        silhouette_by_cluster:
            Dictionary mapping cluster IDs to sorted silhouette scores.
            
        figsize:
            Figure size.

        ax:
            Optional matplotlib Axes.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Remove empty clusters
        silhouette_by_cluster = {
            cluster_id: scores
            for cluster_id, scores in silhouette_by_cluster.items()
            if len(scores) > 0
        }

        if not silhouette_by_cluster:
            return ax

        sorted_cluster_ids = dict(sorted(silhouette_by_cluster.items(), key=lambda item: len(item[1]), reverse=True)).keys()

        all_scores = np.concatenate(
            [np.asarray(scores) for scores in silhouette_by_cluster.values()
        ])

        silhouette_avg = np.mean(all_scores)

        n_samples = len(all_scores)
        n_clusters = len(silhouette_by_cluster)

        ax.set_xlim([-0.1, 1.0])
        ax.set_ylim([0, n_samples + (n_clusters + 1) * cluster_gap])

        y_lower = cluster_gap

        cmap = plt.get_cmap("nipy_spectral")
        for i, cluster_id in enumerate(sorted_cluster_ids):

            scores = silhouette_by_cluster[cluster_id]

            size_cluster = len(scores)

            y_upper = y_lower + size_cluster

            color = cmap(float(i) / max(n_clusters, 1))

            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, scores,
                facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))

            y_lower = y_upper + cluster_gap

        # Average Silhouette
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)

        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        return ax

    @staticmethod
    def plot_cluster_feature_boxplots(
        df: pd.DataFrame, features: list[str],
        figsize: tuple[float, float] | None = None):
        
        n_features = len(features)

        if figsize is None:
            figsize = (5 * n_features, 5)

        fig, axes = plt.subplots(1, n_features, figsize=figsize, squeeze=False)

        axes = axes[0]

        for ax, feature in zip(axes, features):
            sns.boxplot(data=df, x="cluster", y=feature, ax=ax)

            ax.set_title(feature)
            ax.set_xlabel("Cluster")
            ax.set_ylabel(feature)

        plt.tight_layout()

        return fig, axes

    @staticmethod
    def plot_vehicle_trajectories(trace: pd.DataFrame, max_vehicles: int | None = None,
                                alpha: float = 0.25, linewidth: float = 0.8, color="red",
                                figsize: tuple[float, float] = (10, 10), ax=None):
        """
        Plot vehicle trajectories.

        Parameters
        ----------
        trace:
            DataFrame containing `vehicle_id`, `x`, and `y`.

        max_vehicles:
            Maximum number of vehicles to plot. If None, plot all.

        alpha:
            Line transparency.

        linewidth:
            Trajectory line width.

        figsize:
            Figure size when `ax` is not provided.

        ax:
            Optional matplotlib Acommon.deploy_utils import (Trace, Metrics, RSUs)
    from common.plxes.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        vehicle_groups = trace.groupby("vehicle_id", sort=False)

        for i, (_, vehicle) in enumerate(vehicle_groups):

            if max_vehicles is not None and i >= max_vehicles:
                break

            ax.plot(vehicle["x"], vehicle["y"], alpha=alpha, linewidth=linewidth, color=color)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        return ax

    @staticmethod
    def plot_cells_and_rsus(trace: pd.DataFrame, rsus: list[tuple],
                            figsize: tuple[float, float] = (10, 10), ax=None):
        """
        Plot cells visited by vehicles and highlight RSU cells.

        Parameters
        ----------
        trace:
            DataFrame containing `x` and `y` columns.

        rsus:
            List of `(x, y)` RSU positions.

        figsize:
            Figure size when `ax` is not provided.

        ax:
            Optional matplotlib Axes.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Unique cells visited by vehicles.
        cells = trace[["x", "y"]].drop_duplicates()

        for x, y in cells.itertuples(index=False):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=0.5, edgecolor="gray", facecolor="lightgray"))

        # RSUs are drawn after the cells so they appear on top.
        for x, y in rsus:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1.5, edgecolor="red", facecolor="red"))

        ax.relim()
        ax.autoscale_view()

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        return ax

    @staticmethod
    def plot_trajectories_and_rsus(trace: pd.DataFrame, rsus: list[tuple],
                                   max_vehicles: int | None = None,
                                   trajectory_alpha: float = 0.15,
                                   trajectory_linewidth: float = 0.7,
                                   rsus_color="black", trajectory_color="blue",
                                   figsize: tuple[float, float] = (10, 10), ax=None):
        """
        Plot vehicle trajectories and highlight RSUs.

        Trajectories are deliberately drawn with low opacity so that
        RSUs remain clearly visible even in dense traces.

        Parameters
        ----------
        trace:
            DataFrame containing `vehicle_id`, `x`, and `y`.

        rsus:
            List of `(x, y)` RSU positions.

        max_vehicles:
            Maximum number of trajectories to draw. None means all.

        trajectory_alpha:
            Transparency of trajectories.

        trajectory_linewidth:
            Width of trajectory lines.

        figsize:
            Figure size when `ax` is not provided.

        ax:
            Optional matplotlib Axes.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Draw trajectories first.
        for i, (_, vehicle) in enumerate( trace.groupby("vehicle_id", sort=False)):
            if max_vehicles is not None and i >= max_vehicles:
                break

            ax.plot(vehicle["x"], vehicle["y"], alpha=trajectory_alpha, linewidth=trajectory_linewidth, color=trajectory_color)

        # Draw RSUs on top of trajectories.
        if rsus:
            rsu_x, rsu_y = zip(*rsus)

            ax.scatter(rsu_x, rsu_y, s=10, marker="s", zorder=10, label="RSU", c=rsus_color)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        ax.relim()
        ax.autoscale_view()

        if rsus:
            ax.legend()

        return ax

    @staticmethod
    def plot_trajectories_by_cluster(
        trace: pd.DataFrame,
        clusters: pd.DataFrame,
        max_vehicles_per_cluster: int | None = None,
        alpha: float = 0.35,
        linewidth: float = 0.8,
        figsize: tuple[float, float] = (10, 10),
        ax=None,
    ):
        """
        Plot vehicle trajectories colored according to their cluster.

        Parameters
        ----------
        trace:
            DataFrame containing `vehicle_id`, `x`, and `y`.

        clusters:
            DataFrame containing `vehicle_id` and `cluster`.

        max_vehicles_per_cluster:
            Maximum number of trajectories to draw per cluster.

        alpha:
            Trajectory transparency.

        linewidth:
            Trajectory line width.

        figsize:
            Figure size when `ax` is not provided.

        ax:
            Optional matplotlib Axes.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        trace_clustered = trace.merge(clusters[["vehicle_id", "cluster"]], on="vehicle_id", how="inner")

        cluster_ids = sorted(trace_clustered["cluster"].unique())

        # Discrete colors, one for each cluster.
        cmap = plt.get_cmap("tab10")
        cluster_colors = { cluster: cmap(i % 10) for i, cluster in enumerate(cluster_ids) }

        counts = {}
        labeled_clusters = set()

        for vehicle_id, vehicle in trace_clustered.groupby("vehicle_id", sort=False):

            cluster = vehicle["cluster"].iloc[0]

            if (max_vehicles_per_cluster is not None and counts.get(cluster, 0) >= max_vehicles_per_cluster):
                continue

            counts[cluster] = counts.get(cluster, 0) + 1

            cluster_size = trace_clustered[trace_clustered["cluster"] == cluster]["vehicle_id"].nunique()

            label = (f"Cluster {cluster} ({cluster_size})" if cluster not in labeled_clusters else None)

            labeled_clusters.add(cluster)

            ax.plot(vehicle["x"], vehicle["y"], color=cluster_colors[cluster], alpha=alpha, linewidth=linewidth, label=label)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax.set_title("Vehicle trajectories by cluster")

        if labeled_clusters:
            legend = ax.legend(
                loc="upper right",
                fontsize=10,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                edgecolor="black",
                facecolor="white",
                handlelength=2.0,
                borderpad=0.6,
            )

            legend.get_frame().set_linewidth(1.2)

            for handle in legend.legend_handles:
                handle.set_alpha(1.0)
                handle.set_linewidth(3.0)

        return ax

    @staticmethod
    def plot_joint_density(df: pd.DataFrame, x: str, y: str, bins=30,
                           figsize: tuple[float, float] = (10, 10),
                           density=False,
                           cmin=None,
                           ax=None):
        """
        Plot a 2D histogram representing an estimated
        joint probability density.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        h = ax.hist2d(df[x], df[y], bins=bins, density=density, cmin=cmin)

        plt.colorbar(h[3], ax=ax, label="Joint density")

        ax.set_xlabel(x)
        ax.set_ylabel(y)

        return ax

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, features: list[str],
                                method: str = "pearson",
                                figsize: tuple[float, float] = (10, 10),
                                ax=None):

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        corr = df[features].corr(method=method)

        sns.heatmap(corr, annot=True, cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, square=True, ax=ax)

        return ax