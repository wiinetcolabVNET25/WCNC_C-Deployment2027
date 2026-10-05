import pandas as pd
import numpy as np
import networkx as nx

class Features:

    @staticmethod
    def create_basic_cell_df(trace: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts a cell's feature df from a trace df.

        Expects a trace with columns:
            - vehicle_id
            - x
            - y
            - cell_dwell_time

        ordered by vehicle_id.

        Returns
        -------

        A cell dataframe indexed by x:y with columns:
            - start_c: count of trips starting
            - end_c: count of trips ending
            - avg_dwell_time: average vehicle stay time
            - pop: cell popularity (number of vehicles which crossed the cell)
        """

        trace["x:y"] = trace["x"].astype("str") + ":"+  trace["y"].astype("str")

        cell_df = pd.DataFrame(data={
            "start_c": trace[["vehicle_id", "x:y"]].groupby("vehicle_id").first()["x:y"].value_counts(),
            "end_c": trace[["vehicle_id", "x:y"]].groupby("vehicle_id").last()["x:y"].value_counts(),
            "avg_dwell_time": trace[["cell_dwell_time", "x:y"]].groupby("x:y").mean()["cell_dwell_time"],
            "pop": trace["x:y"].value_counts(),
        }).fillna(0)

        cell_df["start_c"] = cell_df["start_c"].astype(np.int64)
        cell_df["end_c"] = cell_df["end_c"].astype(np.int64)

        trace.drop("x:y", axis=1, inplace=True)

        return cell_df

    @staticmethod
    def create_basic_vehicle_df(trace: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts a vehicles' feature df from a trace df.
        
        Expects a trace with columns:
            - vehicle_id
            - x
            - y
            - cell_dwell_time

        ordered by vehicle_id.

        Returns
        -------

        A vehicle dataframe indexed by vehicle ids with columns:
            - trip_duration
            - trip_length
            - trip_speed
            - radius_of_gyration
            - entropy
        """

        def get_radius_of_gyration(subtrace):
            x_mean = subtrace["x"].mean()
            y_mean = subtrace["y"].mean()
            return np.sqrt(((subtrace["x"] - x_mean)**2 + (subtrace["y"] - y_mean)**2).mean())

        def get_cell_shannon_entropy_with_cell_time(subtrace):
            dwell_by_location = subtrace.groupby(["x", "y"])["cell_dwell_time"].sum()
            cell_frequency = dwell_by_location / dwell_by_location.sum()
            return -np.sum(cell_frequency * np.log2(cell_frequency))

        return pd.DataFrame(data={
            "trip_duration": trace[["vehicle_id", "cell_dwell_time"]].groupby("vehicle_id").sum()["cell_dwell_time"],
            "trip_length": trace[["vehicle_id", "time_instant"]].groupby("vehicle_id").count()["time_instant"],
            "trip_speed": (trace[["vehicle_id", "time_instant"]].groupby("vehicle_id").count()["time_instant"] 
            / trace[["vehicle_id", "cell_dwell_time"]].groupby("vehicle_id").sum()["cell_dwell_time"]),
            "radius_of_gyration": trace.groupby("vehicle_id")[["x", "y", "cell_dwell_time"]].apply(
                get_radius_of_gyration),
            "entropy": trace.groupby("vehicle_id")[["x", "y", "cell_dwell_time"]].apply(
                get_cell_shannon_entropy_with_cell_time),
        })
