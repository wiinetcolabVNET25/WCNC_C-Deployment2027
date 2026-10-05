import pandas as pd
import numpy as np

class Trace:

    @staticmethod
    def read_trace(file_path) -> pd.DataFrame:
        """
        Automatically scales cell_dwell_time by 10
        """
        dataset = pd.read_csv(filepath_or_buffer=file_path,
                            sep=";", lineterminator="\n", header=None,
                            names=["vehicle_id", "time_instant", "x", "y", "cell_dwell_time", "null"])
        dataset.drop(labels="null", axis=1, inplace=True)
        dataset["cell_dwell_time"] *= 10

        return dataset

    @staticmethod
    def summarize_trace(trace: pd.DataFrame) -> str:
        
        return f"[+] number of vehicles: {trace["vehicle_id"].nunique()}\n" + \
                f"[+] number of lines: {len(trace)}\n" + \
                f"[+] x interval: [{trace["x"].min()}, {trace["x"].max()}]\n" + \
                f"[+] y interval: [{trace["y"].min()}, {trace["y"].max()}]"

class Metrics:
    """
    Assumes vehicle trace contains contiguous IDs
    """

    @staticmethod
    def prepare_trace(trace: pd.DataFrame, rsus: list[tuple], grid_width: int = 100):
        """
        Convert the trace to NumPy arrays and create the RSU mask.

        Returns
        -------
        vehicle_ids : np.ndarray
        dwell       : np.ndarray
        is_rsu      : np.ndarray
        starts      : np.ndarray
        ends        : np.ndarray
        """

        vehicle_ids = trace["vehicle_id"].to_numpy()

        time = trace["time_instant"].to_numpy()

        x = trace["x"].to_numpy()
        y = trace["y"].to_numpy()

        dwell = trace["cell_dwell_time"].to_numpy()

        # Encode coordinates (x, y) as one integer.
        cell_ids = x * grid_width + y

        # Encode RSU positions the same way.
        rsu_cells = np.array(
            [x * grid_width + y for x, y in rsus],
            dtype=cell_ids.dtype,
        )

        # True for every trace row whose cell contains an RSU.
        is_rsu = np.isin(cell_ids, rsu_cells)        

        # Since vehicle IDs are already ordered, vehicle boundaries
        # can be found without groupby().
        starts = np.flatnonzero(
            np.r_[True, vehicle_ids[1:] != vehicle_ids[:-1]]
        )

        ends = np.r_[starts[1:] - 1, len(trace) - 1]

        return vehicle_ids, time, dwell, is_rsu, starts, ends

    @staticmethod
    def get_covered_vehicles_gamma(trace: pd.DataFrame, rsus: list[tuple], tau: int, grid_width: int = 100):
        """
        Returns IDs of vehicles covered by Gamma.

        A vehicle is covered if it never spends more than `tau`
        seconds without encountering an RSU.
        """

        ( vehicle_ids, _, dwell, is_rsu, starts, ends ) = Metrics.prepare_trace(trace, rsus, grid_width)

        covered = []

        for start, end in zip(starts, ends):
            
            time_without_rsu = 0
            for contact, vehicle_dwell in zip(is_rsu[start:end + 1], dwell[start:end + 1]):
                
                if contact:
                
                    time_without_rsu = 0
                
                else:
               
                    time_without_rsu += vehicle_dwell

                    if time_without_rsu > tau:
                        break
            else:

                # The inner loop completed without exceeding tau.
                covered.append(vehicle_ids[start])

        return covered

    @staticmethod
    def get_covered_vehicles_gamma_from_prep_trace(prepared_trace: tuple, tau: int):
        """
        Returns IDs of vehicles covered by Gamma. Uses precomputed trace from **Metrics.prepare_trace**.

        A vehicle is covered if it never spends more than `tau`
        seconds without encountering an RSU.
        """

        ( vehicle_ids, _, dwell, is_rsu, starts, ends ) = prepared_trace

        covered = []

        for start, end in zip(starts, ends):
            
            time_without_rsu = 0
            for contact, vehicle_dwell in zip(is_rsu[start:end + 1], dwell[start:end + 1]):
                
                if contact:
                
                    time_without_rsu = 0
                
                else:
               
                    time_without_rsu += vehicle_dwell

                    if time_without_rsu > tau:
                        break
            else:

                # The inner loop completed without exceeding tau.
                covered.append(vehicle_ids[start])

        return covered

    @staticmethod
    def get_covered_vehicles_baseline(trace: pd.DataFrame, rsus: list[tuple], grid_width: int = 100):
        """
        Returns IDs of vehicles that encounter at least one RSU.
        """

        ( vehicle_ids, _, _, is_rsu, starts, ends ) = Metrics.prepare_trace(trace, rsus, grid_width)

        covered = []

        for start, end in zip(starts, ends):
            
            if np.any(is_rsu[start:end + 1]):
                covered.append(vehicle_ids[start])

        return covered

    @staticmethod
    def get_covered_vehicles_baseline_from_prep_trace(prepared_trace: tuple):
        """
        Returns IDs of vehicles that encounter at least one RSU. Uses precomputed trace from **Metrics.prepare_trace**.
        """

        ( vehicle_ids, _, _, is_rsu, starts, ends ) = prepared_trace

        covered = []

        for start, end in zip(starts, ends):
            
            if np.any(is_rsu[start:end + 1]):
                covered.append(vehicle_ids[start])

        return covered

    @staticmethod
    def get_covered_vehicles_n_deploy(trace: pd.DataFrame, rsus: list[tuple], N: int, tau: int, grid_width: int = 100):
        """
        Returns IDs of vehicles covered by N-Deploy.

        A vehicle is covered if at least N RSU contacts occur within tau seconds from the beginning of its trajectory.
        """

        ( vehicle_ids, time, _, is_rsu, starts, ends ) = Metrics.prepare_trace(trace, rsus, grid_width)

        covered = []

        for start, end in zip(starts, ends):

            start_time = time[start]
            deadline = start_time + tau

            # Because timestamps are ordered, once we reach the
            # deadline we can stop.
            contacts = 0

            for i in range(start, end + 1):

                if time[i] > deadline:
                    break

                if is_rsu[i]:
                    contacts += 1

                    if contacts >= N:
                        covered.append(vehicle_ids[start])
                        break

        return covered
    
    @staticmethod
    def get_covered_vehicles_n_deploy_from_prep_trace(prepared_trace: tuple, N: int, tau: int):
        """
        Returns IDs of vehicles covered by N-Deploy. Uses precomputed trace from **Metrics.prepare_trace**.

        A vehicle is covered if at least N RSU contacts occur within tau seconds from the beginning of its trajectory.
        """

        ( vehicle_ids, time, _, is_rsu, starts, ends ) = prepared_trace

        covered = []

        for start, end in zip(starts, ends):

            start_time = time[start]
            deadline = start_time + tau

            # Because timestamps are ordered, once we reach the
            # deadline we can stop.
            contacts = 0

            for i in range(start, end + 1):

                if time[i] > deadline:
                    break

                if is_rsu[i]:
                    contacts += 1

                    if contacts >= N:
                        covered.append(vehicle_ids[start])
                        break

        return covered

    @staticmethod
    def get_vehicles_N_tau(trace: pd.DataFrame, rsus: list[tuple], grid_width: int = 100) -> pd.DataFrame:
        """
        Calculate N, tau for every vehicle.

        N:
            Number of RSU contacts.

        tau:
            Time until first contact.
        """

        ( vehicle_ids, _, dwell, is_rsu, starts, ends ) = Metrics.prepare_trace(trace, rsus, grid_width)

        result = []
        for start, end in zip(starts, ends):

            vehicle_dwell = dwell[start:end + 1]
            vehicle_rsu = is_rsu[start:end + 1]

            N = 0
            time_no_rsu = 0

            for contact, d in zip(vehicle_rsu, vehicle_dwell):

                if contact:
                    N += 1

                else:
                    if N == 0:
                        time_no_rsu += d

            result.append((vehicle_ids[start], N, time_no_rsu))

        return pd.DataFrame(result, columns=["vehicle_id", "N", "tau"])

    @staticmethod
    def get_vehicles_N_Tn_Tc(trace: pd.DataFrame, rsus: list[tuple], grid_width: int = 100) -> pd.DataFrame:
        """
        Calculate N, Tn and Tc for every vehicle.

        N:
            Number of RSU contacts.

        Tn:
            Maximum accumulated time without an RSU.

        Tc:
            Minimum dwell time at an RSU.
        """

        ( vehicle_ids, _, dwell, is_rsu, starts, ends ) = Metrics.prepare_trace(trace, rsus, grid_width)

        result = []
        for start, end in zip(starts, ends):

            vehicle_dwell = dwell[start:end + 1]
            vehicle_rsu = is_rsu[start:end + 1]

            N = 0
            current_no_rsu = 0
            Tn = 0
            Tc = None

            for contact, d in zip(vehicle_rsu, vehicle_dwell):

                if contact:
                    N += 1

                    if Tc is None:
                        Tc = d
                    else:
                        Tc = min(Tc, d)

                    # End of the current no-RSU interval.
                    Tn = max(Tn, current_no_rsu)
                    current_no_rsu = 0

                else:
                    current_no_rsu += d

            # Account for a no-RSU interval at the end.
            Tn = max(Tn, current_no_rsu)

            if Tc is None:
                Tc = 0

            result.append((vehicle_ids[start], N, Tn, Tc))

        return pd.DataFrame(result, columns=["vehicle_id", "N", "Tn", "Tc"])

    @staticmethod
    def get_vehicles_N_Tn_Tc_from_prep_trace(prepared_trace: tuple) -> pd.DataFrame:
        """
        Calculate N, Tn and Tc for every vehicle. Uses precomputed trace from **Metrics.prepare_trace**.

        N:
            Number of RSU contacts.

        Tn:
            Maximum accumulated time without an RSU.

        Tc:
            Minimum dwell time at an RSU.
        """

        ( vehicle_ids, _, dwell, is_rsu, starts, ends ) = prepared_trace

        result = []
        for start, end in zip(starts, ends):

            vehicle_dwell = dwell[start:end + 1]
            vehicle_rsu = is_rsu[start:end + 1]

            N = 0
            current_no_rsu = 0
            Tn = 0
            Tc = None

            for contact, d in zip(vehicle_rsu, vehicle_dwell):

                if contact:
                    N += 1

                    if Tc is None:
                        Tc = d
                    else:
                        Tc = min(Tc, d)

                    # End of the current no-RSU interval.
                    Tn = max(Tn, current_no_rsu)
                    current_no_rsu = 0

                else:
                    current_no_rsu += d

            # Account for a no-RSU interval at the end.
            Tn = max(Tn, current_no_rsu)

            if Tc is None:
                Tc = 0

            result.append((vehicle_ids[start], N, Tn, Tc))

        return pd.DataFrame(result, columns=["vehicle_id", "N", "Tn", "Tc"])

class RSUs:

    @staticmethod
    def read_deployment(filename: str) -> set[tuple[int, int]]:
        """
        Expected format:

            x;y
        """

        df = pd.read_csv(filename, sep=";", header=None, names=["x", "y"])

        return list(zip(df["x"].astype(int), df["y"].astype(int)))

if __name__ == "__main__":

    # Tests

    rsus = [
        (1, 1),
        (1, 4),
        (2, 2),
        (2, 5),
        (3, 1),
        (3, 4),
        (4, 2),
        (4, 5),
        (5, 1),
        (5, 4),
    ]

    # VEHICLE_ID, TIME, X, Y, DWELL
    # Some cells are not contiguous (not important, should be guaranteed outside the methods)
    trace = pd.DataFrame([
        # VEHICLE 1
        [1,  0, 3, 3, 20],
        [1, 20, 1, 1, 10],    # RSU
        [1, 30, 2, 3, 30],
        [1, 60, 4, 4, 20],
        [1, 80, 1, 4, 15],    # RSU
        [1, 95, 5, 5, 40],
        [1, 135, 2, 2, 20],   # RSU
        [1, 155, 5, 5, 40],

        # VEHICLE 2
        [2,   0, 4, 4, 70],
        [2,  70, 2, 5, 25],    # RSU
        [2,  95, 5, 5, 30],
        [2, 125, 3, 4, 10],    # RSU
        [2, 135, 5, 2, 80],

        # VEHICLE 3
        [3,   0, 5, 5, 40],
        [3,  40, 4, 4, 30],
        [3,  70, 2, 3, 20],

        # VEHICLE 4
        [4, 0, 6, 6, 100],

    ], columns=[ "vehicle_id", "time_instant", "x", "y", "cell_dwell_time"])
    
    print(Metrics.prepare_trace(trace, rsus))

    assert (Metrics.get_covered_vehicles_baseline(trace, rsus) == [1, 2])

    assert (Metrics.get_covered_vehicles_gamma(trace, rsus, 60) == [1])

    assert (Metrics.get_covered_vehicles_gamma(trace, rsus, 80) == [1, 2])

    assert (Metrics.get_covered_vehicles_gamma(trace, rsus, 120) == [1, 2, 3, 4])

    assert (Metrics.get_covered_vehicles_n_deploy(trace, rsus, N=1, tau=60) == [1])
    
    assert (Metrics.get_covered_vehicles_n_deploy(trace, rsus, N=2, tau=60) == [])

    assert (Metrics.get_covered_vehicles_n_deploy(trace, rsus, N=1, tau=120) == [1, 2])

    assert (Metrics.get_covered_vehicles_n_deploy(trace, rsus, N=2, tau=120) == [1])

    assert (Metrics.get_covered_vehicles_n_deploy(trace, rsus, N=2, tau=200) == [1, 2])

    assert (Metrics.get_vehicles_N_Tn_Tc(trace, rsus)
                .equals(
                pd.DataFrame([
                    (1, 3, 50, 10),
                    (2, 2, 80, 10),
                    (3, 0, 90, 0),
                    (4, 0, 100, 0)],
                    columns=["vehicle_id", "N", "Tn", "Tc"])))

    assert (Metrics.get_vehicles_N_tau(trace, rsus)
                .equals(
                pd.DataFrame([
                    (1, 3, 20),
                    (2, 2, 70),
                    (3, 0, 90),
                    (4, 0, 100)],
                    columns=["vehicle_id", "N", "tau"])))

    print("All tests passed.")
