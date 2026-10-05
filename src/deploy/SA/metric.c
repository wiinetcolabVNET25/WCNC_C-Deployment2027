#include "metric.h"


void metric_calculate_vehicle_metrics(
    const struct_trace *trace,
    const struct_grid *grid,
    struct_vehicle_metrics output_metrics[])
{
    for (int v = 0; v < trace->number_of_vehicles; v++) {

        const struct_vehicle *vehicle = &trace->vehicles[v];

        double N = 0.0;
        double Tn = 0.0;
        double Tc = -1.0;

        double current_no_rsu = 0.0;

        for (int i = vehicle->trace_start; i <= vehicle->trace_end; i++) {

            const struct_trace_record *record = &trace->records[i];
            if (grid_has_rsu(grid, record->grid_x_pos, record->grid_y_pos)) {

                /*
                 * RSU contact.
                 */
                N += 1.0;

                /*
                 * Finish current no-RSU interval.
                 */
                if (current_no_rsu > Tn)
                    Tn = current_no_rsu;

                current_no_rsu = 0.0;

                /*
                 * Minimum dwell time at an RSU.
                 */
                if (Tc < 0.0 || record->cell_dwell_time < Tc) {
                    Tc = record->cell_dwell_time;
                }

            } else {

                /*
                 * No RSU.
                 */
                current_no_rsu += record->cell_dwell_time;
            }
        }


        /*
         * Handle final no-RSU interval.
         */
        if (current_no_rsu > Tn)
            Tn = current_no_rsu;


        /*
         * Vehicle never visited an RSU.
         */
        if (Tc < 0.0)
            Tc = 0.0;


        output_metrics[v].N = N;
        output_metrics[v].Tn = Tn;
        output_metrics[v].Tc = Tc;
    }
}


void metric_calculate_cluster_metrics(
    const struct_vehicle_metrics vehicle_metrics[],
    const int vehicle_clusters[],
    int number_of_vehicles,
    int number_of_clusters,
    struct_cluster_metrics output_metrics[]
)
{
    for (int k = 0; k < number_of_clusters; k++) {

        output_metrics[k].mean_N = 0.0;
        output_metrics[k].mean_Tn = 0.0;
        output_metrics[k].mean_Tc = 0.0;
        output_metrics[k].number_of_vehicles = 0;
    }


    for (int v = 0; v < number_of_vehicles; v++) {

        int cluster = vehicle_clusters[v];
        if (cluster < 0 || cluster >= number_of_clusters) {
            continue;
        }

        output_metrics[cluster].mean_N += vehicle_metrics[v].N;
        output_metrics[cluster].mean_Tn += vehicle_metrics[v].Tn;
        output_metrics[cluster].mean_Tc += vehicle_metrics[v].Tc;
        output_metrics[cluster].number_of_vehicles++;
    }

    for (int k = 0; k < number_of_clusters; k++) {

        int n = output_metrics[k].number_of_vehicles;
        if (n > 0) {

            output_metrics[k].mean_N /= n;
            output_metrics[k].mean_Tn /= n;
            output_metrics[k].mean_Tc /= n;
        }
    }
}


void metric_calculate_satisfaction(
    const struct_cluster_metrics cluster_metrics[],
    const double N_required[],
    const double Tn_required[],
    int number_of_clusters,
    double alpha,
    struct_cluster_satisfaction output_satisfaction[]
)
{
    for (int k = 0; k < number_of_clusters; k++) {

        double N_sat = cluster_metrics[k].mean_N / N_required[k];
        if (N_sat > 1.0)
            N_sat = 1.0;

        double Tn_sat;
        if (cluster_metrics[k].mean_Tn <= 0.0) {

            Tn_sat = 1.0;

        } else {

            Tn_sat = Tn_required[k] / cluster_metrics[k].mean_Tn;
            if (Tn_sat > 1.0)
                Tn_sat = 1.0;
        }

        output_satisfaction[k].N_satisfaction = N_sat;
        output_satisfaction[k].Tn_satisfaction = Tn_sat;
        output_satisfaction[k].satisfaction = alpha * N_sat + (1.0 - alpha) * Tn_sat;
    }
}