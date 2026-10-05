#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "config.h"
#include "trace.h"
#include "grid.h"
#include "metric.h"
#include "sa.h"

#include <sys/stat.h>
#include <sys/types.h>

#define NUMBER_OF_CLUSTERS 3

int read_vehicle_clusters(
    const char *filename, const struct_trace *trace,
    int output_clusters[], char *error_msg);

int write_deployment(const char *filename, const struct_deployment *deployment);

int sa_write_history(const struct_sa_result *result, const char *filename);

void print_usage(char *argv[]);

int main(int argc, char *argv[])
{
    static_assert(NUMBER_OF_CLUSTERS <= MAX_NUMBER_OF_CLUSTERS);

    char error_msg[300];

    if (argc != 17) {

        print_usage(argv);
        return 1;
    }

    /*
     * ========================================================
     * Parse arguments
     * ========================================================
     */

    const char *data_dir = argv[1];
    const char *results_dir = argv[2];

    const char *trace_name = argv[3];
    const char *cluster_name = argv[4];

    char trace_file[300];
    char cluster_file[300];

    sprintf(trace_file, "%s/%s.csv", data_dir, trace_name);
    sprintf(cluster_file, "%s/%s.csv", data_dir, cluster_name);

    int number_of_rsus = atoi(argv[5]);

    double N_required[NUMBER_OF_CLUSTERS];

    N_required[0] = atof(argv[6]);
    N_required[1] = atof(argv[7]);
    N_required[2] = atof(argv[8]);

    double Tn_required[NUMBER_OF_CLUSTERS];

    Tn_required[0] = atof(argv[9]);
    Tn_required[1] = atof(argv[10]);
    Tn_required[2] = atof(argv[11]);

    struct_sa_config config;

    config.number_of_iterations = atoi(argv[12]);
    config.initial_temperature = atof(argv[13]);
    config.cooling_rate = atof(argv[14]);
    config.alpha = atof(argv[15]);

    unsigned int seed = (unsigned int)strtoul(argv[16], NULL, 10);

    mkdir(results_dir, 0755);

    /*
     * Print configuration.
     */

    #if DEBUG == 1
        printf("========================================\n");
        printf("SA configuration\n");
        printf("========================================\n");

        printf("Data dir:               %s\n", data_dir);
        printf("Results dir:            %s\n", results_dir);

        printf("Trace name:             %s\n", trace_name);
        printf("Cluster name:           %s\n", cluster_name);

        printf("Trace file:             %s\n", trace_file);
        printf("Cluster file:           %s\n", cluster_file);

        printf("Number of RSUs:        %d\n", number_of_rsus);

        printf("N required:            %.2f %.2f %.2f\n",
            N_required[0], N_required[1], N_required[2]);

        printf("Tn required:           %.2f %.2f %.2f\n",
            Tn_required[0], Tn_required[1], Tn_required[2]);

        printf("Iterations:            %d\n", config.number_of_iterations);

        printf("Initial temperature:   %.6f\n", config.initial_temperature);

        printf("Cooling rate:          %.6f\n", config.cooling_rate);

        printf("Alpha:                 %.6f\n", config.alpha);

        printf("Seed:                  %u\n", seed);
    #endif

    /* ========================================================
     * 1. Read trace
     * ======================================================== */

    struct_trace trace;
    if (trace_init(&trace, error_msg) != 0)
    {
        printf("TRACE INIT ERROR: %s\n", error_msg);
        return 1;
    }

    if (trace_read(trace_file, 10, &trace, error_msg) != 0)
    {
        printf("TRACE READ ERROR: %s\n", error_msg);
        trace_free(&trace);
        return 1;
    }

    #if DEBUG == 1
        printf("\nTrace size: %d\nNumber of vehicles: %d\n", trace.size, trace.number_of_vehicles);
    #endif

    /* ========================================================
     * 2. Read clusters
     * ======================================================== */

    int vehicle_clusters[MAX_NUMBER_OF_VEHICLES];
    if (read_vehicle_clusters(cluster_file, &trace, vehicle_clusters, error_msg) != 0)
    {
        printf("CLUSTER READING ERROR: %s\n", error_msg);
        trace_free(&trace);
        return 1;
    }

    /* ========================================================
     * 3. Candidate grid
     * ======================================================== */

    struct_grid candidate_grid;
    grid_init(&candidate_grid);

    /*
     * Every cell visited by the trace is a candidate.
     */
    for (int i = 0; i < trace.size; i++) {

        int x = trace.records[i].grid_x_pos;
        int y = trace.records[i].grid_y_pos;

        candidate_grid.rsu[x][y] = 1;
    }

    /*
    * Convert the candidate grid into a flat array.
    * SA will use this array directly.
    */
    struct_cell candidate_cells[MAX_NUMBER_OF_CANDIDATE_CELLS];
    int number_of_candidate_cells = grid_flatten_rsus(&candidate_grid, candidate_cells, MAX_NUMBER_OF_CANDIDATE_CELLS);
    if (number_of_candidate_cells < 0)
    {
        printf("ERROR: Too many candidate cells.\n");
        trace_free(&trace);
        return 1;
    }

    if (number_of_rsus > number_of_candidate_cells)
    {
        printf("ERROR: Requested %d RSUs, but only %d candidate cells exist.\n",
            number_of_rsus, number_of_candidate_cells);
        trace_free(&trace);
        return 1;
    }

    /* ========================================================
     * 4. SA
     * ======================================================== */

    struct_sa_result result;
    if (sa_optimize(
        &trace,
        vehicle_clusters,
        candidate_cells,
        number_of_candidate_cells,
        number_of_rsus,
        N_required,
        Tn_required,
        config,
        seed,
        &result,
        error_msg) != 0)
    {
        printf("OPTIMIZATION ERROR: %s\n", error_msg);
        trace_free(&trace);
        sa_result_free(&result);
        return 1;
    }

    /* ========================================================
     * 5. Print result
     * ======================================================== */

    #if DEBUG == 1
        printf("\nBest objective: %.6f\n", result.best_objective);
        printf("Best deployment contains %d RSUs.\n", result.best_solution.size);
    #endif

    /* ========================================================
     * 6. Write best deployment
     * ======================================================== */
    char deployment_filename[500];
    sprintf(deployment_filename,
        "%s/%s_%s_dep_%d_%.0f_%.0f_%.0f_%.0f_%.0f_%.0f_%d_%.3f_%.3f_%.2f_%u.csv",
        results_dir, trace_name, cluster_name,
        number_of_rsus,
        N_required[0], N_required[1], N_required[2],
        Tn_required[0], Tn_required[1], Tn_required[2],
        config.number_of_iterations, config.initial_temperature, config.cooling_rate,
        config.alpha,
        seed
    );

    if (write_deployment(deployment_filename, &result.best_solution) != 0)
    {
        trace_free(&trace);
        sa_result_free(&result);
        return 1;
    }

    /* ========================================================
     * 7. Write SA history
     * ======================================================== */
    char history_filename[500]; 
    sprintf(
        history_filename,
        "%s/%s_%s_his_%d_%.0f_%.0f_%.0f_%.0f_%.0f_%.0f_%d_%.3f_%.3f_%.2f_%u.csv",
        results_dir, trace_name, cluster_name,
        number_of_rsus,
        N_required[0], N_required[1], N_required[2],
        Tn_required[0], Tn_required[1], Tn_required[2],
        config.number_of_iterations, config.initial_temperature, config.cooling_rate,
        config.alpha,
        seed
    );

    if (sa_write_history(&result, history_filename) != 0)
    {
        trace_free(&trace);
        sa_result_free(&result);
        return 1;
    }

    /* ========================================================
     * 8. Cleanup
     * ======================================================== */

    trace_free(&trace);
    sa_result_free(&result);

    return 0;
}


int write_deployment(const char *filename, const struct_deployment *deployment)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Could not open deployment output file \"%s\".\n", filename);
        return 1;
    }

    for (int i = 0; i < deployment->size; i++)
    {
        fprintf(file, "%d;%d\n", deployment->rsus[i].x, deployment->rsus[i].y);
    }

    fclose(file);

    return 0;
}

int read_vehicle_clusters(
    const char *filename, const struct_trace *trace,
    int output_clusters[], char *output_error_msg)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {

        sprintf(output_error_msg, "can't open cluster file \"%s\"", filename);
        return 1;
    }

    /*
     * Mark every vehicle as unassigned.
     */
    for (int i = 0; i < trace->number_of_vehicles; i++) {
        output_clusters[i] = -1;
    }

    char line[201];
    while (fgets(line, sizeof(line), file) != NULL) {

        int vehicle_id;
        int cluster;

        /*
            Format: id;cluster\n
        */
        if (sscanf(line, "%d;%d", &vehicle_id, &cluster) != 2) {
            continue;
        }

        int vehicle_index = trace_find_vehicle(trace, vehicle_id);
        if (vehicle_index < 0) {

            sprintf(output_error_msg,
            "vehicle ID %d from cluster file does not exist in trace", vehicle_id);
            fclose(file);
            return 1;
        }

        output_clusters[vehicle_index] = cluster;
    }

    fclose(file);

    /*
     * Verify that every vehicle received a cluster.
     */
    for (int i = 0; i < trace->number_of_vehicles; i++) {

        if (output_clusters[i] < 0) {

            sprintf(output_error_msg,
            "vehicle ID %d has no cluster assignment",
            trace->vehicles[i].vehicle_id);
            return 1;
        }
    }

    return 0;
}

int sa_write_history(const struct_sa_result *result, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Could not open SA history output file \"%s\".\n", filename);
        return 1;
    }

    fprintf(file, "iteration;temperature;objective;best_objective");
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        fprintf(file, ";curr_c%d_s", k);
    }
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        fprintf(file, ";best_c%d_s", k);
    }
    fprintf(file, "\n");

    for (int i = 0; i < result->number_of_iterations + 1; i++)
    {
        fprintf(file, "%d;%.6f;%.6f;%.6f",
            i,
            result->temperature_history[i],
            result->objective_history[i],
            result->best_objective_history[i]
        );

        for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
        {
            fprintf(file, ";%.6f", result->current_satisfaction_history[k][i]);
        }

        for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
        {
            fprintf(file, ";%.6f", result->best_satisfaction_history[k][i]);
        }

        fprintf(file, "\n");
    }

    fclose(file);

    return 0;
}

void print_usage(char *argv[])
{
    printf(
        "Usage:\n"
        "  %s <data_dir> <results_dir> "
        "<trace_file_name (no ext, assumed .csv)> <cluster_file_name (no ext, assumed .csv)> "
        "<number_of_rsus> "
        "<N0> <N1> <N2> "
        "<Tn0> <Tn1> <Tn2> "
        "<iterations> "
        "<initial_temperature> "
        "<cooling_rate> "
        "<alpha> "
        "<seed>\n\n",

        argv[0]
    );

    printf("Example:\n");

    printf(
        "  %s data res 6_to_8am clusters "
        "200 1 1 1 200 200 200 3000 1.0 0.995 0.5 42\n",
        argv[0]
    );
}