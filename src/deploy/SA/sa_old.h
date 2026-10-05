#ifndef SA_H
#define SA_H

#include "grid.h"
#include "trace.h"
#include "metric.h"

/*
    Careful: Allowing more rsus than candidate cells hangs the program
*/
#define MAX_NUMBER_OF_RSUS 2500


typedef struct cell_
{
    int x;
    int y;

} struct_cell;


typedef struct deployment_
{
    struct_cell rsus[MAX_NUMBER_OF_RSUS];

    int size;

} struct_deployment;


typedef struct sa_config_
{
    int number_of_iterations;

    double initial_temperature;
    double cooling_rate;

    double alpha;

} struct_sa_config;


typedef struct sa_result_
{
    struct_deployment best_solution;

    double best_objective;

    int number_of_iterations;

    double *temperature_history;
    double *objective_history;
    double *best_objective_history;

    double *current_satisfaction_history[MAX_NUMBER_OF_CLUSTERS];
    double *best_satisfaction_history[MAX_NUMBER_OF_CLUSTERS];

} struct_sa_result;


/*
 * Run simulated annealing for a fixed number of RSUs.
 */
int sa_optimize(
    const struct_trace *trace,
    const int vehicle_clusters[],
    const struct_grid *candidate_grid,
    const int number_of_rsus,
    const double N_required[],
    const double Tn_required[],
    const struct_sa_config config,
    const unsigned int seed,
    struct_sa_result *output_result,
    char* error_msg
);

void sa_result_free(struct_sa_result *result);

#endif