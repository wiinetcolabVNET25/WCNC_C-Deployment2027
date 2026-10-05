#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "config.h"
#include "sa.h"


/* ============================================================
 * Random utilities
 * ============================================================ */

static int random_int(int min, int max) { return min + rand() % (max - min + 1); }


static double random_double(void) { return (double) rand() / (double) RAND_MAX; }


/* ============================================================
 * Deployment utilities
 * ============================================================ */

static int deployment_contains(const struct_deployment *deployment, int x, int y)
{
    for (int i = 0; i < deployment->size; i++) {

        if (deployment->rsus[i].x == x && deployment->rsus[i].y == y) {
            return 1;
        }
    }

    return 0;
}


static void deployment_to_grid(const struct_deployment *deployment, struct_grid *grid)
{
    grid_init(grid);

    for (int i = 0; i < deployment->size; i++) {

        grid_add_rsu(grid, deployment->rsus[i].x, deployment->rsus[i].y);
    }
}


static void random_deployment(const struct_grid *candidate_grid, int number_of_rsus, struct_deployment *output)
{
    output->size = 0;
    while (output->size < number_of_rsus) {

        int x = random_int(0, MAX_CELL_GRID_WIDTH - 1);
        int y = random_int(0, MAX_CELL_GRID_HEIGHT - 1);

        /*
         * Not a valid candidate cell.
         */
        if (!candidate_grid->rsu[x][y])
            continue;

        /*
         * Already selected.
         */
        if (deployment_contains(output, x, y)) {
            continue;
        }

        output->rsus[output->size].x = x;
        output->rsus[output->size].y = y;

        output->size++;
    }
}


/*
 * Move one RSU to a new candidate cell.
 */
static void generate_neighbor(const struct_grid *candidate_grid, const struct_deployment *current, struct_deployment *neighbor)
{
    *neighbor = *current;
    int rsu_index = random_int(0, current->size - 1);

    int x;
    int y;
    while (1) {

        x = random_int(0, MAX_CELL_GRID_WIDTH - 1);
        y = random_int(0, MAX_CELL_GRID_HEIGHT - 1);

        if (!candidate_grid->rsu[x][y])
            continue;

        /*
         * Do not select another RSU's current cell.
         */
        if (deployment_contains(current, x, y)) {
            continue;
        }

        break;
    }

    neighbor->rsus[rsu_index].x = x;
    neighbor->rsus[rsu_index].y = y;
}


/* ============================================================
 * Objective
 * ============================================================ */

/*
* Calculates the complete state of a deployment.
*
* The metric arrays are supplied by the caller and reused
* across SA iterations.
*
* Returns the combined objective:
*
*     objective = -min(cluster satisfaction)
*
* The individual satisfactions are left in:
*
*     satisfaction[k].satisfaction
*/
static double calculate_objective(
    const struct_trace *trace,
    const int vehicle_clusters[],
    const struct_deployment *deployment,
    const double N_required[],
    const double Tn_required[],
    double alpha,

    struct_vehicle_metrics vehicle_metrics[],
    struct_cluster_metrics cluster_metrics[],
    struct_cluster_satisfaction satisfaction[]
)
{
    struct_grid grid;

    deployment_to_grid(deployment, &grid);

    /*
     * Vehicle-level metrics.
     */
    metric_calculate_vehicle_metrics(trace, &grid, vehicle_metrics);

    /*
     * Cluster-level metrics.
     */
    metric_calculate_cluster_metrics(vehicle_metrics, vehicle_clusters,
        trace->number_of_vehicles, MAX_NUMBER_OF_CLUSTERS, cluster_metrics);

    /*
     * Individual cluster satisfactions.
     */
    metric_calculate_satisfaction(cluster_metrics,
        N_required, Tn_required, MAX_NUMBER_OF_CLUSTERS,
        alpha, satisfaction);

    /*
     * Combined objective:
     *
     * maximize minimum satisfaction
     *
     * but SA is implemented as minimization.
     */
    double min_satisfaction = 1.0;
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++) {

        if (satisfaction[k].satisfaction < min_satisfaction) {
            min_satisfaction = satisfaction[k].satisfaction;
        }
    }

    return -min_satisfaction;
}

/* ============================================================
 * Simulated Annealing
 * ============================================================ */

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
)
{
    /*
     * Initialize result.
     */
    memset(output_result, 0, sizeof(struct_sa_result));

    output_result->number_of_iterations = config.number_of_iterations;

    /*
     * s0 + iterations
     */
    int history_size = config.number_of_iterations + 1;

    /* ========================================================
     * Allocate history
     * ======================================================== */

    output_result->temperature_history = malloc(sizeof(double) * history_size);
    output_result->objective_history = malloc(sizeof(double) * history_size);
    output_result->best_objective_history = malloc(sizeof(double) * history_size);
    if (output_result->temperature_history == NULL ||
        output_result->objective_history == NULL ||
        output_result->best_objective_history == NULL)
    {
        sprintf(error_msg, "Could not allocate SA history.");
        sa_result_free(output_result);
        return 1;
    }

    /*
     * Individual cluster satisfaction histories.
     */
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        output_result->current_satisfaction_history[k] = malloc(sizeof(double) * history_size);
        output_result->best_satisfaction_history[k] = malloc(sizeof(double) * history_size);
        if (output_result->current_satisfaction_history[k] == NULL ||
            output_result->best_satisfaction_history[k] == NULL)
        {
            sprintf(error_msg, "Could not allocate cluster satisfaction history.");
            sa_result_free(output_result);
            return 1;
        }
    }

    /* ========================================================
     * Metric arrays
     *
     * Allocated ONCE and reused across all iterations.
     * ======================================================== */

    struct_vehicle_metrics *vehicle_metrics = malloc(sizeof(struct_vehicle_metrics) * trace->number_of_vehicles);
    struct_cluster_metrics *cluster_metrics = malloc(sizeof(struct_cluster_metrics) * MAX_NUMBER_OF_CLUSTERS);
    struct_cluster_satisfaction *satisfaction = malloc(sizeof(struct_cluster_satisfaction) * MAX_NUMBER_OF_CLUSTERS);
    if (vehicle_metrics == NULL || cluster_metrics == NULL || satisfaction == NULL)
    {
        sprintf(error_msg, "Could not allocate SA metric arrays.");

        free(vehicle_metrics);
        free(cluster_metrics);
        free(satisfaction);

        sa_result_free(output_result);

        return 1;
    }

    /* ========================================================
     * Local SA state
     * ======================================================== */

    struct_cluster_satisfaction current_satisfaction[ MAX_NUMBER_OF_CLUSTERS ];
    struct_cluster_satisfaction best_satisfaction[ MAX_NUMBER_OF_CLUSTERS ];

    /* ========================================================
     * Initial solution
     * ======================================================== */

    srand(seed);

    struct_deployment current;
    random_deployment(candidate_grid, number_of_rsus, &current);

    /*
     * Evaluate initial solution.
     *
     */
    double current_objective =
        calculate_objective(
            trace, vehicle_clusters, &current,
            N_required, Tn_required,
            config.alpha,
            vehicle_metrics, cluster_metrics, satisfaction);

    /*
     * Save current satisfaction.
     */
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        current_satisfaction[k] = satisfaction[k];
    }

    /*
     * Initial best = initial current solution.
     */
    struct_deployment best = current;

    double best_objective = current_objective;

    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        best_satisfaction[k] = current_satisfaction[k];
    }

    double temperature = config.initial_temperature;

    /* ========================================================
     * History: iteration 0
     * ======================================================== */

    output_result->temperature_history[0] = temperature;
    output_result->objective_history[0] = current_objective;
    output_result->best_objective_history[0] = best_objective;

    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        output_result->current_satisfaction_history[k][0] = current_satisfaction[k].satisfaction;
        output_result->best_satisfaction_history[k][0] = best_satisfaction[k].satisfaction;
    }

    /* ========================================================
     * SA iterations
     * ======================================================== */

    for (int iteration = 0; iteration < config.number_of_iterations; iteration++)
    {
        #if DEBUG == 1
            if (iteration % 500 == 0)
                printf("Iteration: %d\n", iteration);
        #endif

        /* ----------------------------------------------------
         * Generate neighbor
         * ---------------------------------------------------- */

        struct_deployment neighbor;
        generate_neighbor(candidate_grid, &current, &neighbor);

        /* ----------------------------------------------------
         * Evaluate neighbor
         * ---------------------------------------------------- */

        double neighbor_objective =
            calculate_objective(
                trace, vehicle_clusters, &neighbor,
                N_required, Tn_required,
                config.alpha,
                vehicle_metrics, cluster_metrics, satisfaction);

        /*
         * `satisfaction` now corresponds to NEIGHBOR.
         */
        double delta = neighbor_objective - current_objective;

        /* ----------------------------------------------------
         * Acceptance
         * ---------------------------------------------------- */

        int accept = 0;

        /*
         * Better solution.
         */
        if (delta < 0.0) {

            accept = 1;

        } else {

            /*
             * Worse solution can still be accepted.
             */
            double probability =
                exp(-delta / temperature);

            if (random_double() < probability) {
                accept = 1;
            }
        }

        /* ----------------------------------------------------
         * Update current solution
         * ---------------------------------------------------- */

        if (accept) {

            current = neighbor;
            current_objective = neighbor_objective;

            /*
             * The satisfaction currently stored in `satisfaction`
             * belongs to the accepted neighbor, so it becomes
             * the current satisfaction.
             */
            for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
            {
                current_satisfaction[k] = satisfaction[k];
            }
        }

        /* ----------------------------------------------------
         * Update global best
         * ---------------------------------------------------- */

        if (current_objective < best_objective) {

            best = current;
            best_objective = current_objective;

            for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
            {
                best_satisfaction[k] = current_satisfaction[k];
            }
        }

        /* ----------------------------------------------------
         * Cooling
         * ---------------------------------------------------- */

        temperature *= config.cooling_rate;
        if (temperature < 1e-12)
            temperature = 1e-12;

        /* ----------------------------------------------------
         * Store history
         * ---------------------------------------------------- */

        int history_index = iteration + 1;

        output_result->temperature_history[history_index] = temperature;
        output_result->objective_history[history_index] = current_objective;
        output_result->best_objective_history[history_index] = best_objective;

        for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
        {
            output_result->current_satisfaction_history[k][history_index] = current_satisfaction[k].satisfaction;
            output_result->best_satisfaction_history[k][history_index] = best_satisfaction[k].satisfaction;
        }
    }


    /* ========================================================
     * Final result
     * ======================================================== */

    output_result->best_solution = best;
    output_result->best_objective = best_objective;

    /* ========================================================
     * Cleanup
     * ======================================================== */

    free(vehicle_metrics);
    free(cluster_metrics);
    free(satisfaction);

    return 0;
}

void sa_result_free(struct_sa_result *result)
{
    free(result->temperature_history);
    free(result->objective_history);
    free(result->best_objective_history);

    result->temperature_history = NULL;
    result->objective_history = NULL;
    result->best_objective_history = NULL;
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
        free(result->current_satisfaction_history[k]);
        free(result->best_satisfaction_history[k]);

        result->current_satisfaction_history[k] = NULL;
        result->best_satisfaction_history[k] = NULL;
    }

    result->number_of_iterations = 0;
}