#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "config.h"
#include "sa.h"


/* ============================================================
 * Random utilities
 * ============================================================ */

static int random_int(int min, int max) { return min + rand() % (max - min + 1); }


static double random_double(void) { return (double) rand() / (double) RAND_MAX; }


/* ============================================================
 * SA state initialization
 * ============================================================ */

/*
 * Initialize the SA state from the complete list of candidate
 * cells.
 *
 * Initially:
 *
 *     deployment = empty
 *     available = all candidate cells
 *
 * Then number_of_rsus cells are randomly removed from the
 * available set and placed into the deployment.
 *
 * Selecting an element and replacing it with the last available
 * element makes each selection O(1).
 */
static int initialize_state(const struct_cell candidate_cells[], int number_of_candidate_cells,
    int number_of_rsus, struct_sa_state *state)
{
    if (number_of_rsus > number_of_candidate_cells)
        return 1;

    if (number_of_rsus > MAX_NUMBER_OF_RSUS)
        return 1;

    if (number_of_candidate_cells > MAX_NUMBER_OF_CANDIDATE_CELLS)
        return 1;

    /*
     * Copy all candidate cells into the available array.
     */
    memcpy(state->available, candidate_cells, sizeof(struct_cell) * number_of_candidate_cells);

    state->available_size = number_of_candidate_cells;

    /*
     * Randomly select cells for the initial deployment.
     *
     * At each iteration, choose one element from the current
     * available range and replace it with the last element.
     */
    for (int i = 0; i < number_of_rsus; i++)
    {
        int index = random_int(0, state->available_size - 1);

        /*
         * Move selected cell into the deployment.
         */
        state->deployment.rsus[i] = state->available[index];

        /*
         * Remove selected cell from available cells.
         */
        state->available[index] = state->available[state->available_size - 1];

        state->available_size--;
    }

    state->deployment.size = number_of_rsus;

    return 0;
}


/* ============================================================
 * Grid utilities
 * ============================================================ */

/*
 * Build the RSU occupancy grid from a deployment.
 *
 * This is called only once for the initial solution.
 * During SA iterations the grid is updated incrementally.
 */
static void deployment_to_grid(const struct_deployment *deployment, struct_grid *grid)
{
    grid_init(grid);
    for (int i = 0; i < deployment->size; i++)
    {
        grid_add_rsu(grid, deployment->rsus[i].x, deployment->rsus[i].y);
    }
}


/*
 * Apply a deployment <-> available swap to the RSU grid.
 *
 * old_cell is the RSU position being removed.
 * new_cell is the candidate position being added.
 */
static void apply_swap_to_grid(struct_grid *grid, struct_cell old_cell, struct_cell new_cell)
{
    grid->rsu[old_cell.x][old_cell.y] = 0;
    grid->rsu[new_cell.x][new_cell.y] = 1;
}


/* ============================================================
 * Neighbor generation
 * ============================================================ */

/*
 * Swap one deployed RSU with one available candidate cell.
 *
 * This modifies the state in-place.
 *
 * Before:
 *
 *     deployment[d] = old_cell
 *     available[a]  = new_cell
 *
 * After:
 *
 *     deployment[d] = new_cell
 *     available[a]  = old_cell
 *
 * The grid is updated at the same time.
 */
static void generate_neighbor(
    struct_sa_state *state,
    struct_grid *grid,
    int *deployment_index,
    int *available_index,
    struct_cell *old_cell,
    struct_cell *new_cell)
{
    int d = random_int(0, state->deployment.size - 1);
    int a = random_int(0, state->available_size - 1);

    *deployment_index = d;
    *available_index = a;

    *old_cell = state->deployment.rsus[d];
    *new_cell = state->available[a];

    /*
     * Swap deployment and available cells.
     */
    state->deployment.rsus[d] = *new_cell;
    state->available[a] = *old_cell;

    /*
     * Update RSU occupancy grid.
     */
    apply_swap_to_grid(grid, *old_cell, *new_cell);
}


/*
 * Undo a previously generated neighbor.
 */
static void undo_neighbor(
    struct_sa_state *state,
    struct_grid *grid,
    int deployment_index,
    int available_index,
    struct_cell old_cell,
    struct_cell new_cell)
{
    /*
     * Restore deployment.
     */
    state->deployment.rsus[deployment_index] = old_cell;

    /*
     * Restore available cell.
     */
    state->available[available_index] = new_cell;

    /*
     * Restore grid.
     */
    apply_swap_to_grid(grid, new_cell, old_cell);
}


/* ============================================================
 * Objective
 * ============================================================ */

/*
 * Calculates the complete state of a deployment.
 *
 * The RSU grid is supplied by the caller and is maintained
 * incrementally by the SA algorithm.
 *
 * The metric arrays are supplied by the caller and reused
 * across SA iterations.
 *
 * Returns:
 *
 *     objective = -min(cluster satisfaction)
 */
static double calculate_objective(
    const struct_trace *trace,
    const int vehicle_clusters[],
    const struct_grid *grid,
    const double N_required[],
    const double Tn_required[],
    double alpha,

    struct_vehicle_metrics vehicle_metrics[],
    struct_cluster_metrics cluster_metrics[],
    struct_cluster_satisfaction satisfaction[])
{
    /*
     * Vehicle-level metrics.
     */
    metric_calculate_vehicle_metrics(trace, grid, vehicle_metrics);

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
     * Find minimum cluster satisfaction.
     *
     * The SA implementation minimizes the objective, so
     * maximizing minimum satisfaction is represented as:
     *
     *     objective = -min(satisfaction)
     */
    double min_satisfaction = 1.0;
    for (int k = 0; k < MAX_NUMBER_OF_CLUSTERS; k++)
    {
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
    const struct_cell candidate_cells[],
    const int number_of_candidate_cells,
    const int number_of_rsus,
    const double N_required[],
    const double Tn_required[],
    const struct_sa_config config,
    const unsigned int seed,
    struct_sa_result *output_result,
    char *error_msg
)
{
    static_assert(MAX_NUMBER_OF_RSUS <= MAX_NUMBER_OF_CANDIDATE_CELLS);

    /*
     * Validate input.
     */
    if (number_of_candidate_cells <= 0)
    {
        sprintf(error_msg, "No candidate cells were provided.");
        return 1;
    }

    if (number_of_rsus <= 0)
    {
        sprintf(error_msg, "Number of RSUs must be greater than zero.");
        return 1;
    }

    if (number_of_rsus > number_of_candidate_cells)
    {
        sprintf(error_msg, "Number of RSUs (%d) exceeds number of candidate cells (%d).",
            number_of_rsus, number_of_candidate_cells);
        return 1;
    }

    if (number_of_rsus > MAX_NUMBER_OF_RSUS)
    {
        sprintf(error_msg,
            "Number of RSUs exceeds MAX_NUMBER_OF_RSUS (%d).",
            MAX_NUMBER_OF_RSUS);
        return 1;
    }

    if (config.number_of_iterations < 0)
    {
        sprintf(error_msg, "Number of iterations cannot be negative.");
        return 1;
    }

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

    struct_sa_state current_state;

    struct_grid current_grid;

    struct_cluster_satisfaction current_satisfaction[MAX_NUMBER_OF_CLUSTERS];
    struct_cluster_satisfaction best_satisfaction[MAX_NUMBER_OF_CLUSTERS];


    /* ========================================================
     * Initial solution
     * ======================================================== */

    srand(seed);

    if (initialize_state(candidate_cells, number_of_candidate_cells, number_of_rsus, &current_state) != 0)
    {
        sprintf(error_msg, "Could not initialize SA state.");

        free(vehicle_metrics);
        free(cluster_metrics);
        free(satisfaction);

        sa_result_free(output_result);

        return 1;
    }

    /*
     * Build the RSU grid ONCE.
     *
     * It will be updated incrementally from now on.
     */
    deployment_to_grid(&current_state.deployment, &current_grid);


    /* ========================================================
     * Evaluate initial solution
     * ======================================================== */

    double current_objective =
        calculate_objective(
            trace, vehicle_clusters, &current_grid,
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
    output_result->best_solution = current_state.deployment;

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

        int deployment_index;
        int available_index;

        struct_cell old_cell;
        struct_cell new_cell;

        generate_neighbor(
            &current_state,
            &current_grid,
            &deployment_index,
            &available_index,
            &old_cell,
            &new_cell
        );


        /* ----------------------------------------------------
         * Evaluate neighbor
         * ---------------------------------------------------- */

        double neighbor_objective =
            calculate_objective(
                trace, vehicle_clusters, &current_grid,
                N_required, Tn_required,
                config.alpha,
                vehicle_metrics, cluster_metrics, satisfaction);

        /*
         * `satisfaction` now corresponds to the neighbor.
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

        if (accept)
        {
            /*
             * The state and grid already contain the neighbor.
             */
            current_objective =
                neighbor_objective;

            /*
             * The satisfaction currently stored in `satisfaction`
             * belongs to the accepted neighbor.
             */
            for (int k = 0;
                 k < MAX_NUMBER_OF_CLUSTERS;
                 k++)
            {
                current_satisfaction[k] =
                    satisfaction[k];
            }
        }
        else
        {
            /*
             * Restore previous state and grid.
             */
            undo_neighbor(
                &current_state,
                &current_grid,
                deployment_index,
                available_index,
                old_cell,
                new_cell
            );
        }


        /* ----------------------------------------------------
         * Update global best
         * ---------------------------------------------------- */

        if (current_objective < best_objective)
        {
            output_result->best_solution = current_state.deployment;
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

    output_result->best_objective = best_objective;

    /* ========================================================
     * Cleanup
     * ======================================================== */

    free(vehicle_metrics);
    free(cluster_metrics);
    free(satisfaction);

    return 0;
}


/* ============================================================
 * Result cleanup
 * ============================================================ */

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