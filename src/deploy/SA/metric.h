#ifndef METRIC_H
#define METRIC_H

#include "trace.h"
#include "grid.h"

#define MAX_NUMBER_OF_CLUSTERS 3


typedef struct vehicle_metrics_
{
    double N;
    double Tn;
    double Tc;

} struct_vehicle_metrics;


typedef struct cluster_metrics_
{
    double mean_N;
    double mean_Tn;
    double mean_Tc;

    int number_of_vehicles;

} struct_cluster_metrics;


typedef struct cluster_satisfaction_
{
    double N_satisfaction;
    double Tn_satisfaction;
    double satisfaction;

} struct_cluster_satisfaction;


/*
 * Calculate N, Tn and Tc for every vehicle.
 */
void metric_calculate_vehicle_metrics(
    const struct_trace *trace,
    const struct_grid *grid,
    struct_vehicle_metrics output_metrics[]);


/*
 * Calculate cluster-level averages.
 *
 * vehicle_clusters[i] is the cluster of the vehicle
 * whose internal index is i.
 */
void metric_calculate_cluster_metrics(
    const struct_vehicle_metrics vehicle_metrics[],
    const int vehicle_clusters[],
    int number_of_vehicles,
    int number_of_clusters,
    struct_cluster_metrics output_metrics[]);


/*
 * Calculate service satisfaction.
 */
void metric_calculate_satisfaction(
    const struct_cluster_metrics cluster_metrics[],
    const double N_required[], const double Tn_required[],
    int number_of_clusters, double alpha,
    struct_cluster_satisfaction output_satisfaction[]);

#endif