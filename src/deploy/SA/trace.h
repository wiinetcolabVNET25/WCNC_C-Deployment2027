#ifndef TRACE_H
#define TRACE_H

#include "grid.h"

#define MAX_TRACE_SIZE              2500000
#define MAX_NUMBER_OF_VEHICLES      80000

typedef struct trace_record_
{
    int vehicle_id;
    int time_instant;
    int grid_x_pos;
    int grid_y_pos;
    int cell_dwell_time;

} struct_trace_record;

typedef struct vehicle_
{
    int vehicle_id;
    int trace_start;
    int trace_end;

} struct_vehicle;

typedef struct trace_
{
    struct_trace_record *records;
    struct_vehicle *vehicles;

    int size;
    int number_of_vehicles;
    
} struct_trace;

int trace_init(struct_trace *trace, char *error_msg);

void trace_free(struct_trace *trace);

int trace_read(const char * trace_file_name, int dwell_multiplier, struct_trace *output_trace,
    char *output_error_msg);

int trace_find_vehicle(const struct_trace *trace, int vehicle_id);

const struct_vehicle *trace_get_vehicle( const struct_trace *trace, int vehicle_index);

#endif