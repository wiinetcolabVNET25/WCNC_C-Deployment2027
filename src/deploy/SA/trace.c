#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trace.h"


int trace_init(struct_trace *trace, char *error_msg)
{
    trace->records = malloc(sizeof(struct_trace_record) * MAX_TRACE_SIZE);
    if (trace->records == NULL)
    {
        sprintf(error_msg, "Could not allocate trace records.");
        return 1;
    }

    trace->vehicles = malloc(sizeof(struct_vehicle) * MAX_NUMBER_OF_VEHICLES);
    if (trace->vehicles == NULL)
    {
        sprintf(error_msg, "Could not allocate vehicles.");
        free(trace->records);
        trace->records = NULL;
        return 1;
    }

    trace->size = 0;
    trace->number_of_vehicles = 0;

    return 0;
}


void trace_free(struct_trace *trace)
{
    free(trace->records);
    free(trace->vehicles);

    trace->records = NULL;
    trace->vehicles = NULL;

    trace->size = 0;
    trace->number_of_vehicles = 0;
}


int trace_read(const char *trace_file_name, int dwell_multiplier, struct_trace *output_trace,
    char *output_error_msg)
{
    FILE *trace_file;

    char line[300];

    int vehicle_id;
    int time_instant;
    int x;
    int y;
    int cell_dwell_time;

    int previous_vehicle_id = -1;

    trace_file = fopen(trace_file_name, "r");
    if (trace_file == NULL)
    {
        sprintf(output_error_msg, "Could not open trace file: %s", trace_file_name);
        return 1;
    }

    while (fgets(line, sizeof(line), trace_file) != NULL)
    {
        /*
            Trace format:
            id;time;x;y;dwell;\n
        */
        if (sscanf(line, "%d;%d;%d;%d;%d;", &vehicle_id, &time_instant, &x, &y, &cell_dwell_time) != 5) {
            
            fclose(trace_file);
            sprintf(output_error_msg, "Invalid trace line: %s", line);
            return 1;
        }

        if (output_trace->size >= MAX_TRACE_SIZE)
        {
            fclose(trace_file);
            sprintf(output_error_msg, "Maximum trace size exceeded (%d).", MAX_TRACE_SIZE);
            return 1;
        }

        if (x < 0 || x >= MAX_CELL_GRID_WIDTH || y < 0 || y >= MAX_CELL_GRID_HEIGHT)
        {
            fclose(trace_file);
            sprintf(output_error_msg, "Invalid cell position (%d, %d).", x, y);
            return 1;
        }

        if (cell_dwell_time < 0)
        {
            fclose(trace_file);
            sprintf(output_error_msg, "Invalid cell dwell time: %d.", cell_dwell_time);
            return 1;
        }

        /*
            Trace is grouped by ids
        */
        if (vehicle_id != previous_vehicle_id)
        {
            if (output_trace->number_of_vehicles >= MAX_NUMBER_OF_VEHICLES)
            {
                fclose(trace_file);
                sprintf(output_error_msg, "Maximum number of vehicles exceeded (%d).", MAX_NUMBER_OF_VEHICLES);
                return 1;
            }

            output_trace->vehicles[ output_trace->number_of_vehicles ].vehicle_id = vehicle_id;
            output_trace->vehicles[ output_trace->number_of_vehicles ].trace_start = output_trace->size;
            output_trace->vehicles[ output_trace->number_of_vehicles ].trace_end = output_trace->size;

            output_trace->number_of_vehicles++;

            previous_vehicle_id = vehicle_id;
        }

        output_trace->records[output_trace->size].vehicle_id = vehicle_id;
        output_trace->records[output_trace->size].time_instant = time_instant;
        output_trace->records[output_trace->size].grid_x_pos = x;
        output_trace->records[output_trace->size].grid_y_pos = y;
        output_trace->records[output_trace->size].cell_dwell_time = cell_dwell_time * dwell_multiplier;

        /*
         * Update end position of the current vehicle.
         */
        output_trace->vehicles[ output_trace->number_of_vehicles - 1 ].trace_end = output_trace->size;

        output_trace->size++;
    }

    if (output_trace->size == 0)
    {
        sprintf(output_error_msg, "Trace is empty.");
        fclose(trace_file);
        return 1;
    }

    fclose(trace_file);
    return 0;
}


int trace_find_vehicle(const struct_trace *trace, int vehicle_id)
{
    for (int i = 0; i < trace->number_of_vehicles; i++)
    {
        if (trace->vehicles[i].vehicle_id == vehicle_id)
            return i;
    }

    return -1;
}


const struct_vehicle *trace_get_vehicle(const struct_trace *trace, int vehicle_index)
{
    if (vehicle_index < 0 || vehicle_index >= trace->number_of_vehicles)
    {
        return NULL;
    }

    return &trace->vehicles[vehicle_index];
}