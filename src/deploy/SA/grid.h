#ifndef GRID_H
#define GRID_H

#include "cell.h"

#define MAX_CELL_GRID_WIDTH  100
#define MAX_CELL_GRID_HEIGHT 100

typedef struct grid_
{
    /*
     * rsu[x][y] == 1 if an RSU is deployed in cell (x,y).
     */
    unsigned char rsu[MAX_CELL_GRID_WIDTH][MAX_CELL_GRID_HEIGHT];

} struct_grid;


/*
 * Initialize all cells to zero RSUs.
 */
void grid_init(struct_grid *grid);


/*
 * Add/remove an RSU.
 */
void grid_add_rsu(struct_grid *grid, int x, int y);
void grid_remove_rsu(struct_grid *grid, int x, int y);


/*
 * Return 1 if an RSU exists at (x,y), 0 otherwise.
 */
int grid_has_rsu(const struct_grid *grid, int x, int y);


/*
 * Return the number of deployed RSUs.
 */
int grid_get_num_rsus(const struct_grid *grid);


int grid_flatten_rsus(const struct_grid *grid, struct_cell rsus[], int max_cells);


#endif