#include <string.h>

#include "grid.h"


void grid_init(struct_grid *grid) { memset(grid->rsu, 0, sizeof(grid->rsu)); }


void grid_add_rsu(struct_grid *grid, int x, int y) { grid->rsu[x][y] = 1; }


void grid_remove_rsu(struct_grid *grid, int x, int y) { grid->rsu[x][y] = 0; }


int grid_has_rsu(const struct_grid *grid, int x, int y) { return grid->rsu[x][y] != 0; }


int grid_get_num_rsus(const struct_grid *grid)
{
    int count = 0;
    for (int x = 0; x < MAX_CELL_GRID_WIDTH; x++) {
        for (int y = 0; y < MAX_CELL_GRID_HEIGHT; y++) {

            if (grid->rsu[x][y])
                count++;
        }
    }

    return count;
}


int grid_flatten_rsus(const struct_grid *grid, struct_cell rsus[], int max_cells)
{
    int count = 0;

    for (int x = 0; x < MAX_CELL_GRID_WIDTH; x++) {
        for (int y = 0; y < MAX_CELL_GRID_HEIGHT; y++) {

            if (grid->rsu[x][y]) {

                if (count >= max_cells)
                    return -1;

                rsus[count].x = x;
                rsus[count].y = y;

                count++;
            }
        }
    }

    return count;
}