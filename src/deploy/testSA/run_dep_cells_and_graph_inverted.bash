#!/bin/bash

rsus_values=(
    105 211 316 422 528 633 739 845 950 1056
    1161 1267 1373 1478 1584 1690 1795 1901 2006 2112 
)

for rsus in "${rsus_values[@]}"; do
    echo "$rsus"
done | xargs -P 10 -I {} bash -c '
    echo "Start RSU={}"
    python sa_tune_inverted.py --cleanup --scenarios cell graph --results-dir cell_graph_res_inverted --executable SA_deployment --rsus {}
    echo "End RSU={}"
'
