#!/bin/bash

rsus_values=(
    95 191 286 382 477 573 668 764 859 955 1050
    1146 1241 1337 1432 1528 1623 1719 1814 1910 
)

for rsus in "${rsus_values[@]}"; do
    echo "$rsus"
done | xargs -P 2 -I {} bash -c '
    echo "Start RSU={}"
    python sa_tune_inverted.py --cleanup --scenarios vehicle --results-dir veic_res_inverted --executable SA_deployment --rsus {}
    echo "End RSU={}"
'
