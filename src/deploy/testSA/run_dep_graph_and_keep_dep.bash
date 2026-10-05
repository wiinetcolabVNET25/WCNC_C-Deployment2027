#!/bin/bash

rsus_values=(1056)

for rsus in "${rsus_values[@]}"; do
    echo "$rsus"
done | xargs -P 8 -I {} bash -c '
    echo "Start RSU={}"
    python sa_tune.py --scenarios graph --results-dir graph_res_keep_dep --executable SA_deployment --rsus {}
    echo "End RSU={}"
'