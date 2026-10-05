## Root folders

**common/**: Utilities
    - clustering and visualization
    - deploying (read trace, read deployment, calculate metrics)
    - features (get feature dataframes)
    - graph (get graph, community clustering)
    - plot (plot traj., plot deployments)

**data/**: 6am - 8am mobility trace

**src/**: cell, graph, vehicle clustering, deployment methods, see **notebooks** and **deploy** methods description below

## Notebooks description

#### Clustering with cells' features and Graph Communities:

    src/cells/c_features.ipynb:
        - basic cells' features (pop, dwell time, start count (trips starting in cell), end count (strips ending in cell)) visualization

    src/cells/c_clust_kmeans.ipynb:
        - testing kmeans performance and sensitivity with many cells' features combinations
        - also, visualizations of specific clusterings

    src/cells/c_clust_kmeans_interp.ipynb:
        - comparing cells clusters data distribution and aggregate data and
        - visualizing trips and comparing data distribution and aggregate data of vehicles separated in each cluster by some rule, such as the cluster they stayed the most time within

    src/cells/c_graph_community.ipynb
        - testing community clustering with cells' as a weighted directed graph
        - also, like in "src/cells/c_clust_kmeans_interp.ipynb", comparing induced vehicles' clusters

The definitive utilized clusters are **"vc_from_c.csv"** (from cells, cell representation) and **"vc_from_g.csv"** (from graph, graph representation)

#### Vehicles' oriented:

    src/veic/v_features.ipynb:
        - basic vehicle's features (trip time, entropy, etc) visualization, exploratory analysis
        - complete trace and reduced trace (20%) comparison

    src/veic/v_clust_kmeans.ipynb:
        - testing kmeans performance and sensitivity with many vehicles' features combinations, also, visualizations
        - may be necessary to reduce the number of vehicles via extracting spatiotemporal patterns or sampling

    src/veic/v_clust_kmeans_interp.ipynb:
        - interpretation of specific clustering results
            - comparing vehicles data distribution and aggregate data
            - visualizating vehicles clusters

The definitive utilized cluster is **vc_from_v.csv** (cluster from the vehicles themselves), which is from a reduced trace (**v_trace_reduced.csv**)

## Deploy methods

    src/deploy/SA
        - Simmulated annealing source (must compile with make)
    
    src/deploy/testSA
        - SA results for multiple scenarios/budgets
        - sa_tune.py: runs SA for the original policy (requires SA_deployment binary)
        - sa_tune_inverted.py: runs SA for the inverted policy (requires SA_deployment binary) 
        - .bash files: run for cells, graph, vehicles clusters (results saved in directories)
        - see src/deploy/BASELINES for comparison of SA results with other methods

    src/deploy/BASELINES
        - runs baseline methods (dense, random) and compare against SA results for the 6 scenarios (see src/deploy/testSA)
        - detailed graphs inside "src/deploy/BASELINES/results/plots"

## Paper Contents

- For **Table I** service assignments, consult "src/deploy/testSA/sa_tune.py" and "src/deploy/testSA/sa_tune_inverted.py"

- For the paper's **Figure 2** graphs, see "src/deploy/BASELINES/simple_graphs.ipynb" (plots in "src/deploy/BASELINES/results/plots/*objetive.png/pdf")

- For **Tables II and III + individual compliance values**, consult "src/other_tables"