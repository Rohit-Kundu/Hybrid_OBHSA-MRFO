# Hybrid_OBHSA-MRFO

This repository presents a feature selection algorithm that can be used on both machine learning datasets (UCI datasets) or deep feature sets.

Hybrid of two Wrapper feature selection algorithms- Opposition Based Harmony Search (OBHSA) and Manta Ray Foraging Optimization (MRFO). Populations are generated from both the methods and the final population from both are stacked and the average ranks calculated to determine importance of features. Top 'k' highest ranked features are used. The hybridization strategy is based on the paper: [A Hybrid Meta-Heuristic Feature Selection Method Using Golden Ratio and Equilibrium Optimization Algorithms for Speech Emotion Recognition](https://ieeexplore.ieee.org/abstract/document/9247182)

To run the Hybrid OBHSA-MRFO algorithm on a feature set, first arrange the features in a csv file such that it the last column of the csv contains the original labels of the samples. Then run the following using the command prompt.

`python main.py --csv_name features.csv`

Other available parameters are: `--csv_headers` whose default value `no` suggests that the csv file has no headers. Change to `yes` if your file has headers. `--generations` has the default value of `20` meaning each algorithm (OBHSA and MRFO) will run for 20 generations to generate the populations. `popSize` has the default value of `20`, i.e., the deafult size of population is 20.
