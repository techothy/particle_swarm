# particle_swarm
Static TF-IDF thresholds are a weak point in many NLP pipelines: they are often guessed, corpus-blind, and misaligned with the downstream task. This project treats `(min_df, max_df)` as a 2D continuous search problem and optimizes them with PSO using stratified CV macro-F1, the same family of metric used in the final benchmark.
