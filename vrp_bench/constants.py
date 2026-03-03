CUSTOMER = "customer"
DEPOT = "depot"
NUM_INSTANCES = 10
DEMAND_RANGE = (0, 100) #-> not used in tsp_tw
DYNAMIC_PERCENTAGE = 0.1
MAP_SIZE = (1000, 1000)
REALIZATIONS_PER_MAP = 5
# Evaluation protocol (paper Section 4-5)
NUM_STOCHASTIC_REALIZATIONS = 5
OR_TOOLS_TIME_LIMIT_SECONDS = 300
# Per-instance time limit for Tabu Search (avoids runaway on large instances)
TABU_TIME_LIMIT_SECONDS = 300
MAX_STAGNANT_ITERATIONS = 1000
# Reproducibility: fixed seed for paper-aligned experiments
PAPER_SEED = 42
