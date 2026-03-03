"""
Travel time model matching paper Section 2.1 (Eq.1-11).
T(a,b,t) = D(a,b)/V + B(a,b,t)*R(t) + I_accidents(t)*D_accident
- B = α * F_time(t) * F_distance(D), F_time peaks at 8h and 17h (480, 1020 min), σpeak=1.5h (90 min).
- R(t) ~ LogNormal(μ(t), σ(t)); μbase=0, σbase=0.3; δ=0.1, ε=0.2.
- Accidents: Poisson λ(t) peaks at μnight=21h (1260 min), σacc=2h (120 min); delay U(0.5, 2)h = 30-120 min.
Fixed seed in evaluation yields reproducible T(·) for the same inputs.
"""
import math
import random
from datetime import datetime, time
from city import Map
import numpy as np

def normal_distribution(x, mean, std_dev):
    """Gaussian f(t; μ, σ) per paper Eq.4."""
    return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))

def time_factor(current_time):
    """F_time(t): peaks at 8:00 (480 min) and 17:00 (1020 min), σpeak=90 min (1.5h). Paper Eq.3."""
    morning_peak = normal_distribution(current_time, 480, 90)
    evening_peak = normal_distribution(current_time, 1020, 90)
    return 0.5 + 2 * (morning_peak + evening_peak)

def random_factor(current_time):
    """R(t) ~ LogNormal(μ(t), σ(t)). Paper Eq.6-8: μbase=0, σbase=0.3, δ=0.1, ε=0.2."""
    rush_hour_effect = normal_distribution(current_time, 480, 90) + normal_distribution(current_time, 1020, 90)
    mu = 0 + 0.1 * rush_hour_effect
    sigma = 0.3 + 0.2 * rush_hour_effect
    return random.lognormvariate(mu, sigma)

def sample_accidents(current_time):
    """Poisson accidents; peak at 21:00 (1260 min), σacc=120 min. Paper Eq.9-10. Delay U(0.5, 2)h. Eq.11."""
    accident_rate = 0.05 * normal_distribution(current_time, 1260, 120)
    if accident_rate < 0:
        accident_rate = 0
    return np.random.poisson(lam=accident_rate)

def calculate_delay(distance, current_time):
    """B(a,b,t)*R(t) + accident term. F_distance(D) = 1 - exp(-D/λdist), λdist=50. Paper Eq.2, Eq.5."""
    time_fac = time_factor(current_time)
    distance_factor = 1 - math.exp(-distance / 50)
    base_delay = 0.25 * time_fac * distance_factor
    rand_factor = random_factor(current_time)
    delay = base_delay * rand_factor

    num_accidents = sample_accidents(current_time)
    accident_delay = 0
    if num_accidents > 0:
        durations = np.random.uniform(30, 120, size=num_accidents)  # 0.5h--2h in minutes
        accident_delay = np.sum(durations)
    delay += accident_delay

    return delay

def deterministic_travel_time(a, b, distances, velocity=1):
    """Deterministic travel time D(a,b)/V (no stochastic delay). Use for 'current segment' in simulation."""
    if a == b:
        return 0.0
    return distances[(a, b)] / velocity


def sample_travel_time(a, b, distances, current_time, velocity=1):
    """T(a,b,t) = D(a,b)/V + delay. Paper Eq.1."""
    if a == b:
        return 0.0
    distance = distances[(a, b)]
    delay = calculate_delay(distance, current_time)
    return distance / velocity + delay

def get_distances(map):
    distances = {}
    locations = map.locations
    for i in range(len(locations)):
        for j in range(len(locations)):
            distance = locations[i].distance(locations[j])
            distances[(i, j)] = distance
    return distances

if __name__ == "__main__":
    map = Map((100, 100), 1, 1)
    map.sample_locations(2)
    
    distances = get_distances(map) # calculate euclidean distances between all locations in map.locations
    current_time = 500 # start time of travel in [1, 1440]
    a = 0 # index of departure location in map.locations 
    b = 1 # index of arrival location in map.locations
    travel_time = sample_travel_time(a, b, distances, current_time, velocity=1)
    print(travel_time)
    
