import pandas as pd
import numpy as np

# Define the probability parameters and number of draws
prob_A = 1 / 7
prob_B = 1 / 7
num_draws = 6  # Number of draws in each trial

# Perform a single trial with 6 draws
def single_trial():
    # Simulate 6 draws where 1 represents A, 2 represents B, and 0 represents neither
    draws = np.random.choice([0, 1, 2], size=num_draws, p=[1 - (prob_A + prob_B), prob_A, prob_B])
    return draws

# Run multiple trials and store results
num_trials = 100000000  # Number of trials for the simulation
results = []

for _ in range(num_trials):
    trial = single_trial()
    # Check if both A (1) and B (2) appeared in the 6 draws
    both_occurred = int(1 in trial and 2 in trial)
    results.append({"Draws": list(trial), "Both_A_and_B_Occurred": both_occurred})

# Create a DataFrame to display results
df_results = pd.DataFrame(results)

# Calculate the percentage of trials where both A and B occurred at least once
percent_both_occurred = df_results["Both_A_and_B_Occurred"].mean() * 100

# Display the results table and the percentage
print(f"Percentage of trials where both A and B occurred at least once: {percent_both_occurred:.2f}%")
