import os

missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

for m_r in missing_rates:
    print(f"***Missing Rate = {m_r}")
    os.system(f"python Impute-GRUD.py --data small_challenge_data --generate_files false --induce_missingness true --missing_rate {m_r}")

print("Completed all experiments.")
