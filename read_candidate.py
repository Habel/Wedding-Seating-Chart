# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "thefuzz",
#     "python-Levenshtein",
# ]
# ///


import pandas as pd
import sys

if __name__ == "__main__":
    fn = sys.argv[1]
    candidate = pd.read_csv(fn)
    for t, group in candidate.groupby("Table"):
        print(f"{t}: {group['guest'].to_list()}")
