# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///

import pandas as pd
import random
import math
import functools

# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///

import pandas as pd
import numpy as np
import random
import math

# --- CONFIGURATION ---

REPLACEMENTS = [
    ["Close Friends", "cf"],
    ["Ali Close Family", "acf"],
    ["Matt Close Family", "mcf"],
    ["Ali HS Friends", "ahs"],
    ["Other Habels", "dan"],
    ["Matt HS Friends", "mhs"],
    ["Matt Family Friends", "mff"],
    ["Ali Family Friends", "aff"],
    ["Ali Family", "af"],
    ["Cornell", "cornell"],
    ["UT", "ut"],
    ["SF Friends", "sff"],
    ["Malays", "malays"],
]

# Higher values = stronger desire to be seated near each other.
# Base 'Same Group' score in get_pair_affinity is 25.
AFFINITY_SCORES = {
    # --- FAMILY CLUSTERS ---
    ("acf", "af"): 15,  # Ali Close Family <-> Ali General Family
    ("af", "aff"): 10,  # Ali Family <-> Ali Family Friends
    ("mcf", "dan"): 15,  # Matt Close Family <-> Other Habels
    ("mff", "mcf"): 10,  # Matt Close Family <-> Other Habels
    ("acf", "mcf"): 15,  # The two immediate families
    ("malays", "mcf"): 10,
    ("malays", "dan"): 5,
    # --- FRIEND CLUSTERS ---
    ("cf", "sff"): 15,  # Close Friends <-> SF Friends (High overlap)
    ("cornell", "mhs"): 5,  # Ali HS <-> Matt HS (Common ground)
    ("ut", "ahs"): 10,
    ("cornell", "ut"): 5,  # College friends from different eras
    # --- CROSS-POLLINATION (Keep these lower to allow the optimizer flexibility) ---
    ("sff", "cornell"): 5,
    ("sff", "ut"): 5,
}

# Higher = better. Define overlaps between different groups here.
AFFINITY_MAP = {tuple(sorted(k)): v for k, v in AFFINITY_SCORES.items()}


def replace_groups(table_group):
    if pd.isna(table_group):
        return ["unknown"]
    group_list = [g.strip() for g in str(table_group).split(",")]
    new_groups = []
    for group in group_list:
        for match, rep in REPLACEMENTS:
            if group == match:
                new_groups.append(rep)
    return tuple(new_groups) if new_groups else ("unknown")


@functools.lru_cache(maxsize=None)
def get_pair_affinity(g1_list, g2_list):
    score = 0
    for g1 in g1_list:
        for g2 in g2_list:
            if g1 == g2:
                score += (
                    25  # High bonus for same group (keeps couples/friends together)
                )
            else:
                pair = tuple(sorted((g1, g2)))
                score += AFFINITY_MAP.get(pair, 0)
    return score


def get_local_score(idx, sequence):
    """Calculates affinity for a single person based on their neighbors."""
    score = 0
    n = len(sequence)
    # Tiers as discussed: T1 (1,4,3,5) @ 1.0, T2 (2,6) @ 0.5, T3 (7) @ 0.2
    offsets = {1: 1.0, 4: 1.0, 3: 1.0, 5: 1.0, 2: 0.5, 6: 0.5, 7: 0.2}

    for off, weight in offsets.items():
        # Check both directions since we are calculating individual happiness
        for direction in [off, -off]:
            neighbor_idx = idx + direction
            if 0 <= neighbor_idx < n:
                score += (
                    get_pair_affinity(
                        tuple(sequence[idx]["groups"]),
                        tuple(sequence[neighbor_idx]["groups"]),
                    )
                    * weight
                )
    return score


def calculate_total_score(sequence):
    return sum(get_local_score(i, sequence) for i in range(len(sequence))) / 2


def simulated_annealing(guests, iterations=150000, temp=10.0, cooling=0.99999):
    current_seq = list(guests)
    random.shuffle(current_seq)
    current_score = calculate_total_score(current_seq)

    best_seq = list(current_seq)
    best_score = current_score

    print(f"Starting Score: {current_score:.2f}")

    for i in range(iterations):
        # Swap logic: 80% small swaps (nearby people), 20% big jumps
        idx1 = random.randint(0, len(current_seq) - 1)
        if random.random() < 0.8:
            idx2 = (idx1 + random.randint(-10, 10)) % len(current_seq)
        else:
            idx2 = random.randint(0, len(current_seq) - 1)

        current_seq[idx1], current_seq[idx2] = current_seq[idx2], current_seq[idx1]
        new_score = calculate_total_score(current_seq)

        if new_score > current_score or (
            temp > 0 and random.random() < math.exp((new_score - current_score) / temp)
        ):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_seq = list(current_seq)
        else:
            current_seq[idx1], current_seq[idx2] = current_seq[idx2], current_seq[idx1]

        temp *= cooling
        if i % 25000 == 0:
            print(
                f"Iteration {i}... Score: {current_score:.2f} (Best: {best_score:.2f})"
            )

    return best_seq


def print_table_stats(result_df, affinity_map):
    print("\n--- TABLE HARMONY REPORT ---")
    for table_num, group in result_df.groupby("Table"):
        table_guests = group.to_dict("records")
        score = 0
        # Calculate internal table affinity
        for i in range(len(table_guests)):
            for j in range(i + 1, len(table_guests)):
                score += get_pair_affinity(
                    table_guests[i]["groups"], table_guests[j]["groups"]
                )

        avg_score = score / len(table_guests)
        print(f"Table {table_num}: Harmony Score = {avg_score:.2f}")


if __name__ == "__main__":
    # 1. Load and Prep
    df = pd.read_csv("attending_list.csv")

    # Simple replace_groups inline
    def map_groups(val):
        gs = [g.strip() for g in str(val).split(",")]
        res = [rep for match, rep in REPLACEMENTS for g in gs if g == match]
        return res if res else ["unknown"]

    df["groups"] = df["table group"].apply(map_groups)

    # 2. Run Optimization
    optimized_list = simulated_annealing(df.to_dict("records"))

    # 3. Final Scoring and Report
    for i, guest in enumerate(optimized_list):
        guest["Table"] = i // 8 + 1
        guest["Individual_Harmony"] = get_local_score(i, optimized_list)

    result_df = pd.DataFrame(optimized_list)

    # 4. Print Summary
    print("\n--- TABLE HARMONY REPORT ---")
    table_stats = result_df.groupby("Table")["Individual_Harmony"].mean()
    for table, score in table_stats.items():
        print(f"Table {table}: Avg Harmony {score:.2f}")

    # 5. Export
    result_df.drop(columns=["groups"]).to_csv(
        "final_seating_chart_with_harmony.csv", index=False
    )
    print("\nFile 'final_seating_chart_with_harmony.csv' is ready.")
