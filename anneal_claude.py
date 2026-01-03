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
from itertools import product

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

AFFINITY_SCORES = {
    # --- POSITIVE AFFINITIES (People who should sit together) ---
    ("acf", "af"): 15,
    ("af", "aff"): 10,
    ("mcf", "dan"): 15,
    ("mff", "mcf"): 10,
    ("acf", "mcf"): 15,
    ("malays", "mcf"): 10,
    ("malays", "dan"): 5,
    ("cf", "sff"): 15,
    ("cornell", "mhs"): 5,
    ("ut", "ahs"): 10,
    ("cornell", "ut"): 5,
    ("sff", "cornell"): 5,
    ("sff", "ut"): 5,
}

AFFINITY_MAP = {tuple(sorted(k)): v for k, v in AFFINITY_SCORES.items()}

# Define cohorts and their relationships
COHORTS = {
    "family": ["acf", "mcf", "af", "dan", "malays"],
    "family_friends": ["aff", "mff"],
    "young_friends": ["cf", "sff", "ut", "cornell"],
    "hs": ["ahs", "mhs"],
}

# Cohort-to-cohort penalties (negative = discourage mixing)
COHORT_AFFINITY = {
    ("family", "young_friends"): -5,  # Family vs young friends - age gap
    ("family_friends", "young_friends"): -3,  # Family friends vs young friends
}

COHORT_AFFINITY_MAP = {tuple(sorted(k)): v for k, v in COHORT_AFFINITY.items()}

# Build reverse lookup: group -> cohort
GROUP_TO_COHORT = {}
for cohort_name, groups in COHORTS.items():
    for group in groups:
        GROUP_TO_COHORT[group] = cohort_name

# Proximity weights: position offset -> weight
PROXIMITY_WEIGHTS = {1: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 2: 0.5, 6: 0.5, 7: 0.2}


def replace_groups(table_group):
    """Convert full group names to abbreviated codes."""
    if pd.isna(table_group):
        return tuple(["unknown"])

    group_list = [g.strip() for g in str(table_group).split(",")]
    new_groups = []

    for group in group_list:
        for match, rep in REPLACEMENTS:
            if group == match:
                new_groups.append(rep)
                break

    return tuple(new_groups) if new_groups else tuple(["unknown"])


def precalculate_all_affinities(all_group_tuples):
    """
    Precalculate affinity scores for all possible pairs of group tuples.
    This is much faster than memoization for repeated lookups.
    """
    affinity_cache = {}

    for g1_tuple, g2_tuple in product(all_group_tuples, repeat=2):
        score = 0
        for g1 in g1_tuple:
            for g2 in g2_tuple:
                if g1 == g2:
                    score += 25  # Same group bonus
                else:
                    pair = tuple(sorted((g1, g2)))
                    score += AFFINITY_MAP.get(pair, 0)

        affinity_cache[(g1_tuple, g2_tuple)] = score

    return affinity_cache


def get_local_score(idx, sequence, affinity_cache):
    """Calculate affinity for a single person based on neighbors."""
    score = 0
    n = len(sequence)
    person_groups = sequence[idx]["groups"]

    for offset, weight in PROXIMITY_WEIGHTS.items():
        for direction in [offset, -offset]:
            neighbor_idx = idx + direction
            if 0 <= neighbor_idx < n:
                neighbor_groups = sequence[neighbor_idx]["groups"]
                score += affinity_cache[(person_groups, neighbor_groups)] * weight

    return score


def calculate_delta_score(idx1, idx2, sequence, affinity_cache):
    """
    Calculate the change in score from swapping two people.
    This is MUCH faster than recalculating the entire sequence.
    """
    old_score = get_local_score(idx1, sequence, affinity_cache) + get_local_score(
        idx2, sequence, affinity_cache
    )

    # Also need to account for affected neighbors
    neighbors = set()
    for idx in [idx1, idx2]:
        for offset in list(PROXIMITY_WEIGHTS.keys()):
            for direction in [offset, -offset]:
                neighbor_idx = idx + direction
                if 0 <= neighbor_idx < len(sequence) and neighbor_idx not in [
                    idx1,
                    idx2,
                ]:
                    neighbors.add(neighbor_idx)

    for neighbor_idx in neighbors:
        old_score += get_local_score(neighbor_idx, sequence, affinity_cache)

    # Swap
    sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]

    # Calculate new score
    new_score = get_local_score(idx1, sequence, affinity_cache) + get_local_score(
        idx2, sequence, affinity_cache
    )
    for neighbor_idx in neighbors:
        new_score += get_local_score(neighbor_idx, sequence, affinity_cache)

    # Swap back
    sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]

    # Divide by 2 since we count each pair twice
    return (new_score - old_score) / 2


def calculate_total_score(sequence, affinity_cache):
    """Calculate total harmony score for the entire sequence."""
    return (
        sum(get_local_score(i, sequence, affinity_cache) for i in range(len(sequence)))
        / 2
    )


def simulated_annealing(
    guests, affinity_cache, iterations=150000, temp=100.0, cooling=0.9999
):
    """
    Optimize seating arrangement using simulated annealing.
    Uses delta scoring for massive speedup.
    """
    current_seq = list(guests)
    random.shuffle(current_seq)
    current_score = calculate_total_score(current_seq, affinity_cache)

    best_seq = list(current_seq)
    best_score = current_score

    print(f"Starting Score: {current_score:.2f}")
    print(f"Initial Temperature: {temp:.2f}")

    for i in range(iterations):
        # 80% local swaps, 20% random jumps
        idx1 = random.randint(0, len(current_seq) - 1)
        if random.random() < 0.8:
            idx2 = (idx1 + random.randint(-10, 10)) % len(current_seq)
        else:
            idx2 = random.randint(0, len(current_seq) - 1)

        if idx1 == idx2:
            continue

        # Calculate delta instead of full recalculation
        delta = calculate_delta_score(idx1, idx2, current_seq, affinity_cache)

        # Accept if improvement or probabilistically accept if worse
        if delta > 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
            current_seq[idx1], current_seq[idx2] = current_seq[idx2], current_seq[idx1]
            current_score += delta

            if current_score > best_score:
                best_score = current_score
                best_seq = list(current_seq)

        temp *= cooling

        if i % 25000 == 0:
            print(
                f"Iteration {i:,}... Score: {current_score:.2f} (Best: {best_score:.2f}, Temp: {temp:.4f})"
            )

    print(f"\nFinal Temperature: {temp:.4f}")
    return best_seq


def print_table_stats(result_df, affinity_cache):
    """Print harmony statistics for each table."""
    print("\n" + "=" * 50)
    print("TABLE HARMONY REPORT")
    print("=" * 50)

    for table_num, group in result_df.groupby("Table"):
        table_guests = group.to_dict("records")
        score = 0

        # Calculate internal table affinity
        for i in range(len(table_guests)):
            for j in range(i + 1, len(table_guests)):
                g1 = table_guests[i]["groups"]
                g2 = table_guests[j]["groups"]
                score += affinity_cache.get((g1, g2), 0)

        avg_score = score / len(table_guests) if len(table_guests) > 0 else 0
        print(
            f"Table {table_num}: Harmony Score = {avg_score:.2f} ({len(table_guests)} guests)"
        )

        # Show group composition
        all_groups = set()
        for guest in table_guests:
            all_groups.update(guest["groups"])
        print(f"  Groups: {', '.join(sorted(all_groups))}")


if __name__ == "__main__":
    print("Loading guest list...")
    df = pd.read_csv("attending_list.csv")

    # Map groups using corrected logic
    df["groups"] = df["table group"].apply(replace_groups)

    # Get all unique group tuples
    all_group_tuples = set(df["groups"].values)

    print(f"Found {len(all_group_tuples)} unique group combinations")
    print("Precalculating affinity scores...")

    # Precalculate ALL possible affinity scores
    affinity_cache = precalculate_all_affinities(all_group_tuples)
    print(f"Precalculated {len(affinity_cache)} affinity pairs")

    # Run optimization
    print("\nStarting optimization...\n")
    optimized_list = simulated_annealing(df.to_dict("records"), affinity_cache)

    # Assign tables and calculate individual harmony scores
    print("\nCalculating final harmony scores...")
    for i, guest in enumerate(optimized_list):
        guest["Table"] = i // 8 + 1
        guest["Seat_Position"] = i % 8 + 1
        guest["Individual_Harmony"] = get_local_score(i, optimized_list, affinity_cache)

    result_df = pd.DataFrame(optimized_list)

    # Print summary statistics
    print_table_stats(result_df, affinity_cache)

    print("\n" + "=" * 50)
    print("OVERALL STATISTICS")
    print("=" * 50)
    print(f"Total Guests: {len(result_df)}")
    print(f"Number of Tables: {result_df['Table'].max()}")
    print(f"Average Harmony Score: {result_df['Individual_Harmony'].mean():.2f}")
    print(f"Min Harmony: {result_df['Individual_Harmony'].min():.2f}")
    print(f"Max Harmony: {result_df['Individual_Harmony'].max():.2f}")

    groups = result_df.groupby("Table")["guest"].apply(list)
    with pd.option_context("display.max_colwidth", None):
        print(f"Groups: {groups.to_string()}")

    # Export results
    output_cols = [col for col in result_df.columns if col != "groups"]
    result_df[output_cols].to_csv("final_seating_chart_with_harmony.csv", index=False)
    print("\n✓ File 'final_seating_chart_with_harmony.csv' saved successfully!")
