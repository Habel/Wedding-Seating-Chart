# /// script
# requires-python = ">=3.11"
# dependencies = [
#      "pandas",
#      "numpy",
#      "thefuzz",
#      "python-Levenshtein",
# ]
# ///

import pandas as pd
import numpy as np
import random
import math
from itertools import product
from thefuzz import process, fuzz

# --- TABLE LAYOUT CONFIGURATION ---
# Banquet 1: 0-39 (40 seats), Banquet 2: 40-65 (26 seats), Banquet 3: 66-97 (32 seats)
TABLE_BOUNDARIES = [40, 66, 98]

# --- FORCED SEATING ---
# Map Guest Name -> Exact Seat Index
FORCED_SEATS = {
    "Drew Habel": 40,
    "Katie Habel": 41,
    "Ali de Jong": 45,
    "Matt Habel": 46,
    "Joe Habel": 42,
    "Olivia Frymark": 43,
    "Annabel Wang": 44,
}


def get_table_num(idx):
    """Determine banquet table number (1-3) based on seat index."""
    for table_num, boundary in enumerate(TABLE_BOUNDARIES, start=1):
        if idx < boundary:
            return table_num
    return len(TABLE_BOUNDARIES) + 1


def get_sub_table_id(idx):
    """Determine the specific 8-person cluster (e.g., '1-0', '1-1', etc)."""
    banquet = get_table_num(idx)

    # Get the boundaries of this specific banquet table
    table_start = 0 if banquet == 1 else TABLE_BOUNDARIES[banquet - 2]
    table_end = TABLE_BOUNDARIES[banquet - 1]

    total_seats = table_end - table_start
    local_idx = idx - table_start

    # Calculate how many full 8-person clusters exist
    num_full_clusters = total_seats // 8

    # The last cluster ID is num_full_clusters - 1
    # If we are in or beyond the last full cluster, cap the ID
    sub = local_idx // 8
    if sub >= num_full_clusters:
        sub = num_full_clusters - 1

    return f"{banquet}-{max(0, sub)}"


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

SPECIAL_COUPLES = [
    ("Joe Habel", "Olivia Frymark"),
    ("Taylor Zhao", "Janette Tang"),
    ("Brian Hoang", "Kat Madrinan"),
    ("Casey Deyo", "Victoria Morgan"),
    ("Tyler Philolius", "Javi Hernandez"),
    ("Andrew Clissold", "Paige Seibold"),
    ("Lauren Witt", "Steve Wrobel"),
    ("Helen Clark", "Adam Collins"),
    ("Sean Kirmao", "Charlotte O'Keefe Stralka"),
    ("Ari Tooch", "Gabe Diamond"),
    ("Dillon Short", "Shannon Barry (Short)"),
    ("Lilly Elam", "Holly SanMiguel"),
    ("Hunter Stephan", "Jenna Stephan"),
    ("Nick Singley", "Monica Raysberg"),
    ("Maria Mu", "Shlagha Karjee"),
    ("Hannah Kitto", "Abby Kitto"),
]

AFFINITY_SCORES = {
    ("acf", "af"): 80,
    ("acf", "aff"): 80,
    ("mcf", "dan"): 80,
    ("mff", "mcf"): 100,
    ("acf", "mcf"): 50,
    ("malays", "mcf"): 80,
    ("malays", "dan"): 40,
    ("cf", "sff"): 125,
    ("cornell", "mhs"): 40,
    ("ut", "ahs"): 60,
    ("cornell", "ut"): 50,
    ("sff", "cornell"): 50,
    ("sff", "ut"): 50,
    ("mhs", "sff"): 50,
    ("ahs", "sff"): 50,
}
AFFINITY_MAP = {tuple(sorted(k)): v for k, v in AFFINITY_SCORES.items()}

COHORTS = {
    "family": ["acf", "mcf", "af", "dan", "malays"],
    "family_friends": ["aff", "mff"],
    "young_friends": ["cf", "sff", "ut", "cornell"],
    "hs": ["ahs", "mhs"],
}

COHORT_AFFINITY = {
    ("family", "young_friends"): -500,
    ("family_friends", "young_friends"): -300,
    ("hs", "family"): -200,
    ("hs", "young_friends"): 50,
}
COHORT_AFFINITY_MAP = {tuple(sorted(k)): v for k, v in COHORT_AFFINITY.items()}
GROUP_TO_COHORT = {g: c for c, gs in COHORTS.items() for g in gs}

BLOCK_BONUS = 1500
SAME_GROUP_BONUS = 300
FRAGMENTATION_PENALTY = 900

# --- GEOMETRY & SCORING ---


def get_neighbor_offsets(idx, n):
    neighbors = []
    current_table = get_table_num(idx)
    x = idx // 2
    y = idx % 2

    table_start = 0
    for boundary in [0] + TABLE_BOUNDARIES:
        if idx >= boundary:
            table_start = boundary
        else:
            break

    table_end = n
    for boundary in TABLE_BOUNDARIES:
        if idx < boundary:
            table_end = min(n, boundary)
            break

    for n_idx in range(table_start, table_end):
        if n_idx == idx:
            continue
        nx = n_idx // 2
        ny = n_idx % 2
        distance = abs(x - nx) + abs(y - ny)
        if distance > 0:
            weight = 1.5 / distance
            neighbors.append((n_idx, weight))
    return neighbors


def precalculate_all_affinities(all_group_tuples):
    cache = {}
    for g1_t, g2_t in product(all_group_tuples, repeat=2):
        score = 0
        for g1 in g1_t:
            for g2 in g2_t:
                ts = tuple(sorted((g1, g2)))
                if ts in AFFINITY_MAP:
                    score += AFFINITY_MAP.get(ts, 0)
                elif g1 == g2:
                    score += SAME_GROUP_BONUS

                else:
                    score += AFFINITY_MAP.get(tuple(sorted((g1, g2))), 0)

        c1s = {GROUP_TO_COHORT.get(g) for g in g1_t if g in GROUP_TO_COHORT}
        c2s = {GROUP_TO_COHORT.get(g) for g in g2_t if g in GROUP_TO_COHORT}

        if not (c1s & c2s):
            for c1, c2 in product(c1s, c2s):
                score += COHORT_AFFINITY_MAP.get(tuple(sorted((c1, c2))), 0)
        cache[(g1_t, g2_t)] = score
    return cache


def calculate_total_score(sequence, affinity_cache):
    n = len(sequence)
    total_score = 0

    for i in range(n):
        p1 = sequence[i]
        for neighbor_idx, weight in get_neighbor_offsets(i, n):
            p2 = sequence[neighbor_idx]
            total_score += affinity_cache[(p1["groups"], p2["groups"])] * weight
            if (
                p1.get("block_id") == p2.get("block_id")
                and p1.get("block_id") is not None
            ):
                if weight >= 0.75:
                    total_score += BLOCK_BONUS

    penalty = 0
    group_dist = {}
    for i, g in enumerate(sequence):
        t = get_table_num(i)
        for grp in g["groups"]:
            group_dist.setdefault(grp, {}).setdefault(t, 0)
            group_dist[grp][t] += 1

    for grp, distribution in group_dist.items():
        if len(distribution) > 1 and sum(distribution.values()) <= 12:
            penalty += (len(distribution) - 1) * FRAGMENTATION_PENALTY

    return (total_score / 2) - penalty


# --- DATA PREPROCESSING & SIMULATED ANNEALING ---


def replace_groups_tuple(table_group):
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


def get_rsvp_info(name, joy_df):
    match, score, index = process.extractOne(
        name, joy_df["full name"], scorer=fuzz.token_sort_ratio
    )
    if score > 75:
        return {
            "rsvp": joy_df.loc[index, "rsvp"],
            "j_idx": index,
            "last_name": str(joy_df.loc[index, "last name"]).strip().lower(),
        }
    return {"rsvp": "No Match", "j_idx": None, "last_name": ""}


def generate_attendance_list():
    base_df = pd.read_csv("base_table.csv")
    base_df.columns = base_df.columns.str.lower()
    joy_df = pd.read_csv("withjoy.csv")
    joy_df.columns = joy_df.columns.str.lower()
    joy_df["full name"] = (
        joy_df["first name"].fillna("") + " " + joy_df["last name"].fillna("")
    ).str.strip()
    rsvp_data = base_df["guest"].apply(lambda x: get_rsvp_info(x, joy_df))
    base_df["rsvp"] = rsvp_data.apply(lambda x: x["rsvp"])
    base_df["j_idx"] = rsvp_data.apply(lambda x: x["j_idx"])
    base_df["last_name"] = rsvp_data.apply(lambda x: x["last_name"])
    attending = base_df[base_df["rsvp"] == "Will Attend"].copy().reset_index(drop=True)
    attending["block_id"] = None
    attending["is_forced"] = attending["guest"].isin(FORCED_SEATS.keys())

    block_counter = 0
    for couple in SPECIAL_COUPLES:
        indices = attending.index[attending["guest"].isin(couple)].tolist()
        if len(indices) >= 2:
            for idx in indices:
                attending.at[idx, "block_id"] = block_counter
            block_counter += 1
    i = 0
    while i < len(attending):
        if attending.iloc[i]["block_id"] is not None:
            i += 1
            continue
        curr = [i]
        j = i + 1
        while j < len(attending):
            p1, p2 = attending.iloc[j - 1], attending.iloc[j]
            if p2["block_id"] is not None:
                break
            if (
                pd.notna(p1["j_idx"])
                and pd.notna(p2["j_idx"])
                and abs(p1["j_idx"] - p2["j_idx"]) == 1
            ):
                if p1["table group"] == p2["table group"] and (
                    p1["last_name"] == p2["last_name"] and p1["last_name"] != ""
                ):
                    curr.append(j)
                    j += 1
                else:
                    break
            else:
                break
        if len(curr) > 1:
            for idx in curr:
                attending.at[idx, "block_id"] = block_counter
            block_counter += 1
        i = j
    return attending


def simulated_annealing(guests, affinity_cache, iterations=500000):
    n = len(guests)
    sequence = [None] * n
    remaining_guests = []

    for g in guests:
        if g["guest"] in FORCED_SEATS:
            target_idx = FORCED_SEATS[g["guest"]]
            if target_idx < n:
                sequence[target_idx] = g
            else:
                remaining_guests.append(g)
        else:
            remaining_guests.append(g)

    random.shuffle(remaining_guests)
    for i in range(n):
        if sequence[i] is None:
            sequence[i] = remaining_guests.pop()

    def get_block(seq, b_id):
        return [idx for idx, g in enumerate(seq) if g.get("block_id") == b_id]

    current_score = calculate_total_score(sequence, affinity_cache)
    best_seq, best_score = list(sequence), current_score

    temp_start, temp_end = 200.0, 0.01
    cooling = (temp_end / temp_start) ** (1.0 / iterations)
    temp = temp_start
    best_score_last_changed = 0
    patience = 40000

    for i in range(iterations):
        if i - best_score_last_changed > patience:
            print(f"Exiting early at iteration {i} due to convergence.")
            break

        idx1 = random.randint(0, n - 1)
        while sequence[idx1].get("is_forced"):
            idx1 = random.randint(0, n - 1)

        p1 = sequence[idx1]
        old_seq = None

        if p1.get("block_id") is not None and random.random() < 0.7:
            b_indices = get_block(sequence, p1["block_id"])
            if any(sequence[idx].get("is_forced") for idx in b_indices):
                idx2 = random.randint(0, n - 1)
                while sequence[idx2].get("is_forced") or idx1 == idx2:
                    idx2 = random.randint(0, n - 1)
                sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]
            else:
                available_indices = [
                    j
                    for j, g in enumerate(sequence)
                    if not g.get("is_forced") and j not in b_indices
                ]
                if available_indices:
                    idx2 = random.choice(available_indices)
                    sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]
        else:
            idx2 = random.randint(0, n - 1)
            while sequence[idx2].get("is_forced") or idx1 == idx2:
                idx2 = random.randint(0, n - 1)
            sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]

        new_score = calculate_total_score(sequence, affinity_cache)
        if new_score > current_score or (
            temp > 0 and random.random() < math.exp((new_score - current_score) / temp)
        ):
            current_score = new_score
            if current_score > best_score:
                best_seq, best_score = list(sequence), current_score
                best_score_last_changed = i
        else:
            sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]

        temp *= cooling
        if i % 50000 == 0:
            print(
                f"Iter {i:,} | Score: {current_score:.2f} | Best: {best_score:.2f} | Temp: {temp:.4f}"
            )

    return best_seq


if __name__ == "__main__":
    att_df = generate_attendance_list()
    att_df["groups"] = att_df["table group"].apply(replace_groups_tuple)

    # Debug: Print detected blocks
    print("=" * 50 + "\nDETECTED SEATING BLOCKS\n" + "=" * 50)
    for b_id, group in att_df[att_df["block_id"].notna()].groupby("block_id"):
        print(f"Block {b_id}: {', '.join(group['guest'].tolist())}")
    print("=" * 50 + "\n")

    affinity_cache = precalculate_all_affinities(set(att_df["groups"].values))
    final = simulated_annealing(att_df.to_dict("records"), affinity_cache)

    for i, g in enumerate(final):
        g["Table"] = f"Table {get_sub_table_id(i)}"
        h = sum(
            affinity_cache[(g["groups"], final[nb]["groups"])] * w
            for nb, w in get_neighbor_offsets(i, len(final))
        )
        if g.get("block_id") is not None:
            for nb, w in get_neighbor_offsets(i, len(final)):
                if final[nb].get("block_id") == g["block_id"] and w >= 0.75:
                    h += BLOCK_BONUS
        g["Harmony"] = h

    res = pd.DataFrame(final)[["guest", "block_id", "is_forced", "Table", "Harmony"]]
    print("\n" + "=" * 50 + "\nTABLE ANALYSIS\n" + "=" * 50)
    for t, group in res.groupby("Table"):
        print(f"[{t}] Avg Harmony: {group['Harmony'].mean():.2f}")
        print(f"  Guests: {', '.join(group['guest'].tolist())}\n")

    res.to_csv("final_seating_chart_with_harmony.csv", index=False)
    print(f"✓ Saved to 'final_seating_chart_with_harmony.csv'")
