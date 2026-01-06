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
import functools
import math
import cProfile
import pstats
import copy
from itertools import product
from thefuzz import process, fuzz

# --- TABLE LAYOUT CONFIGURATION ---
TABLE_BOUNDARIES = [40, 66, 98]

# --- FORCED SEATING ---
FORCED_SEATS = {
    #    "Drew Habel": 40,
    #    "Katie Habel": 41,
    "Ali de Jong": 44,
    "Matt Habel": 45,
    #    "Joe Habel": 42,
    #    "Olivia Frymark": 43,
    #    "Annabel Wang": 46,
    #    "Phyllis Luedke": 46,
    #    "Helen Clark": 47,
    #    "Adam Collins": 48,
    # "Lori de Jong": 66,
    # "Mark McNeill": 67,
    # "Mary Kay Habel": 0,
    # "Rich Habel": 1,
}


def get_table_num(idx):
    for table_num, boundary in enumerate(TABLE_BOUNDARIES, start=1):
        if idx < boundary:
            return table_num
    return len(TABLE_BOUNDARIES) + 1


def get_sub_table_id(idx):
    banquet = get_table_num(idx)
    table_start = 0 if banquet == 1 else TABLE_BOUNDARIES[banquet - 2]
    table_end = TABLE_BOUNDARIES[banquet - 1]
    total_seats = table_end - table_start
    local_idx = idx - table_start
    num_full_clusters = total_seats // 8
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

AFFINITY_SCORES = {
    ("acf", "af"): 120,
    ("acf", "aff"): 100,
    ("mcf", "dan"): 100,
    ("mff", "mcf"): 75,
    ("mff", "dan"): 75,
    ("acf", "mcf"): 100,
    ("malays", "mcf"): 40,
    ("malays", "dan"): 50,
    ("cf", "sff"): 125,
    ("cf", "ut"): 40,
    ("cf", "cornell"): 40,
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
    ("family_friends", "young_friends"): -700,
    ("hs", "family"): -200,
    ("hs", "young_friends"): 100,
}
COHORT_AFFINITY_MAP = {tuple(sorted(k)): v for k, v in COHORT_AFFINITY.items()}
GROUP_TO_COHORT = {g: c for c, gs in COHORTS.items() for g in gs}

BLOCK_BONUS = 3000
SAME_GROUP_BONUS = 300
FRAGMENTATION_PENALTY = 900


@functools.lru_cache(maxsize=None)
def get_neighbor_offsets(idx, n):
    neighbors = []
    x, y = idx // 2, idx % 2
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
        nx, ny = n_idx // 2, n_idx % 2
        distance = abs(x - nx) + abs(y - ny)
        if distance > 0:
            neighbors.append((n_idx, 1.5 / distance))
    return neighbors


def precalculate_all_affinities(all_group_tuples):
    cache = {}
    for g1_t, g2_t in product(all_group_tuples, repeat=2):
        score = 0
        for g1 in g1_t:
            for g2 in g2_t:
                ts = tuple(sorted((g1, g2)))
                if ts in AFFINITY_MAP:
                    score += AFFINITY_MAP[ts]
                elif g1 == g2:
                    score += SAME_GROUP_BONUS
        c1s = {GROUP_TO_COHORT.get(g) for g in g1_t if g in GROUP_TO_COHORT}
        c2s = {GROUP_TO_COHORT.get(g) for g in g2_t if g in GROUP_TO_COHORT}
        if not (c1s & c2s):
            for c1, c2 in product(c1s, c2s):
                score += COHORT_AFFINITY_MAP.get(tuple(sorted((c1, c2))), 0)
        cache[(g1_t, g2_t)] = score
    return cache


def get_local_harmony(idx, sequence, affinity_cache):
    """Calculate harmony only for one seat and its neighbors."""
    n = len(sequence)
    p1 = sequence[idx]
    score = 0
    for n_idx, weight in get_neighbor_offsets(idx, n):
        if weight < 0.025:
            continue
        p2 = sequence[n_idx]
        score += affinity_cache[(p1["groups"], p2["groups"])] * weight
        if p1.get("block_id") is not None and p1.get("block_id") == p2.get("block_id"):
            if weight >= 0.75:
                score += BLOCK_BONUS
    return score


def get_fragmentation_penalty(group_dist):
    """Calculate penalty based on existing group distribution tracker."""
    penalty = 0
    for grp, distribution in group_dist.items():
        total_count = sum(distribution.values())
        num_tables = len(distribution)
        if num_tables > 1 and total_count <= 12:
            penalty += (num_tables - 1) * FRAGMENTATION_PENALTY
    return penalty


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
            "meal": joy_df.loc[index, "meal / wedding"],
            "j_idx": index,
            "last_name": str(joy_df.loc[index, "last name"]).strip().lower(),
        }
    return {"rsvp": "No Match", "j_idx": None, "last_name": "", "meal": None}


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
    base_df["meal"] = rsvp_data.apply(lambda x: x["meal"])
    base_df["last_name"] = rsvp_data.apply(lambda x: x["last_name"])
    attending = base_df[base_df["rsvp"] == "Will Attend"].copy().reset_index(drop=True)
    attending["block_id"] = None
    attending["is_forced"] = attending["guest"].isin(FORCED_SEATS.keys())

    SPECIAL_COUPLES = [
        ("Joe Habel", "Olivia Frymark"),
        ("Taylor Zhao", "Janette Tang"),
        ("Brian Hoang", "Kat Madrinan"),
        ("Casey Deyo", "Victoria Morgan"),
        ("Tyler Philolius", "Javi Hernandez"),
        ("Andrew Clissold", "Paige Seibold"),
        ("Lauren Witt", "Steve Wrobel"),
        ("Helen Clark", "Adam Collins"),
        ("Sean Kirmani", "Charlotte O'Keefe Stralka"),
        ("Ari Tooch", "Gabe Diamond"),
        ("Dillon Short", "Shannon Short"),
        ("Lilly Elam", "Holly SanMiguel"),
        ("Hunter Stephan", "Jenna Stephan"),
        ("Nick Singley", "Monica Raysberg"),
        ("Maria Mu", "Shlagha Karjee"),
        ("Hannah Kitto", "Abby Kitto"),
        ("Boston Hukari", "Heather Hukari"),
    ]

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
        curr, j = [i], i + 1
        while j < len(attending):
            p1, p2 = attending.iloc[j - 1], attending.iloc[j]
            if p2["block_id"] is not None:
                break
            if (
                pd.notna(p1["j_idx"])
                and pd.notna(p2["j_idx"])
                and abs(p1["j_idx"] - p2["j_idx"]) == 1
            ):
                if (
                    p1["table group"] == p2["table group"]
                    and p1["last_name"] == p2["last_name"]
                    and p1["last_name"] != ""
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


def simulated_annealing(guests, affinity_cache, iterations=2500000, patience=75000):
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

    for i in range(n):
        if sequence[i] is None:
            sequence[i] = remaining_guests.pop()

    # INITIAL SCORING
    harmony_score = 0
    for i in range(n):
        harmony_score += get_local_harmony(i, sequence, affinity_cache)
    harmony_score /= 2

    group_dist = {}
    for i, g in enumerate(sequence):
        t = get_table_num(i)
        for grp in g["groups"]:
            group_dist.setdefault(grp, {}).setdefault(t, 0)
            group_dist[grp][t] += 1

    penalty_score = get_fragmentation_penalty(group_dist)
    current_score = harmony_score - penalty_score

    best_seq, best_score = list(sequence), current_score
    temp_start, temp_end = 200.0, 0.01
    cooling = (temp_end / temp_start) ** (1.0 / iterations)
    temp = temp_start

    iters_without_improvement = 0
    improvement_threshold = 0.0001  # 0.01% improvement required to reset patience

    print(
        f"Starting simulation for {iterations} iterations with Block Swapping (Patience: {patience})..."
    )

    for i in range(iterations):
        # Pick first seat
        idx1 = random.randint(0, n - 1)
        while sequence[idx1].get("is_forced"):
            idx1 = random.randint(0, n - 1)

        # Decide if we move a block or a single guest
        bid1 = sequence[idx1].get("block_id")
        indices1 = [idx1]
        if bid1 is not None and random.random() < 0.5:
            indices1 = [
                idx for idx, g in enumerate(sequence) if g.get("block_id") == bid1
            ]
            if any(sequence[idx].get("is_forced") for idx in indices1):
                indices1 = [idx1]

        # Pick second seat/block
        valid_indices2 = False
        while not valid_indices2:
            idx2 = random.randint(0, n - len(indices1))
            indices2 = list(range(idx2, idx2 + len(indices1)))
            # Ensure no overlaps and no forced seats
            if not any(idx in indices1 for idx in indices2) and not any(
                sequence[idx].get("is_forced") for idx in indices2
            ):
                valid_indices2 = True

        # --- SNAPSHOT FOR REVERSION ---
        # Storing a deep copy of the distribution state is faster than manual logic-heavy revert
        dist_snapshot = copy.deepcopy(group_dist)

        # --- DELTA CALCULATION ---
        affected = set(indices1) | set(indices2)
        for idx in list(affected):
            for n_idx, _ in get_neighbor_offsets(idx, n):
                affected.add(n_idx)

        old_local_harmony = sum(
            get_local_harmony(idx, sequence, affinity_cache) for idx in affected
        )

        # UPDATE GROUP DIST (PRE-SWAP)
        for idx in indices1 + indices2:
            g = sequence[idx]
            t = get_table_num(idx)
            for grp in g["groups"]:
                group_dist[grp][t] -= 1
                if group_dist[grp][t] == 0:
                    del group_dist[grp][t]

        # DO SWAP
        for k in range(len(indices1)):
            sequence[indices1[k]], sequence[indices2[k]] = (
                sequence[indices2[k]],
                sequence[indices1[k]],
            )

        # UPDATE GROUP DIST (POST-SWAP)
        for idx in indices1 + indices2:
            g = sequence[idx]
            t = get_table_num(idx)
            for grp in g["groups"]:
                group_dist.setdefault(grp, {}).setdefault(t, 0)
                group_dist[grp][t] += 1

        new_local_harmony = sum(
            get_local_harmony(idx, sequence, affinity_cache) for idx in affected
        )
        new_penalty = get_fragmentation_penalty(group_dist)

        delta_harmony = (new_local_harmony - old_local_harmony) / 2
        new_score = harmony_score + delta_harmony - new_penalty

        # Acceptance check
        if new_score > current_score or (
            temp > 0 and random.random() < math.exp((new_score - current_score) / temp)
        ):
            # Check for meaningful improvement (0.1%)
            if (
                best_score == 0
                or (new_score - best_score) / max(1, abs(best_score))
                >= improvement_threshold
            ):
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

            current_score = new_score
            harmony_score += delta_harmony
            penalty_score = new_penalty

            if current_score > best_score:
                best_seq, best_score = list(sequence), current_score
        else:
            # --- REVERT ---
            iters_without_improvement += 1
            # Simple restoration from snapshot
            group_dist = dist_snapshot
            for k in range(len(indices1)):
                sequence[indices1[k]], sequence[indices2[k]] = (
                    sequence[indices2[k]],
                    sequence[indices1[k]],
                )

        temp *= cooling
        if i % 25000 == 0:
            print(
                f"Iter {i:,} | Score: {current_score:.2f} | Best: {best_score:.2f} | Temp: {temp:.4f}"
            )

        if iters_without_improvement >= patience:
            print(
                f"Early convergence at iteration {i:,} (no improvement >= {improvement_threshold*100}% for {patience} steps)."
            )
            break

    return best_seq


def main():
    att_df = generate_attendance_list()
    att_df["groups"] = att_df["table group"].apply(replace_groups_tuple)
    affinity_cache = precalculate_all_affinities(set(att_df["groups"].values))
    final = simulated_annealing(
        att_df.to_dict("records"),
        affinity_cache,
    )

    # Final calculations for table metrics
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

    res = pd.DataFrame(final)[
        [
            "guest",
            "Table",
            "Harmony",
            "groups",
            "block_id",
            "is_forced",
            "meal",
        ]
    ]
    res.to_csv("final_seating_chart_optimized.csv", index=False)
    print(f"✓ Saved to 'final_seating_chart_optimized.csv'")

    unique_tables = sorted(res["Table"].unique())

    # --- DETAILED TABLE BREAKDOWN ---
    print("\n" + "=" * 60)
    print("DETAILED SEATING ASSIGNMENTS BY TABLE")
    print("=" * 60)

    for table_id in unique_tables:
        table_guests = res[res["Table"] == table_id].sort_values(
            by="Harmony", ascending=False
        )
        table_total = table_guests["Harmony"].sum()

        print(f"\n>>> {table_id} (Total Harmony: {table_total:,.2f})")
        print("-" * 45)
        for _, guest in table_guests.iterrows():
            forced_tag = "[FORCED] " if guest["is_forced"] else ""
            groups_tag = f"({', '.join(guest['groups'])})"
            print(
                f" {guest['Harmony']:7.2f} | {forced_tag}{guest['guest']:25} {groups_tag}"
            )

    # --- CONDENSED OVERVIEW ---
    print("\n" + "=" * 60)
    print("CONDENSED TABLE OVERVIEW")
    print("=" * 60)
    for table_id in unique_tables:
        table_guests = res[res["Table"] == table_id]["guest"].tolist()
        print(f"{table_id:10} | {', '.join(table_guests)}")

    # --- FINAL STATS SUMMARY ---
    print("\n" + "=" * 40)
    print("FINAL SEATING SUMMARY STATISTICS")
    print("=" * 40)

    total_harmony = (
        res["Harmony"].sum() / 2
    )  # Total score is sum of individual harmonies divided by 2
    print(f"Total System Harmony: {total_harmony:,.2f}")

    print("\nGuest Highlights:")
    print(
        f"- Highest individual harmony: {res['Harmony'].max():.2f} ({res.loc[res['Harmony'].idxmax(), 'guest']})"
    )
    print(
        f"- Lowest individual harmony:  {res['Harmony'].min():.2f} ({res.loc[res['Harmony'].idxmin(), 'guest']})"
    )
    print("=" * 40 + "\n")


if __name__ == "__main__":
    # Using cProfile to analyze performance
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # Save the full profile results to /tmp/profile.stats
    profile_path = "/tmp/profile.stats"
    profiler.dump_stats(profile_path)
    print(f"✓ Profiling data written to '{profile_path}'")

    # Print top 20 functions by total time to console
    print("\nPERFORMANCE PROFILE (TOP 20 FUNCTIONS):")
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(20)
