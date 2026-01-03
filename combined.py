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
import random
import math
from itertools import product
from thefuzz import process, fuzz

# --- TABLE LAYOUT CONFIGURATION ---
# Define physical table boundaries for serpentine seating
TABLE_BOUNDARIES = [32, 64, 100]  # End of table 1, end of table 2, end of table 3
# This means: Table 1 = [0-31], Table 2 = [32-63], Table 3 = [64-95]


def get_table_num(idx):
    """Get which physical table this seat is at."""
    for table_num, boundary in enumerate(TABLE_BOUNDARIES, start=1):
        if idx < boundary:
            return table_num
    return len(TABLE_BOUNDARIES) + 1  # Last table


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


# Proximity weights for serpentine banquet seating
# Even positions (0,2,4...) on one side, odd positions (1,3,5...) on other side
# Tables are separate - neighbors cannot cross table boundaries
def get_neighbor_offsets(idx, n):
    """
    Calculate neighbor positions for serpentine banquet seating.
    Returns list of (neighbor_idx, weight) tuples.
    IMPORTANT: Does not allow neighbors across table boundaries.
    """
    neighbors = []
    is_even = idx % 2 == 0
    current_table = get_table_num(idx)

    def is_valid_neighbor(neighbor_idx):
        """Check if neighbor is valid (in bounds and same table)."""
        if neighbor_idx < 0 or neighbor_idx >= n:
            return False
        return get_table_num(neighbor_idx) == current_table

    # Same side neighbors (horizontal)
    if is_even:  # Top side (even positions)
        if is_valid_neighbor(idx + 2):  # Right neighbor
            neighbors.append((idx + 2, 1.0))
        if is_valid_neighbor(idx - 2):  # Left neighbor
            neighbors.append((idx - 2, 1.0))
    else:  # Bottom side (odd positions)
        if is_valid_neighbor(idx + 2):  # Right neighbor
            neighbors.append((idx + 2, 1.0))
        if is_valid_neighbor(idx - 2):  # Left neighbor
            neighbors.append((idx - 2, 1.0))

    # Across table (vertical - strongest connection besides immediate neighbors)
    if is_even and is_valid_neighbor(idx + 1):  # Directly across
        neighbors.append((idx + 1, 1.0))
    elif not is_even and is_valid_neighbor(idx - 1):  # Directly across
        neighbors.append((idx - 1, 1.0))

    # Diagonal neighbors (weaker)
    if is_even:
        if is_valid_neighbor(idx + 3):  # Diagonal right-down
            neighbors.append((idx + 3, 0.5))
        if is_valid_neighbor(idx - 1):  # Diagonal left-down
            neighbors.append((idx - 1, 0.5))
    else:
        if is_valid_neighbor(idx + 1):  # Diagonal right-up
            neighbors.append((idx + 1, 0.5))
        if is_valid_neighbor(idx - 3):  # Diagonal left-up
            neighbors.append((idx - 3, 0.5))

    # Further diagonal (even weaker)
    if is_even:
        if is_valid_neighbor(idx + 5):
            neighbors.append((idx + 5, 0.3))
        if is_valid_neighbor(idx - 3):
            neighbors.append((idx - 3, 0.3))
    else:
        if is_valid_neighbor(idx + 3):
            neighbors.append((idx + 3, 0.3))
        if is_valid_neighbor(idx - 5):
            neighbors.append((idx - 5, 0.3))

    return neighbors


# Couple bonus - applied when partners sit at position offset 1 (next to each other)
COUPLE_BONUS = 200  # Very high to ensure couples stay together


# --- ATTENDANCE LIST GENERATION ---


def replace_groups_str(table_group):
    """Convert group names to abbreviations, return as comma-separated string."""
    group_list = [g.strip() for g in table_group.split(",")]
    new_groups = []
    for group in group_list:
        for match, rep in REPLACEMENTS:
            if group == match:
                new_groups.append(rep)
    if len(new_groups) == 0:
        raise Exception(f"NO GROUP FOUND {group_list}")
    return ",".join(new_groups)


def replace_groups_tuple(table_group):
    """Convert full group names to abbreviated codes as tuple."""
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


def get_rsvp_and_index(name, sheet4_df):
    """Find best match in sheet4 and return RSVP status + original index."""
    match, score, index = process.extractOne(
        name, sheet4_df["full name"], scorer=fuzz.token_sort_ratio
    )
    if score > 70:
        return sheet4_df.loc[index, "rsvp"], index
    return "No Match Found", None


def generate_attendance_list():
    """Generate attending list with partner detection."""
    print("=" * 60)
    print("STEP 1: GENERATING ATTENDANCE LIST")
    print("=" * 60)

    # Load CSV files
    all_df = pd.read_csv("base_table.csv")
    all_df.columns = all_df.columns.str.lower()

    sheet4_df = pd.read_csv("withjoy.csv")
    sheet4_df.columns = sheet4_df.columns.str.lower()

    # Create full name column for matching
    sheet4_df["full name"] = (
        sheet4_df["first name"].fillna("") + " " + sheet4_df["last name"].fillna("")
    ).str.strip()

    # Apply matching logic
    all_df["match_result"] = all_df["guest"].apply(
        lambda x: get_rsvp_and_index(x, sheet4_df)
    )
    all_df["rsvp_status"] = all_df["match_result"].apply(lambda x: x[0])
    all_df["sheet4_index"] = all_df["match_result"].apply(lambda x: x[1])
    all_df["groups"] = all_df["table group"].apply(replace_groups_str)

    # Filter for attendees
    attending_df = all_df[all_df["rsvp_status"] == "Will Attend"].copy()

    # Detect partners
    attending_df["partner"] = ""
    attending_df["last_name"] = attending_df["guest"].str.split().str[-1]
    attending_df = attending_df.sort_values("sheet4_index").reset_index(drop=True)

    couples_found = []
    for i in range(len(attending_df) - 1):
        curr_guest = attending_df.loc[i]
        next_guest = attending_df.loc[i + 1]

        # Check adjacency
        if pd.notna(curr_guest["sheet4_index"]) and pd.notna(
            next_guest["sheet4_index"]
        ):
            indices_consecutive = (
                next_guest["sheet4_index"] - curr_guest["sheet4_index"]
            ) == 1
        else:
            indices_consecutive = False

        # Check all three conditions
        same_last_name = curr_guest["last_name"] == next_guest["last_name"]
        same_groups = curr_guest["groups"] == next_guest["groups"]

        if indices_consecutive and same_last_name and same_groups:
            attending_df.at[i, "partner"] = next_guest["guest"]
            attending_df.at[i + 1, "partner"] = curr_guest["guest"]
            couples_found.append((curr_guest["guest"], next_guest["guest"]))

    # Save and return
    output_df = attending_df[["guest", "table group", "partner"]]
    output_df.to_csv("attending_list.csv", index=False)

    print(f"\n✓ Generated attendance list with {len(output_df)} guests")
    print(f"✓ Detected {len(couples_found)} couples:")
    for couple in couples_found:
        print(f"  • {couple[0]} ↔ {couple[1]}")

    return output_df


# --- SEATING OPTIMIZATION ---


def precalculate_all_affinities(all_group_tuples):
    """
    Precalculate affinity scores for all possible pairs of group tuples.
    Includes both direct group affinities and cohort-based penalties.
    """
    affinity_cache = {}

    for g1_tuple, g2_tuple in product(all_group_tuples, repeat=2):
        score = 0

        # Calculate direct group affinity
        for g1 in g1_tuple:
            for g2 in g2_tuple:
                if g1 == g2:
                    score += 25  # Same group bonus
                else:
                    pair = tuple(sorted((g1, g2)))
                    score += AFFINITY_MAP.get(pair, 0)

        # Apply cohort-based penalties only if NO cohorts overlap
        cohorts_1 = {GROUP_TO_COHORT.get(g) for g in g1_tuple if GROUP_TO_COHORT.get(g)}
        cohorts_2 = {GROUP_TO_COHORT.get(g) for g in g2_tuple if GROUP_TO_COHORT.get(g)}

        # Check if they share any cohorts
        shared_cohorts = cohorts_1 & cohorts_2

        if not shared_cohorts:  # Only penalize if they have NO cohorts in common
            for c1 in cohorts_1:
                for c2 in cohorts_2:
                    cohort_pair = tuple(sorted((c1, c2)))
                    cohort_penalty = COHORT_AFFINITY_MAP.get(cohort_pair, 0)
                    # Average the penalty if person belongs to multiple cohorts
                    score += cohort_penalty / (len(cohorts_1) * len(cohorts_2))

        affinity_cache[(g1_tuple, g2_tuple)] = score

    return affinity_cache


def get_local_score(idx, sequence, affinity_cache):
    """Calculate affinity for a single person based on neighbors."""
    score = 0
    n = len(sequence)
    person_groups = sequence[idx]["groups"]
    person_name = sequence[idx].get("guest", "")

    for offset, weight in get_neighbor_offsets(idx, 32):
        for direction in [offset, -offset]:
            neighbor_idx = idx + direction
            if 0 <= neighbor_idx < n:
                neighbor_groups = sequence[neighbor_idx]["groups"]
                neighbor_name = sequence[neighbor_idx].get("guest", "")

                # Base affinity score
                affinity_score = (
                    affinity_cache[(person_groups, neighbor_groups)] * weight
                )

                # Add massive bonus if they're partners sitting next to each other (offset = 1)
                if abs(offset) == 1:
                    person_partner = sequence[idx].get("partner", "")
                    if (
                        person_partner
                        and person_partner.strip().lower()
                        == neighbor_name.strip().lower()
                    ):
                        affinity_score += COUPLE_BONUS

                score += affinity_score

    return score


def calculate_delta_score(idx1, idx2, sequence, affinity_cache):
    """Calculate the change in score from swapping two people."""
    old_score = get_local_score(idx1, sequence, affinity_cache) + get_local_score(
        idx2, sequence, affinity_cache
    )

    # Account for affected neighbors
    neighbors = set()
    for idx in [idx1, idx2]:
        for offset, _ in get_neighbor_offsets(idx, 32):
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

    return (new_score - old_score) / 2


def calculate_total_score(sequence, affinity_cache):
    """Calculate total harmony score including group fragmentation penalty."""
    base_score = (
        sum(get_local_score(i, sequence, affinity_cache) for i in range(len(sequence)))
        / 2
    )

    # Add penalty for fragmented small groups
    fragmentation_penalty = calculate_fragmentation_penalty(sequence)

    return base_score - fragmentation_penalty


def calculate_fragmentation_penalty(sequence):
    """
    Penalize tables where small groups (≤8 people) are fragmented.
    Uses TABLE_BOUNDARIES configuration.
    """
    penalty = 0

    # Group people by table
    tables = {}
    for i, guest in enumerate(sequence):
        table_num = get_table_num(i)
        if table_num not in tables:
            tables[table_num] = []
        tables[table_num].append(guest)

    # Count group members per table
    group_counts = {}  # (group, table) -> count
    total_group_counts = {}  # group -> total count

    for table_num, guests in tables.items():
        for guest in guests:
            for group in guest["groups"]:
                key = (group, table_num)
                group_counts[key] = group_counts.get(key, 0) + 1
                total_group_counts[group] = total_group_counts.get(group, 0) + 1

    # Apply penalty for fragmented small groups
    for group, total_count in total_group_counts.items():
        if total_count <= 8:  # Only penalize small groups
            # Find how the group is distributed across tables
            tables_with_group = [
                (table, count)
                for (g, table), count in group_counts.items()
                if g == group
            ]

            if len(tables_with_group) > 1:  # Group is fragmented
                # Penalty increases with fragmentation severity
                for table, count in tables_with_group:
                    if count < total_count / 2:  # Minority fragments get higher penalty
                        penalty += count * 15  # 15 points per person in a fragment

    return penalty


def simulated_annealing(
    guests, affinity_cache, iterations=150000, temp=100.0, cooling=0.9999
):
    """Optimize seating arrangement using simulated annealing."""
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
        positive_score = 0
        negative_score = 0

        # Calculate internal table affinity
        for i in range(len(table_guests)):
            for j in range(i + 1, len(table_guests)):
                g1 = table_guests[i]["groups"]
                g2 = table_guests[j]["groups"]
                pair_score = affinity_cache.get((g1, g2), 0)
                score += pair_score
                if pair_score > 0:
                    positive_score += pair_score
                elif pair_score < 0:
                    negative_score += pair_score

        avg_score = score / len(table_guests) if len(table_guests) > 0 else 0
        print(
            f"\nTable {table_num}: Harmony Score = {avg_score:.2f} ({len(table_guests)} guests)"
        )
        print(f"  Positive: +{positive_score:.1f} | Negative: {negative_score:.1f}")

        # Show group composition
        all_groups = set()
        for guest in table_guests:
            all_groups.update(guest["groups"])
        print(f"  Groups: {', '.join(sorted(all_groups))}")

        # Identify cohort mix
        cohort_counts = {cohort: 0 for cohort in COHORTS.keys()}
        for guest in table_guests:
            for cohort_name, cohort_groups in COHORTS.items():
                if any(g in cohort_groups for g in guest["groups"]):
                    cohort_counts[cohort_name] += 1

        active_cohorts = {k: v for k, v in cohort_counts.items() if v > 0}
        if active_cohorts:
            print(
                f"  Cohorts: {', '.join(f'{k}={v}' for k, v in active_cohorts.items())}"
            )


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Step 1: Generate attendance list with partner detection
    attending_df = generate_attendance_list()

    print("\n" + "=" * 60)
    print("STEP 2: OPTIMIZING SEATING ARRANGEMENT")
    print("=" * 60)

    # Map groups to tuples for optimization
    attending_df["groups"] = attending_df["table group"].apply(replace_groups_tuple)

    # Get all unique group tuples
    all_group_tuples = set(attending_df["groups"].values)
    print(f"\nFound {len(all_group_tuples)} unique group combinations")

    # Precalculate affinity scores
    print("Precalculating affinity scores...")
    affinity_cache = precalculate_all_affinities(all_group_tuples)
    print(f"Precalculated {len(affinity_cache)} affinity pairs")

    # Run optimization
    print("\nStarting optimization...\n")
    optimized_list = simulated_annealing(
        attending_df.to_dict("records"), affinity_cache
    )

    # Assign tables and calculate individual harmony scores
    print("\nCalculating final harmony scores...")
    for i, guest in enumerate(optimized_list):
        table_num = i // 8 if i < 96 else 12

        guest["Table"] = table_num
        guest["Seat_Position"] = i
        guest["Individual_Harmony"] = get_local_score(i, optimized_list, affinity_cache)

    result_df = pd.DataFrame(optimized_list)

    # Print statistics
    print_table_stats(result_df, affinity_cache)

    print("\n" + "=" * 50)
    print("OVERALL STATISTICS")
    print("=" * 50)
    print(f"Total Guests: {len(result_df)}")
    print(f"Number of Tables: {result_df['Table'].max()}")
    print(f"Average Harmony Score: {result_df['Individual_Harmony'].mean():.2f}")
    print(f"Min Harmony: {result_df['Individual_Harmony'].min():.2f}")
    print(f"Max Harmony: {result_df['Individual_Harmony'].max():.2f}")

    # Check if couples stayed together
    print("\n" + "=" * 50)
    print("COUPLES CHECK")
    print("=" * 50)
    couples_ok = True
    for idx, guest in result_df.iterrows():
        partner_name = guest.get("partner", "")
        if partner_name:
            partner_row = result_df[
                result_df["guest"].str.lower() == partner_name.strip().lower()
            ]
            if not partner_row.empty:
                partner_table = partner_row.iloc[0]["Table"]
                partner_seat = partner_row.iloc[0]["Seat_Position"]
                guest_table = guest["Table"]
                guest_seat = guest["Seat_Position"]

                if guest_table == partner_table:
                    seats_adjacent = abs(guest_seat - partner_seat) == 1
                    status = "✓ (adjacent)" if seats_adjacent else "✓ (same table)"
                    print(
                        f"{status} {guest['guest']} & {partner_name}: Table {guest_table}"
                    )
                else:
                    print(
                        f"✗ WARNING: {guest['guest']} (Table {guest_table}) separated from {partner_name} (Table {partner_table})"
                    )
                    couples_ok = False

    if couples_ok:
        print("\n✓ All couples successfully seated together!")

    # Show table assignments
    groups = result_df.groupby("Table")["guest"].apply(list)
    print("\n" + "=" * 50)
    print("TABLE ASSIGNMENTS")
    print("=" * 50)
    with pd.option_context("display.max_colwidth", None):
        print(groups.to_string())

    # Export results
    output_cols = [col for col in result_df.columns if col != "groups"]
    result_df[output_cols].to_csv("final_seating_chart_with_harmony.csv", index=False)
    print("\n" + "=" * 60)
    print("✓ File 'final_seating_chart_with_harmony.csv' saved successfully!")
    print("=" * 60)
