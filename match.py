# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "thefuzz",
#     "python-Levenshtein",
# ]
# ///

import pandas as pd
from thefuzz import process, fuzz


# 1. Load your CSV files
all_df = pd.read_csv("base_table.csv")  # Exported "All" tab
sheet4_df = pd.read_csv("withjoy.csv")  # Exported "Sheet4"

# 2. Create a "Full Name" column in Sheet4 for matching
sheet4_df["full name"] = (
    sheet4_df["first name"].fillna("") + " " + sheet4_df["last name"].fillna("")
)
sheet4_df["full name"] = sheet4_df["full name"].str.strip()


# 3. Define a function to find the best match and return the RSVP status
def get_rsvp_status(name):
    # Find the best match for 'name' in the Sheet4 'Full Name' list
    # limit=1 returns the single best match
    match, score, index = process.extractOne(
        name, sheet4_df["full name"], scorer=fuzz.token_sort_ratio
    )

    # Only return status if the match is high confidence (e.g., > 70%)
    if score > 70:
        return sheet4_df.loc[index, "rsvp"]
    return "No Match Found"


replacements = [
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


def replace_groups(table_group):
    group_list = [g.strip() for g in table_group.split(",")]
    new_groups = []
    for group in group_list:
        for match, rep in replacements:
            if group == match:
                new_groups.append(rep)
    if len(new_groups) == 0:
        raise Exception(f"NO GROUP FOUND ${group_list}")
    return new_groups


# 4. Apply the matching logic to the "All" table
all_df["rsvp_status"] = all_df["guest"].apply(get_rsvp_status)
all_df["groups"] = all_df["table group"].apply(replace_groups)

# 5. Filter for only those who "Will Attend"
attending_df = all_df[all_df["rsvp_status"] == "Will Attend"][["guest", "table group"]]

# print(attending_df)

# 6. Save to a new CSV
attending_df.to_csv("attending_list.csv", index=False)

print("Success! Filtered list created with fuzzy matching.")
