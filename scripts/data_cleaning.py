import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

import json
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_raw_to_strings(row: pd.Series) -> Tuple[int, str, str, str, int, str]:
    org_id = row.get("org_id")
    org_name = row.get("org_name")
    org_base_name = row.get("base_name")
    org_string = (
        row.get("str_address") + " "
        + ", ".join(row.get("industry_description").values() if "industry_description" in row else [])
        + (" " + row.get("prh_description") if "prh_description" in row else "")
        + (" " + row.get("linkedin_description") if "linkedin_description" in row else "")
    )
    change_item_id = row.get("change_item_id")
    change_item_string = row.get("change_item_str")
    return org_id, org_name, org_base_name, org_string, change_item_id, change_item_string


def combine_change_item_contents(row: pd.Series) -> Optional[str]:
    if not (pd.isna(row.get("title")) or pd.isna(row.get("content"))):
        return row.get("title") + "\n" + row.get("content")
    elif not pd.isna(row.get("title")):
        return row.get("title")
    elif not pd.isna(row.get("content")):
        return row.get("content")
    else:
        return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sql_result", type=Path, required=False, default="data/25k_final.json",
                        help="The path to the data extracted from the DB")
    parser.add_argument("--output_file", type=Path, required=False, default="data/cleaned_data.json",
                        help="The file where to save the cleaned data")
    parser.add_argument("--inference", action="store_true", required=False, default=False,
                        help="Specify if the data is meant for use in inference or training")
    args = parser.parse_args()

    # Read data from DB dump and convert industry description to JSON
    logging.info("Reading data from JSON...")
    raw_data = pd.read_json(args.sql_result).convert_dtypes()
    raw_data["industry_description"] = raw_data.apply(lambda row: json.loads(row["industry_description"]), axis=1)

    # Drop NaNs in industry_description
    raw_data = raw_data.dropna(subset=["industry_description"])

    # Strip Oy and other type of company denominations out of the organisation name
    logging.info("Stripping away company denominations...")
    companies_types = [
        "Oy", "Oyj", "Ay", "Ltd", "Ab", "Ky",                       # Finnish
        "AS", "ASA", "ANS", "DA", "KS", "BA", "HF", "IKS", "SA",    # Norwegian
    ]
    regex_string = r"\b(?:" + r"|".join(companies_types) + r")\b"
    raw_data["base_name"] = (
        raw_data["org_name"].str.replace(pat=regex_string, repl="", regex=True, flags=re.IGNORECASE).str.strip()
    )

    logging.info("Combine fields making up the change item content...")
    raw_data["change_item_str"] = raw_data.apply(combine_change_item_contents, axis=1)

    if not args.inference:
        logging.info("Dropping rows where base name is not included in the change item content...")
        raw_data = raw_data[raw_data.apply(lambda row: row["base_name"] in row["change_item_str"], axis=1)]

    logging.info("Parse raw data into DataFrame cleaned format...")
    data = pd.DataFrame(
        raw_data.apply(parse_raw_to_strings, axis=1).tolist(),
        columns=["org_id", "org_name", "org_base_name", "org_str", "change_item_id", "change_item_str"]
    )

    if not args.inference:
        # Remove NaNs and duplicate change_items
        data = data[~data["change_item_str"].isna()]
        data = data.drop_duplicates(subset="change_item_id")

        logging.info("Filter out orgs with too many change items")
        org_counts = data.org_id.value_counts()
        org_indices = org_counts[org_counts < 9].index.tolist()
        uncommon_orgs = data[data.org_id.isin(org_indices)].reset_index(drop=True)
        common_orgs = data[~data.org_id.isin(org_indices)].groupby("org_id").sample(8).reset_index(drop=True)
        final_dataset = pd.concat([uncommon_orgs, common_orgs]).reset_index(drop=True)
    else:
        final_dataset = data

    logging.info(f"Final dataset size: {len(final_dataset)}")

    logging.info("Writing final datasets to file...")
    final_dataset.to_json(args.output_file, orient="records")

    if not args.inference:
        # # Split dataset in fit and test, making sure we don't have leaked orgs from fit to test
        # test_org_ids = final_dataset["org_id"].value_counts().sample(frac=0.2)
        # test_df = final_dataset[final_dataset["org_id"].isin(test_org_ids.index)]
        test_df = final_dataset.sample(frac=0.2)
        fit_df = final_dataset[~(final_dataset.index.isin(test_df.index))]

        assert len(fit_df) + len(test_df) == len(final_dataset)

        # Split fit into and val
        val_df = fit_df.sample(frac=0.15)
        train_df = fit_df[~(fit_df.index.isin(val_df.index))]

        train_df.to_json("data/train.json", orient="records")
        val_df.to_json("data/val.json", orient="records")
        test_df.to_json("data/test.json", orient="records")

    logging.info("Done ✔️")
