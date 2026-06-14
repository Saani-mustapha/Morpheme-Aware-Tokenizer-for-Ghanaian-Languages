import argparse
import csv
from pathlib import Path


def extract_local_column(input_csv: str, output_txt: str, column_name: str = "local"):
    input_path = Path(input_csv)
    output_path = Path(output_txt)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile, \
         output_path.open("w", encoding="utf-8", newline="\n") as outfile:

        reader = csv.DictReader(infile)

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found. Available columns: {reader.fieldnames}"
            )

        for row in reader:
            text = row.get(column_name, "").strip()

            if not text:
                continue

            outfile.write(text + "\n")
            count += 1

    print(f"Extracted {count} lines to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the local language column from a CSV file into a plain text file."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output TXT file"
    )

    parser.add_argument(
        "--column",
        default="local",
        help="Name of the local-language column. Default: local"
    )

    args = parser.parse_args()

    extract_local_column(args.input, args.output, args.column)