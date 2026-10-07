import os
import sys
import argparse
from pathlib import Path


def aggregate_python_files(target_directory, output_file_name="aggregated_code.txt"):
    target_path = Path(target_directory)
    output_path = target_path / output_file_name

    # Exclude the script itself and common virtual environment/cache folders
    exclude_dirs = {".git", "__pycache__", "venv", ".venv", "env", "node_modules"}

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"Aggregate text in folder ''\n")
        # Recursively walk through directories
        for root, dirs, files in os.walk(target_path):
            # Modifying dirs in-place lets os.walk skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if (
                    file.endswith(".py") or file.endswith(".csv")
                ) and file != output_file_name:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(target_path)

                    # Write a clear visual header for each file
                    outfile.write(f"\n{'='*80}\n")
                    outfile.write(f"FILE: {relative_path}\n")
                    outfile.write(f"{'='*80}\n\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                        outfile.write("\n")  # Add spacing after file content
                    except Exception as e:
                        outfile.write(f"[ERROR READING FILE: {e}]\n")

    print(f"Successfully aggregated files into: {output_path.resolve()}")


# Run the aggregator in the current directory
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate files in this directory")
    parser.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="The target directory to scan (default: current directory '.')",
    )
    args = parser.parse_args()

    aggregate_python_files(args.dir)
