import os
import shutil
import argparse
from glob import glob

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--librimix_dir", type=str, default=None, help="Path to librispeech root directory"
)


def main(args):
    librimix_dir = args.librimix_dir
    create_local_metadata(librimix_dir)


def MergeEnvFile(newfile, oldfile):
    with open(newfile, "a") as new_file, open(oldfile, "r") as old_file:
        old_file_lines = old_file.readlines()
        for idx, line in enumerate(old_file_lines):
            if idx == 0:
                continue
            new_file.write(line.strip())
            new_file.write("\n")


def create_local_metadata(librimix_dir):

    md_dirs = [f for f in glob(os.path.join(librimix_dir, "*/*/*")) if f.endswith("metadata")]
    for md_dir in md_dirs:
        md_files = [f for f in os.listdir(md_dir) if f.startswith("mix")]
        for md_file in md_files:
            subset = md_file.split("_")[1]
            local_path = os.path.join(
                "data", os.path.relpath(md_dir, librimix_dir), subset
            ).replace("/metadata", "")
            os.makedirs(local_path, exist_ok=True)
            print(local_path)
            print(md_file)
            if os.path.exists(local_path):
                MergeEnvFile(os.path.join(local_path, md_file), os.path.join(md_dir, md_file))
            else:
                shutil.copy(os.path.join(md_dir, md_file), local_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
