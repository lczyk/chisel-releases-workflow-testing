import subprocess as sub
from pathlib import Path
import re


GIT_PATH = "git"
REPO_ROOT = Path(__file__).parent
SLICE_PATH = "./slices/"  # Path to search for SDF yaml file in


# TODO: add argparse with format flag


def get_branches():
    """Get all branch names from local repo"""
    raw_output = sub.check_output(
        [GIT_PATH, "branch", "-l", "--format=%(refname:short)"], cwd=REPO_ROOT
    )
    branches = raw_output.decode("utf-8").splitlines()
    filtered_branches = set(filter(lambda s: s != "main", branches))

    return filtered_branches


def get_slices(branch):
    """Get slice yaml filenames from local branch"""
    raw_output = sub.check_output(
        [GIT_PATH, "show", f"{branch}:{SLICE_PATH}"], cwd=REPO_ROOT
    )
    slices = raw_output.decode("utf-8").splitlines()
    slices = slices[2:]
    return slices


# get all the branches in this repo, then filter out non-release ones
branches = get_branches()
branches_filtered = list(
    sorted(filter(lambda name: re.match("^ubuntu-", name), branches))
)

# generate dictionary mapping SDF to individual releases as sets
slice_map = dict()
for bf in branches_filtered:
    for sl in get_slices(bf):

        # if the slice isn't included it in the map, add it
        if sl not in slice_map:
            slice_map[sl] = set()

        slice_map[sl].add(bf)


# make a new table for the output, add a header
# slice name, ubuntu-X, ubuntu-Y, ubuntu-Z ...
availability_table = [["slice"] + branches_filtered]

# use pathlib to extract the slice name from filename
availability_table += [
    [Path(k).stem] + [b in slice_map[k] for b in branches_filtered]
    for k in sorted(slice_map.keys())
]

# TODO: add csv (from module), json and markdown support
for row in availability_table:
    print(",".join(map(str, row)))
