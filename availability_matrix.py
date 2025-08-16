# spell-checker: words levelname
import subprocess as sub
from pathlib import Path
import re
import argparse
import logging
import tempfile
import shutil
from dataclasses import dataclass, replace
import json
from typing import no_type_check
import os

__version__ = "0.3.1"


## CLI #########################################################################


@dataclass
class Args:
    repo: str
    log_level: str = "info"
    check_repo: bool = True
    check_prs: bool = False
    check_branches: bool = False
    version_filter: str = "*"
    package_filter: str = "*"
    branch_filter: str = "*"
    pr_filter: str = "*"
    temp_dir: Path | None = None


def parse_args() -> Args:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check a repo for missing forward porting of packages to Ubuntu releases."
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
        help="Show script version and exit.",
    )
    parser.add_argument(
        "repo",
        type=str,
        help="The repository to check. This can be a local path or a remote URL. "
        "If a local path, it should point to a Git repository. "
        "If a remote URL, it should be a valid Git repository URL (e.g., "
        "'github.com:canonical/chisel-releases')",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warn", "error", "critical"],
        help="Set the log level.",
    )

    parser.add_argument(
        "--check-repo",
        type=lambda x: x.lower() in ("true", "t", "1"),
        default=True,
        help="If true, check the state of the repository pointed to by <repo>. "
        "Defaults to true.",
    )
    parser.add_argument(
        "--check-prs",
        type=lambda x: x.lower() in ("true", "t", "1"),
        default=False,
        help="If true, check the state of the PRs to the <repo>. "
        "If <repo> is a remote URL, this will fetch all open PRs. "
        "If <repo> is a local path, this will attempt to find the remote URL "
        "based on the url of the origin remote. Defaults to false.",
    )
    parser.add_argument(
        "--check-branches",
        type=lambda x: x.lower() in ("true", "t", "1"),
        default=False,
        help="If true, treat any branches in the repository as if they were "
        " pull requests, and check them for forward porting. If --check-prs is true, "
        "they will be cross-checked with the branches (and vice versa). "
        "This is useful for testing local branches that are not yet pushed to the "
        "remote repository. Defaults to false.",
    )
    parser.add_argument(
        "--version-filter",
        "--vf",
        type=str,
        default="*",
        help="Specify the Ubuntu versions to check for forward porting. "
        "This should be a comma-separated list of versions, e.g., '22.04,24.04'. "
        "If not specified, it will default to all the versions found in the repository.",
    )
    parser.add_argument(
        "--package-filter",
        "--pf",
        type=str,
        default="*",
        help="Specify the packages to check for forward porting. "
        "This should be a comma-separated list of package names, e.g., 'foo,bar'. "
        "If not specified, it will default to all the packages found in every branch.",
    )
    parser.add_argument(
        "--pull-request-filter",
        "--prf",
        type=str,
        default="*",
        help="Specify the PRs to check for forward porting. "
        "This should be a comma-separated list of PR numbers, or usernames, "
        "e.g., '123,456,username'. If not specified, it will default to all the "
        "PRs found in the repository. Note that only the PRs which are 1) open, "
        "2) have a base branch starting with 'ubuntu-', and 3) match the "
        "version filter will be considered.",
    )
    parser.add_argument(
        "--branch-filter",
        "--bf",
        type=str,
        default="*",
        help="Specify the branches to check for forward porting. "
        "This should be a comma-separated list of branch names, e.g., 'feature-foo,bugfix-bar'. "
        "If not specified, it will default to all the local branches found in the repository. "
        "This is only used if --check-branches is true.",
    )

    parsed = parser.parse_args()
    return Args(
        repo=parsed.repo,
        log_level=parsed.log_level,
        check_repo=parsed.check_repo,
        check_prs=parsed.check_prs,
        check_branches=parsed.check_branches,
        version_filter=parsed.version_filter,
        package_filter=parsed.package_filter,
        pr_filter=parsed.pull_request_filter,
        branch_filter=parsed.branch_filter,
    )


GIT_PATH = os.environ.get("GIT_PATH", "git")


def git(command: tuple[str, ...], cwd: str | Path | None = None) -> str:
    """Run a git command and return the output."""
    try:
        return (
            sub.check_output(
                [GIT_PATH, *command],
                cwd=str(cwd),
            )
            .decode("utf-8")
            .strip()
        )
    except sub.CalledProcessError as e:
        logging.error("Git command failed: %s", e)
        raise RuntimeError(f"Git command failed: {e}") from e


def setup_logging(log_level: str):
    """Set up logging based on the log level."""

    _logger = logging.getLogger()
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter: type[logging.Formatter] = logging.Formatter
    try:
        # Try to use colorlog for colored output
        import colorlog  # type: ignore

        fmt = re.sub(r"%\(levelname\)s", "%(log_color)s%(levelname)s%(reset)s", fmt)
        formatter = colorlog.ColoredFormatter  # type: ignore
    except ImportError:
        pass

    handler.setFormatter(formatter(fmt, datefmt))  # type: ignore
    _logger.addHandler(handler)
    _logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def setup_temp_dir() -> Path:
    name = tempfile.mktemp(prefix="availability_matrix_")
    temp_dir = Path(name)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
        logging.debug("Created default clone directory: %s", temp_dir)
    else:
        logging.debug("Using existing clone directory: %s", temp_dir)
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        logging.debug("Cleaned up existing clone directory: %s", temp_dir)

    return temp_dir


def looks_like_ulr(url: str) -> bool:
    if "github.com" in url:
        return True
    if "https://" in url or "http://" in url:
        return True
    return False


@dataclass
class Work:
    """Data class to represent the work done by the script."""

    # Slices in the base branch
    slices_by_branch: dict[str, set[str]]
    # Slices in the PR branches keyed by (head, base)
    slices_by_pr: dict["PR", set[str]]
    # Base repository directory
    base_repo_dir: Path
    # PRs repository directory (if different from base_repo_dir)
    prs_repo_dir: Path | None


def get_slices(repo: str, branch: str) -> set[str]:
    """Get the slices for a given branch."""
    slices = git(
        ("show", f"{branch}:./slices/"),
        cwd=Path(repo),
    )
    slices = slices.splitlines()
    slices = [s.strip() for s in slices if s.strip()]
    slices = [s.removesuffix(".yaml") for s in slices if s.endswith(".yaml")]
    return set(slices)


def pull_remote_repo(args: Args) -> Work:
    logging.info("Using remote repository URL: %s", args.repo)

    # Clone the repo to the temp directory
    assert args.temp_dir is not None, "Temporary directory must be set."
    git(
        ("clone", "--quiet", args.repo, str(args.temp_dir / "repo")),
        cwd=args.temp_dir.parent,
    )
    logging.info("Cloned repository to: %s", args.temp_dir / "repo")
    args.repo = str(args.temp_dir / "repo")

    # Also pull any branches that might be relevant
    remote_branches = git(
        ("branch", "-r", "--format=%(refname:short)"),
        cwd=args.repo,
    )
    remote_branches = remote_branches.splitlines()
    remote_branches = [b for b in remote_branches if b.startswith("origin/")]

    # Filter down to branches that start with "ubuntu-"
    branches = [
        b.replace("origin/", "")
        for b in remote_branches
        if b.startswith("origin/ubuntu-")
    ]
    logging.debug(
        "Found %d remote branches: %s",
        len(branches),
        ", ".join(b.removeprefix("ubuntu-") for b in branches),
    )

    # filter based on the versions filter
    if args.version_filter != "*":
        regex = re.compile(rf"^ubuntu-({'|'.join(args.version_filter.split(','))})")
        logging.debug("Filtering branches with regex: %s", regex.pattern)
        branches = [b for b in branches if regex.match(b)]

    # NOTE: We delay pulling of the branches until we have pulled the
    # branches for the PRs, such that, if we are only checking PRs,
    # we can additionally filter down the ubuntu branches to only those
    # which are the base branches of the PRs.

    # If we are checking PRs, we need to fetch them as well
    slices_by_pr: dict["PR", set[str]] = {}
    prs_repo_dir: Path | None = None
    if args.check_prs:
        slices_by_pr = pull_prs(args)
        # same repo as the base repo
        prs_repo_dir = Path(args.repo)

    slices_by_branches: dict["Branch", set[str]] = {}
    branches_repo_dir: Path | None = None
    if args.check_branches:
        non_ubuntu_branches = [b for b in branches if not b.startswith("ubuntu-")]
        logging.debug(
            "Found %d non-ubuntu branches: %s",
            len(non_ubuntu_branches),
            ", ".join(non_ubuntu_branches),
        )

        # filter branches based on the branch filter
        if args.branch_filter != "*":
            regex = re.compile(rf"^({'|'.join(args.branch_filter.split(','))})")
            logging.debug("Filtering branches with regex: %s", regex.pattern)
            branches = [b for b in branches if regex.match(b)]

        # Pull the non-ubuntu branches if we are checking branches
        for branch in non_ubuntu_branches:
            logging.info("Pulling branch: %s", branch)
            git(
                ("checkout", "--quiet", "--track", f"origin/{branch}", "-b", branch),
                cwd=args.repo,
            )
            git(
                ("pull", "--quiet", "origin", branch),
                cwd=args.repo,
            )

        # Now we have the non-ubuntu branches pulled, we can get the slices for each branch
        for branch in non_ubuntu_branches:
            slices = get_slices(args.repo, branch)
            slices_by_branches[branch] = slices
            logging.debug("Found %d slices in branch '%s'", len(slices), branch)

        # same repo as the base repo
        branches_repo_dir = Path(args.repo)

    # Filter branches based on the PRs we have pulled
    if args.check_prs and not args.check_repo:
        pr_bases = set(pr.base.ref for pr in slices_by_pr.keys())
        if len(pr_bases) < len(branches):
            logging.info(
                "Filtering branches to only those that are base branches of PRs: %s",
                ", ".join(pr_bases),
            )
            branches = [b for b in branches if b in pr_bases]

    # Filter branches based on the branches we have pulled
    if args.check_branches and not args.check_repo:
        branch_bases = set(slices_by_branches.keys())
        if len(branch_bases) < len(branches):
            logging.info(
                "Filtering branches to only those that are branches we have pulled: %s",
                ", ".join(branch_bases),
            )
            branches = [b for b in branches if b in branch_bases]

    # # Pull all the matching branches
    # for branch in branches:
    #     logging.info("Pulling branch: %s", branch)
    #     git(
    #         ("checkout", "--quiet", "--track", f"origin/{branch}", "-b", branch),
    #         cwd=args.repo,
    #     )
    #     git(
    #         ("pull", "--quiet", "origin", branch),
    #         cwd=args.repo,
    #     )

    # slices_by_branch: dict[str, set[str]] = {}
    # for branch in branches:
    #     slices = get_slices(args.repo, branch)
    #     slices_by_branch[branch] = slices
    #     logging.debug("Found %d slices in branch '%s'", len(slices), branch)
    slices_by_branch: dict[str, set[str]] = {}
    for branch in branches:
        with CheckoutTemporaryBranch(
            "origin",
            branch,
            args.repo,
        ) as branch_name:
            slices = get_slices(args.repo, branch_name)
            slices_by_branch[branch] = slices
            logging.debug("Found %d slices in branch '%s'", len(slices), branch)

    return Work(
        slices_by_branch=slices_by_branch,
        slices_by_pr=slices_by_pr,
        base_repo_dir=Path(args.repo),
        prs_repo_dir=prs_repo_dir,
    )

from contextlib import contextmanager
import uuid


def remote_url_to_human_readable_infix(remote: str) -> str:
    """Convert a remote URL to a human-readable infix."""
    out = ""
    if "github.com" in remote:
        # For GitHub URLs, we can use the owner/repo format
        parts = remote.split("/")
        if len(parts) >= 2:
            out = f"{parts[-2]}_{parts[-1]}"
    else:
        # Not a github URL! unsure whether this is supported yet, but let's
        # not fall over and try ot make something out of it
        out = remote.replace("https://", "").replace("http://", "")

    # final cleanup
    out = re.sub(r"[^a-zA-Z0-9_.-]", "_", out)
    out = out.strip("_")
    if not out:
        out = "xxx"
    return out


@contextmanager
def AddTemporaryRemote(
    remote_url: str,
    repo: str,
    remote_name: str | None = None,
):
    """Context manager to add a temporary remote to the repository."""
    if remote_name is None:
        remote_name = "_{name_prefix}_{human_readable}_{uuid}".format(
            name_prefix="temp-remote",
            human_readable=remote_url_to_human_readable_infix(remote_url),
            uuid=uuid.uuid4().hex[:8],
        )

    try:
        git(("remote", "add", remote_name, remote_url), cwd=repo)
        logging.debug(
            "Added temporary remote '%s' under the name '%s'",
            remote_url,
            remote_name,
        )
        yield remote_name
    finally:
        git(("remote", "remove", remote_name), cwd=repo)
        logging.debug("Removed temporary remote '%s'", remote_name)


def remote_branch_name_to_human_readable(remote_branch_name: str) -> str:
    """Convert a remote branch name to a human-readable format."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", remote_branch_name).strip("_")


@contextmanager
def CheckoutTemporaryBranch(
    remote_name: str,
    remote_branch_name: str,
    repo: str,
    local_branch_name: str | None = None,
):
    """Context manager to fetch a temporary branch from a remote."""
    if local_branch_name is None:
        local_branch_name = "_{name_prefix}_{human_readable}_{uuid}".format(
            name_prefix="temp-branch",
            human_readable=remote_branch_name_to_human_readable(remote_branch_name),
            uuid=uuid.uuid4().hex[:8],
        )

    try:
        git(("fetch", "--quiet", remote_name, remote_branch_name), cwd=repo)
        logging.debug(
            "Fetched branch '%s' from remote '%s'", remote_branch_name, remote_name
        )
        git(
            (
                "checkout",
                "--quiet",
                "--track",
                f"{remote_name}/{remote_branch_name}",
                "-b",
                local_branch_name,
            ),
            cwd=repo,
        )
        logging.debug(
            "Checked out branch '%s' from remote '%s' under the name '%s'",
            remote_branch_name,
            remote_name,
            local_branch_name,
        )
        yield local_branch_name
    finally:
        # Remove the temporary branch
        git(("checkout", "main"), cwd=repo)  # Switch back to main branch
        git(
            ("branch", "-D", local_branch_name), cwd=repo
        )  # Delete the temporary branch
        logging.debug("Deleted temporary branch '%s'", local_branch_name)

def pull_prs(
    args: Args,
    *,
    override_repo_dir: Path | None = None,
    additional_version_filter: str | None = None,
) -> dict["PR", set[str]]:
    if override_repo_dir is not None:
        args = replace(args, repo=str(override_repo_dir))

    remote_url = git(
        ("config", "--get", "remote.origin.url"),
        cwd=args.repo,
    ).removesuffix(".git")
    logging.debug("Remote URL for PRs: %s", remote_url)

    assert "github.com" in remote_url, (
        "The repository is not hosted on GitHub, so PR checking is not supported."
    )

    # Fetch all PRs
    prs = get_all_prs(remote_url)

    pr_filter_regex: re.Pattern[str] | None = None
    # TODO: This filtering can be done much better
    # For example filter pr's by user and *then* by number doe snot work atm
    if args.pr_filter != "*":
        pr_filter_regex = re.compile(rf"^({'|'.join(args.pr_filter.split(','))})$")
        logging.debug("Filtering PRs with regex: %s", pr_filter_regex.pattern)

    version_filter_regex: re.Pattern[str] | None = None
    if args.version_filter != "*":
        version_filter_regex = re.compile(
            rf"^ubuntu-({'|'.join(args.version_filter.split(','))})"
        )
        logging.debug(
            "Filtering PR base refs with regex: %s", version_filter_regex.pattern
        )

    additional_version_filter_regex: re.Pattern[str] | None = None
    if additional_version_filter:
        additional_version_filter_regex = re.compile(
            rf"^ubuntu-({'|'.join(additional_version_filter.split(','))})"
        )
        logging.debug(
            "Filtering PR base refs with additional regex: %s",
            additional_version_filter_regex.pattern,
        )

    filtered_prs: list[PR] = []
    for pr in prs:
        # Check whether the base_ref is a valid branch in the repo
        if not pr.base.ref.startswith("ubuntu-"):
            if pr.base.ref != "main":
                logging.warning(
                    "Weird base ref '%s' in PR '%s'. Skipping it.",
                    pr.base.ref,
                    pr.url,
                )
                continue

        # Check whether the base_ref is one we care about
        # We might have filtered these earlier
        if version_filter_regex and not version_filter_regex.match(pr.base.ref):
            logging.debug(
                "Skipping PR #%d because base ref '%s' does not match version filter.",
                pr.number,
                pr.base.ref,
            )
            continue

        if (
            additional_version_filter_regex
            and not additional_version_filter_regex.match(pr.base.ref)
        ):
            logging.debug(
                "Skipping PR #%d because base ref '%s' does not match additional version filter.",
                pr.number,
                pr.base.ref,
            )
            continue

        # Check whether the PR matches the filter
        if pr_filter_regex and (
            not pr_filter_regex.match(str(pr.number))
            and not pr_filter_regex.match(pr.user)
        ):
            logging.debug(
                "Skipping PR #%d by '%s' because it does not match the PR filter.",
                pr.number,
                pr.user,
            )
            continue

        filtered_prs.append(pr)

    # Group PRs by remote
    prs_by_remote_url: dict[str, set[PR]] = {}

    for pr in filtered_prs:
        prs_by_remote_url.setdefault(pr.head.repo.url, set()).add(pr)

        logging.debug(
            "Found PR by %s from '%s' at '%s' to '%s'",
            pr.user,
            pr.head.ref,
            pr.head.repo.url,
            pr.base.ref,
        )

    slices_by_pr: dict[PR, set[str]] = {}

    for remote, prs in prs_by_remote_url.items():
        with AddTemporaryRemote(remote, args.repo) as remote_name:
            for pr in prs:
                with CheckoutTemporaryBranch(
                    remote_name,
                    pr.head.ref,
                    args.repo,
                ) as branch_name:
                    slices = get_slices(args.repo, branch_name)
                    slices_by_pr[pr] = slices
                    logging.debug(
                        "Found %d slices in PR '%s' from '%s' to '%s'",
                        len(slices),
                        pr.url,
                        pr.head.ref,
                        pr.base.ref,
                    )

    logging.info(
        "Found %d open PRs in the repository '%s'.",
        len(filtered_prs),
        remote_url,
    )

    return slices_by_pr


# geturl from https://github.com/lczyk/geturl 0.4.4
def geturl(url: str) -> tuple[int, bytes]:
    """Make a GET request to a URL and return the response and status code."""

    import urllib
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url) as r:
            code = r.getcode()
            res = r.read()

    except urllib.error.HTTPError as e:
        code = e.code
        res = e.read()

    assert isinstance(code, int), "Expected code to be int."
    assert isinstance(res, bytes), "Expected response to be bytes."

    return code, res


# from typing import Protocol, TYPE_CHECKING

# class Head(Protocol):
#     """Protocol for a head branch
# ...


# @dataclass
# class Branch: ...


@dataclass(frozen=True, unsafe_hash=True)
class Repo:
    """Data class to represent a Git repository."""

    name: str  # Name of the repository
    url: str  # URL of the repository
    owner: str  # Owner of the repository

    @no_type_check
    @classmethod
    def from_payload(cls, data: dict) -> "Repo":
        return cls(
            name=data["name"],
            url=data["html_url"],
            owner=data["owner"]["login"],
        )


@dataclass(frozen=True, unsafe_hash=True)
class Commit:
    """Data class to represent a Git commit."""

    commit: str  # SHA of the commit
    ref: str
    repo: Repo

    @no_type_check
    @classmethod
    def from_payload(cls, data: dict) -> "Commit":
        return cls(
            commit=data["sha"],
            ref=data["ref"],
            repo=Repo.from_payload(data["repo"]),
        )


@dataclass(frozen=True, unsafe_hash=True)
class PR:
    """Data class to represent a Pull Request."""

    url: str  # URL of the PR, e.g. http
    number: int  # number of the PR, e.g #601
    title: str  # title of the PR

    user: str  # user who created the PR (usually, but not necessarily, the author)

    head: Commit
    base: Commit

    state: str  # open, closed, merged, draft etc

    @no_type_check
    @classmethod
    def from_payload(cls, data: dict) -> "PR":
        head = Commit.from_payload(data["head"])
        base = Commit.from_payload(data["base"])

        return cls(
            url=data["html_url"],
            number=data["number"],
            title=data["title"],
            user=data["user"]["login"],
            state=data["state"],
            head=head,
            base=base,
        )

    # @property
    # def head_repo_full_name(self) -> str:
    #     """Return the full name of the head repository."""
    #     return f"{self.user}/{self.head_repo}"

    # def remote_name(self, limit: int = 100) -> str:
    #     """Create a valid remote name from the remote URL."""
    #     remote_name = self.remote.replace("https://", "").replace("http://", "")
    #     remote_name = remote_name.replace("github.com", "")
    #     remote_name = remote_name.replace("/", "_").replace(":", "_")
    #     remote_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", remote_name)
    #     if limit > 0 and len(remote_name) > limit:
    #         remote_name = remote_name[:limit]
    #     if not remote_name:
    #         raise ValueError("Remote name cannot be empty.")
    #     remote_name = remote_name.strip("_")
    #     remote_name = remote_name.strip("-")
    #     if not remote_name:
    #         raise ValueError("Remote name cannot be empty after stripping.")
    #     return remote_name

    # def branch_name(self) -> str:
    #     """Create a branch name for the PR."""
    #     # Use the remote name to avoid conflicts with local branches
    #     remote_name = self.remote_name()
    #     # Use the PR number and base ref to create a unique branch name
    #     return f"{remote_name}/PR-{self.number}-{self.base_ref}"


def get_all_prs(remote_url: str, per_page: int = 100) -> list[PR]:
    """Fetch all PRs from the remote repository using the GitHub API."""
    page = 1
    api_url = (
        remote_url.replace("github.com", "api.github.com/repos").rstrip("/")
        + f"/pulls?state=open&per_page={per_page}"
    )
    logging.info("Fetching PRs")
    logging.debug("Fetching PRs from GitHub API: %s", api_url)

    results = []
    while True:
        code, result = geturl(api_url + f"&page={page}")
        if code != 200:
            raise RuntimeError(
                f"Failed to fetch PRs from '{remote_url}'. HTTP status code: {code}"
            )
        parsed_result = json.loads(result.decode("utf-8"))
        assert isinstance(parsed_result, list), "Expected response to be a list of PRs."
        results.extend(parsed_result)  # type: ignore

        # Check if there are more pages
        if len(parsed_result) < page:  # type: ignore
            break

        # Update the API URL for the next page
        page += 1

    return [PR.from_payload(pr) for pr in results]  # type: ignore


def setup_local_repo(args: Args) -> Work:
    """Set up the local repository and return the work object."""
    logging.info("Using local repository path: %s", args.repo)

    if not Path(args.repo).is_dir():
        raise FileNotFoundError(f"Local repository path '{args.repo}' does not exist.")

    # Check if the repo is a valid Git repository
    if not (Path(args.repo) / ".git").is_dir():
        raise ValueError(f"The path '{args.repo}' is not a valid Git repository.")

    # Get the names of all branches in the local repository
    branches = git(
        ("branch", "--format=%(refname:short)"),
        cwd=args.repo,
    ).splitlines()

    branches = [b for b in branches if b.startswith("ubuntu-")]

    # Filter branches based on the version filter
    if args.version_filter != "*":
        regex = re.compile(rf"^ubuntu-({'|'.join(args.version_filter.split(','))})")
        logging.debug("Filtering branches with regex: %s", regex.pattern)
        branches = [b for b in branches if regex.match(b)]

    logging.debug("Found %d branches: %s", len(branches), ", ".join(branches))

    # If we are checking PRs we need to set up a separate repo clone and fetch them
    # there, since we dont want to mess with the local repo
    slices_by_pr: dict[PR, set[str]] = {}
    prs_repo_dir: Path | None = None
    if args.check_prs:
        logging.info("Checking PRs in the local repository")
        remote_url = git(
            ("config", "--get", "remote.origin.url"),
            cwd=args.repo,
        ).removesuffix(".git")
        if not remote_url:
            raise ValueError(
                "The local repository does not have a remote URL set. "
                "Please set the remote URL using 'git remote add origin <url>'."
            )
        logging.info("Using remote repository URL: %s", remote_url)

        # Clone the remote repository to a temporary directory
        assert args.temp_dir is not None, "Temporary directory must be set."

        prs_repo_dir = args.temp_dir / "repo"
        git(
            ("clone", "--quiet", remote_url, str(prs_repo_dir)),
            cwd=args.temp_dir.parent,
        )

        logging.info("Cloned remote repository to: %s", prs_repo_dir)

        # In this case we already have base branches and these are the ones we
        # will check the PRs against. This means we can do some additional
        # filtering of the PRs based on the branches we have.

        additional_version_filter: str = ",".join(
            b.removeprefix("ubuntu-") for b in branches if b.startswith("ubuntu-")
        )

        # Pull the PRs from the remote repository
        slices_by_pr = pull_prs(
            args,
            override_repo_dir=prs_repo_dir,
            additional_version_filter=additional_version_filter,
        )

    # Finally get the slices for each branch
    slices_by_branch: dict[str, set[str]] = {}
    for branch in branches:
        slices = get_slices(args.repo, branch)
        slices_by_branch[branch] = slices
        logging.debug("Found %d slices in branch '%s'", len(slices), branch)

    return Work(
        slices_by_branch=slices_by_branch,
        slices_by_pr=slices_by_pr,
        base_repo_dir=Path(args.repo),
        prs_repo_dir=prs_repo_dir,
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    assert args.check_repo or args.check_prs or args.check_branches, (
        "You must specify at least one of --check-repo, --check-prs or --check-local-branches."
    )

    # Make the temp directory if it doesn't exist
    args.temp_dir = setup_temp_dir()

    logging.debug("Parsed arguments: %s", args)

    # Check if the repo is a path or a URL
    work: Work
    if looks_like_ulr(args.repo):
        work = pull_remote_repo(args)
    else:
        work = setup_local_repo(args)

    logging.info("Base repository directory: %s", work.base_repo_dir)
    for base, slices in work.slices_by_branch.items():
        logging.info(
            "Found %d slices in branch '%s'",
            len(slices),
            base,
        )

    logging.info("PRs repository directory: %s", work.prs_repo_dir)
    for pr, slices in work.slices_by_pr.items():
        logging.info(
            "Found %d slices in PR from '%s' to '%s'",
            len(slices),
            f"{pr.head.repo.owner}/{pr.head.repo.name}/{pr.head.ref}",
            pr.base.ref,
        )

    raise NotImplementedError("end of main")

    repo_root = Path(__file__).parent
    if not repo_root.is_dir():
        raise FileNotFoundError(
            f"Repository root {repo_root} does not exist or is not a directory."
        )

    # get all the branches in this repo, then filter out non-release ones
    branches = get_branches(repo_root)
    branches_filtered = list(
        sorted(filter(lambda name: re.match("^ubuntu-", name), branches))
    )

    # generate dictionary mapping SDF to individual releases as sets
    slice_map: dict[str, set[str]] = {}
    for bf in branches_filtered:
        for sl in get_slices(repo_root, bf):
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


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        logging.warning("Some functionality is not implemented yet: %s", e)
        exit(1)
    except AssertionError as e:
        logging.error(e)
        exit(1)
    except Exception as e:
        e_str = str(e)
        if not e_str:
            e_str = "An unknown error occurred."
        logging.error(e_str, exc_info=True)
        exit(1)
