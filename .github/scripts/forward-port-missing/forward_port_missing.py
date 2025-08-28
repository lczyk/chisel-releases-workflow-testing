#!/usr/bin/env python3
"""
Check labels on PRs and forward-port if needed.
"""
# spell-checker: ignore Marcin Konowalczyk lczyk
# spell-checker: words levelname
# mypy: disable-error-code="unused-ignore"

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser

__version__ = "0.0.6"
__author__ = "Marcin Konowalczyk"

__changelog__ = [
    ("0.0.6", "main forward-porting logic implemented", "@lczyk"),
    ("0.0.5", "slice diff from the merge base, not the branch head", "@lczyk"),
    ("0.0.4", "--jobs for parallel slice fetching", "@lczyk"),
    ("0.0.3", "get_slices_in_pr implemented", "@lczyk"),
    ("0.0.2", "currently_supported_ubuntu_releases implemented", "@lczyk"),
    ("0.0.1", "initial testing", "@lczyk"),
    ("0.0.0", "boilerplate", "@lczyk"),
]

## CONSTANTS ###################################################################

# spell-checker: ignore dists utopic yakkety eoan
VERSION_TO_CODENAME = {
    "14.04": "trusty",
    "14.10": "utopic",
    "15.04": "vivid",
    "15.10": "wily",
    "16.04": "xenial",
    "16.10": "yakkety",
    "17.04": "zesty",
    "17.10": "artful",
    "18.04": "bionic",
    "18.10": "cosmic",
    "19.04": "disco",
    "19.10": "eoan",
    "20.04": "focal",
    "20.10": "groovy",
    "21.04": "hirsute",
    "21.10": "impish",
    "22.04": "jammy",
    "22.10": "kinetic",
    "23.04": "lunar",
    "23.10": "mantic",
    "24.04": "noble",
    "24.10": "oracular",
    "25.04": "plucky",
    "25.10": "questing",
}

CODENAME_TO_VERSION = {v: k for k, v in VERSION_TO_CODENAME.items()}

DISTS_URL = "https://archive.ubuntu.com/ubuntu/dists"

CHISEL_RELEASES_URL = "https://github.com/canonical/chisel-releases"


## LIB #########################################################################


# geturl from https://github.com/lczyk/geturl 0.4.5
def geturl(
    url: str,
    params: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """Make a GET request to a URL and return the response and status code."""

    import urllib
    import urllib.error
    import urllib.parse
    import urllib.request

    if params is not None:
        if "?" in url:
            params = dict(params)  # make a modifiable copy
            existing_params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            params = {**existing_params, **params}  # params take precedence
            url = url.split("?")[0]
        url = url + "?" + urllib.parse.urlencode(params)

    request = urllib.request.Request(url)
    if headers is not None:
        for h_key, h_value in headers.items():
            request.add_header(h_key, h_value)

    try:
        with urllib.request.urlopen(request) as r:
            code = r.getcode()
            res = r.read()

    except urllib.error.HTTPError as e:
        code = e.code
        res = e.read()

    assert isinstance(code, int), "Expected code to be int."
    assert isinstance(res, bytes), "Expected response to be bytes."

    return code, res


if os.environ.get("USE_MEMORY", "0") in ("1", "true", "True", "TRUE"):
    # we don't own the apis we call so, during development, its only polite to cache
    from joblib import Memory

    memory = Memory(".cache-geturl", verbose=0)
    geturl = memory.cache(geturl)  # type: ignore


class RateLimit(Exception): ...


def handle_code(code: int, url: str) -> None:
    if code == 200:
        return
    if code == 404:
        raise Exception(f"Resource not found at '{url}'.")
    if code == 403:
        if "github.com" in url:
            raise RateLimit("Rate limited by GitHub API. Try again later.")
        else:
            raise Exception(f"Access forbidden to '{url}'.")
    if code == 401:
        if "github.com" in url:
            raise Exception(f"Unauthorized access to '{url}'. Maybe bad creds? Check GITHUB_TOKEN.")
        else:
            raise Exception(f"Unauthorized access to '{url}'.")
    raise Exception(f"Failed to fetch '{url}'. HTTP status code: {code}")


## IMPL ########################################################################


class DistsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.dists: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href", "")
            if not href or not href.endswith("/") or href.startswith("/"):
                return
            dist = href.strip("/")
            # change, for example "bionic-updates" to just "bionic"
            dist = dist.split("-")[0] if "-" in dist else dist
            if dist == "devel":
                return
            self.dists.add(dist)


def _fallback_get_version(codename: str) -> str:
    """Fetch version and codename from the web as a fallback."""
    logging.warning("Unknown codename %s, trying to fetch version from the web.", codename)
    url = f"{DISTS_URL}/{codename}/Release"
    code, res = geturl(url)
    handle_code(code, url)
    content = res.decode("utf-8")
    for line in content.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
            return version
    raise Exception(f"Could not find version for codename '{codename}'.")


def get_version(codename: str) -> str:
    if codename in CODENAME_TO_VERSION:
        return CODENAME_TO_VERSION[codename]
    return _fallback_get_version(codename)


@dataclass(frozen=True, order=True)
class UbuntuRelease:
    version: str
    codename: str

    def __str__(self) -> str:
        return f"ubuntu-{self.version} ({self.codename})"


def currently_supported_ubuntu_releases() -> list[UbuntuRelease]:
    code, res = geturl(DISTS_URL)
    handle_code(code, DISTS_URL)
    parser = DistsParser()
    parser.feed(res.decode("utf-8"))
    out = [(get_version(codename), codename) for codename in parser.dists]
    out.sort()  # sort by version
    return [UbuntuRelease(version=v, codename=c) for v, c in out]


################################################################################


@dataclass(frozen=True, unsafe_hash=True)
class Commit:
    commit: str  # SHA of the commit
    ref: str  # Branch name
    repo_name: str  # Name of the repository
    repo_url: str  # URL of the repository
    repo_owner: str  # Owner of the repository

    @classmethod
    def from_json(cls, data: dict) -> Commit:
        return cls(
            commit=data["sha"],
            ref=data["ref"],
            repo_name=data["repo"]["name"],
            repo_url=data["repo"]["html_url"],
            repo_owner=data["repo"]["owner"]["login"],
        )


@dataclass(frozen=True, unsafe_hash=True)
class PR:
    url: str  # URL of the PR, e.g. http
    number: int  # number of the PR, e.g #601
    title: str  # title of the PR

    user: str  # user who created the PR (usually, but not necessarily, the author)

    head: Commit
    base: Commit

    draft: bool

    @classmethod
    def from_json(cls, data: dict) -> PR:
        head = Commit.from_json(data["head"])
        base = Commit.from_json(data["base"])

        return cls(
            url=data["html_url"],
            number=data["number"],
            title=data["title"],
            user=data["user"]["login"],
            head=head,
            base=base,
            draft=data.get("draft", False),  # draft field was added later
        )


def get_merge_base(base: Commit, head: Commit) -> str:
    """Get the SHA of the merge base between head and base."""
    url = (
        f"https://api.github.com/repos/{base.repo_owner}/{base.repo_name}/compare/"
        f"{base.repo_owner}:{base.ref}...{head.repo_owner}:{head.ref}?per_page=1"
    )
    code, res = geturl_github(url)
    handle_code(code, url)
    parsed_result = json.loads(res.decode("utf-8"))
    assert isinstance(parsed_result, dict), "Expected response to be a dict."
    if "merge_base_commit" not in parsed_result:
        raise Exception(f"Could not find merge_base_commit in response from '{url}'.")
    merge_base_commit = parsed_result["merge_base_commit"]
    assert isinstance(merge_base_commit, dict), "Expected merge_base_commit to be a dict."
    if "sha" not in merge_base_commit:
        raise Exception(f"Could not find sha in merge_base_commit from '{url}'.")
    sha = merge_base_commit["sha"]
    assert isinstance(sha, str), "Expected sha to be a str."
    return sha


def geturl_github(url: str, params: dict[str, object] | None = None) -> tuple[int, bytes]:
    assert "github.com" in url, "Only GitHub URLs are supported."
    url = url.replace("github.com", "api.github.com/repos") if "api.github.com" not in url else url
    url = url.rstrip("/")
    headers = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
    github_token = os.getenv("GITHUB_TOKEN", None)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return geturl(url, params=params, headers=headers)


def ubuntu_branches_in_chisel_releases() -> set[UbuntuRelease]:
    code, res = geturl_github(f"{CHISEL_RELEASES_URL}/branches", params={"per_page": 100})
    handle_code(code, CHISEL_RELEASES_URL)
    parsed_result = json.loads(res.decode("utf-8"))
    assert isinstance(parsed_result, list), "Expected response to be a list of branches."
    branches = {branch["name"] for branch in parsed_result if branch["name"].startswith("ubuntu-")}
    ubuntu_releases = set()
    for branch in branches:
        version = branch.split("-", 1)[1]
        codename = VERSION_TO_CODENAME.get(version, "unknown")
        ubuntu_releases.add(UbuntuRelease(version, codename))
    return ubuntu_releases


def get_all_prs(url: str, per_page: int = 100) -> set[PR]:
    """Fetch all PRs from the remote repository using the GitHub API. The url
    should be the URL of the repository, e.g. www.github.com/canonical/chisel-releases.
    """
    assert per_page > 0, "per_page must be a positive integer."
    url = url.rstrip("/") + "/pulls"

    params = {"state": "open", "per_page": per_page, "page": 1}

    results = []
    while True:
        code, result = geturl_github(url, params=params)
        handle_code(code, url)
        parsed_result = json.loads(result.decode("utf-8"))
        assert isinstance(parsed_result, list), "Expected response to be a list of PRs."
        results.extend(parsed_result)
        if len(parsed_result) < per_page:
            break
        params["page"] += 1  # type: ignore

    return set(PR.from_json(pr) for pr in results)


def prs_into_chisel_releases() -> set[PR]:
    prs = get_all_prs(url=CHISEL_RELEASES_URL)
    # filter down to PRs into branches named "ubuntu-XX.XX"
    prs = {pr for pr in prs if pr.base.ref.startswith("ubuntu-")}
    # filter out draft PRs
    prs = {pr for pr in prs if not pr.draft}
    return prs


################################################################################


def get_slices(repo_owner: str, repo_name: str, ref: str) -> set[str]:
    """Get the list of files in the /slices directory in the given ref.
    ref can be a branch name, tag name, or commit SHA.
    """

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/slices"
    code, res = geturl_github(
        url,
        params={"ref": ref},
    )
    handle_code(code, url)
    parsed_result = json.loads(res.decode("utf-8"))
    assert isinstance(parsed_result, list), "Expected response to be a list of files."

    files = {item["name"] for item in parsed_result if item["type"] == "file"}
    files = {f.removesuffix(".yaml") for f in files if f.endswith(".yaml")}
    return files


def get_merge_bases_by_pr(prs: set[PR], jobs: int | None = 1) -> dict[PR, str]:
    merge_bases_by_pr: dict[PR, str] = {}
    if jobs == 1:
        # NOTE: it is much nicer to debug/profile without parallelism
        merge_bases_by_pr = {pr: get_merge_base(pr.base, pr.head) for pr in prs}
    else:
        from concurrent.futures import ThreadPoolExecutor

        _prs = list(prs)  # we want list for zipping with results
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            thread_pool_size = getattr(executor, "_max_workers", -1)
            results = list(executor.map(lambda pr: get_merge_base(pr.base, pr.head), _prs))
        logging.debug("Used a thread pool of size %d.", thread_pool_size)
        merge_bases_by_pr = {pr: mb for pr, mb in zip(_prs, results)}

    logging.info("Fetched merge bases for %d PRs.", len(merge_bases_by_pr))
    for pr, mb in merge_bases_by_pr.items():
        if pr.base.commit != mb:
            logging.warning(
                "PR #%d: base branch '%s' has advanced since the PR was created/updated. Consider rebasing.",
                pr.number,
                pr.base.ref,
            )

    return merge_bases_by_pr


def get_slices_by_pr(
    prs: set[PR], merge_bases_by_pr: dict[PR, str], jobs: int | None = 1
) -> tuple[dict[PR, set[str]], dict[PR, set[str]]]:
    # For each PR, get the list of files in the /slices directory in the base branch
    slices_in_head_by_pr: dict[PR, set[str]] = {}
    slices_in_base_by_pr: dict[PR, set[str]] = {}
    get_slices_base = lambda pr: get_slices(pr.base.repo_owner, pr.base.repo_name, merge_bases_by_pr[pr])
    get_slices_head = lambda pr: get_slices(pr.head.repo_owner, pr.head.repo_name, pr.head.ref)

    tic = time.perf_counter()
    if args.jobs == 1:
        # NOTE: it is much nicer to debug/profile without parallelism
        slices_in_head_by_pr = {pr: get_slices_head(pr) for pr in prs}
        slices_in_base_by_pr = {pr: get_slices_base(pr) for pr in prs}

    else:
        from concurrent.futures import ThreadPoolExecutor

        _prs = list(prs)  # we want list for zipping with results
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            thread_pool_size = getattr(executor, "_max_workers", -1)
            results_head = list(executor.map(get_slices_head, _prs))
            results_base = list(executor.map(get_slices_base, _prs))

        logging.debug("Used a thread pool of size %d.", thread_pool_size)
        slices_in_head_by_pr = {pr: slices for pr, slices in zip(_prs, results_head)}
        slices_in_base_by_pr = {pr: slices for pr, slices in zip(_prs, results_base)}
    toc = time.perf_counter()

    total_slices = sum(len(slices) for slices in slices_in_head_by_pr.values())
    total_slices += sum(len(slices) for slices in slices_in_base_by_pr.values())
    logging.info("Fetched %d slices for %d PRs in %.2f seconds.", total_slices, len(prs), toc - tic)

    return slices_in_head_by_pr, slices_in_base_by_pr


def get_results_by_pr(
    prs_by_ubuntu_release: dict[UbuntuRelease, set[PR]],
    new_slices_by_pr: dict[PR, set[str]],
) -> dict[PR, dict[UbuntuRelease, set[PR | None]]]:
    prs: set[PR] = set()
    for prs_in_release in prs_by_ubuntu_release.values():
        prs.update(prs_in_release)

    ubuntu_releases = sorted(prs_by_ubuntu_release.keys())

    # For each PR we have a mapping from ubuntu release to a set of PRs that
    # forward-port the new slices to that release. An empty set means no
    # forward-port found, a set with None means no new slices to forward-port.
    status_by_pr: dict[PR, dict[UbuntuRelease, set[PR | None]]] = {pr: {} for pr in prs}

    for ubuntu_release, prs_in_release in prs_by_ubuntu_release.items():
        future_releases = [r for r in ubuntu_releases if r > ubuntu_release]
        if not future_releases:
            logging.debug("No future releases for %s. Skipping all PRs into it.", ubuntu_release)
            continue

        for pr in prs_in_release:
            assert pr in status_by_pr
            new_slices = new_slices_by_pr.get(pr, set())

            for future_release in future_releases:
                status_by_pr[pr][future_release] = set()
                if not new_slices:
                    logging.debug("PR #%d contains no new slices. Marking as OK.", pr.number)
                    # No new slices to forward-port, so definitely OK
                    status_by_pr[pr][future_release].add(None)
                    continue

                prs_into_future_release = prs_by_ubuntu_release.get(future_release, set())
                if not prs_into_future_release:
                    logging.debug("No PRs into future release %s of PR #%d", future_release, pr.number)
                    # No PRs into this future release
                    continue

                for pr_future in prs_into_future_release:
                    new_slices_in_future = new_slices_by_pr.get(pr_future, set())
                    if not new_slices_in_future:
                        # logging.debug(
                        #     "PR #%d cannot be the forward-port of PR #%d because it contains no new slices.",
                        #     pr_future.number,
                        #     pr.number,
                        # )
                        continue
                    if new_slices.issubset(new_slices_in_future):
                        # Hooray! We found a PR that forward-ports all new slices!
                        status_by_pr[pr][future_release].add(pr_future)

    return status_by_pr


## MAIN ########################################################################


def main(args: argparse.Namespace) -> None:
    ubuntu_releases = currently_supported_ubuntu_releases()
    ubuntu_branches = ubuntu_branches_in_chisel_releases()

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        ubuntu_releases_str = ", ".join(str(r) for r in ubuntu_releases)
        logging.debug(
            "Found %d supported Ubuntu releases in the archives %s", len(ubuntu_releases), ubuntu_releases_str
        )
        ubuntu_branches_str = ", ".join(str(r) for r in sorted(ubuntu_branches))
        logging.debug("Found %d Ubuntu branches in chisel-releases: %s", len(ubuntu_branches), ubuntu_branches_str)
        will_drop = set(ubuntu_releases) - ubuntu_branches
        if will_drop:
            will_drop_str = ", ".join(str(r) for r in sorted(will_drop))
            logging.debug("Will drop %d supported releases without branches: %s", len(will_drop), will_drop_str)

    ubuntu_releases = [r for r in ubuntu_releases if r in ubuntu_branches]
    logging.info(
        "Considering %d supported Ubuntu releases with branches in chisel-releases: %s",
        len(ubuntu_releases),
        ", ".join(str(r) for r in ubuntu_releases),
    )

    prs = prs_into_chisel_releases()
    logging.info("Found %d open PRs in %s", len(prs), CHISEL_RELEASES_URL)

    prs_by_ubuntu_release: dict[UbuntuRelease, set[PR]] = {ubuntu_release: set() for ubuntu_release in ubuntu_releases}
    for pr in prs:
        branch = pr.base.ref
        assert branch.startswith("ubuntu-"), "Only PRs into branches named 'ubuntu-XX.XX' are supported."
        version = branch.split("-", 1)[1]
        codename = VERSION_TO_CODENAME.get(version, "unknown")
        key = UbuntuRelease(version, codename)
        if key not in prs_by_ubuntu_release:
            logging.warning(
                "PR #%d is into unsupported Ubuntu release %s (%s). Skipping.", pr.number, version, codename
            )
            continue
        prs_by_ubuntu_release[key].add(pr)

    # filter out releases with no PRs
    prs_by_ubuntu_release = {k: v for k, v in prs_by_ubuntu_release.items() if len(v) > 0}

    merge_bases_by_pr = get_merge_bases_by_pr(prs, args.jobs)
    slices_in_head_by_pr, slices_in_base_by_pr = get_slices_by_pr(prs, merge_bases_by_pr, args.jobs)

    # Log some info
    for ubuntu_release, prs_in_release in prs_by_ubuntu_release.items():
        logging.info("Found %d open PRs into %s", len(prs_in_release), ubuntu_release)
        for pr in prs_in_release:
            logging.info(
                "  #%d: %s (%d slices in head, %d slices in merge base)",
                pr.number,
                pr.title,
                len(slices_in_head_by_pr.get(pr, set())),
                len(slices_in_base_by_pr.get(pr, set())),
            )

    new_slices_by_pr: dict[PR, set[str]] = {}
    for pr in prs:
        slices_in_head = slices_in_head_by_pr.get(pr, set())
        slices_in_base = slices_in_base_by_pr.get(pr, set())
        new_slices = slices_in_head - slices_in_base
        removed_sliced = slices_in_base - slices_in_head
        if removed_sliced and logging.getLogger().isEnabledFor(logging.WARNING):
            slices_string = ", ".join(sorted(removed_sliced))
            slices_string = slices_string if len(slices_string) < 100 else slices_string[:97] + "..."
            logging.warning("PR #%d removed %d slices: %s", pr.number, len(removed_sliced), slices_string)
        if new_slices:
            new_slices_by_pr[pr] = new_slices
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                slices_string = ", ".join(sorted(new_slices))
                slices_string = slices_string if len(slices_string) < 100 else slices_string[:97] + "..."
                logging.debug("PR #%d introduces %d new slices: %s", pr.number, len(new_slices), slices_string)

    results_by_pr: dict[PR, dict[UbuntuRelease, set[PR | None]]] = get_results_by_pr(
        prs_by_ubuntu_release, new_slices_by_pr
    )

    for pr, results_by_future in results_by_pr.items():
        for future_release, results in results_by_future.items():
            if not results:
                logging.warning(
                    "PR #%d into %s: No PRs found into future release %s.",
                    pr.number,
                    pr.base.ref,
                    future_release,
                )
            elif None in results:
                logging.info(
                    "PR #%d into %s has no new slices, so it's OK for future release %s.",
                    pr.number,
                    pr.base.ref,
                    future_release,
                )
            else:
                _results: set[PR] = results  # type: ignore
                logging.info(
                    "PR #%d into %s is forward-ported to future release %s by %d PR(s) (%s).",
                    pr.number,
                    pr.base.ref,
                    future_release,
                    len(results),
                    ", ".join(f"#{p.number}" for p in _results),
                )

    # make sure we didn't drop any PRs
    all_prs_in_results = set(results_by_pr.keys())
    assert all_prs_in_results == prs, "Some PRs were dropped from the results."

    raise NotImplementedError("Main logic not implemented yet.")


## BOILERPLATE #################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check labels on PRs and forward-port if needed.",
        epilog="Example: ./forward-port-missing.py --log-level debug",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "fatal", "critical"],
        help="Set the logging level (default: info).",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,  # -1 = as many as possible, 1 = no parallelism
        help="Number of parallel jobs to use when fetching PR details. Default is 1 (no parallelism).",
    )
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be a positive integer or -1 for unlimited.")
    args.jobs = None if args.jobs == -1 else args.jobs  # None = as many as possible
    return args


def setup_logging(log_level: str) -> None:
    _logger = logging.getLogger()
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter: type[logging.Formatter] = logging.Formatter
    # Try to use colorlog for colored output
    try:
        import colorlog  # type: ignore

        fmt = fmt.replace("%(levelname)s", "%(log_color)s%(levelname)s%(reset)s")
        formatter = colorlog.ColoredFormatter  # type: ignore
    except ImportError:
        pass

    handler.setFormatter(formatter(fmt, datefmt))  # type: ignore
    _logger.addHandler(handler)
    log_level = "critical" if log_level.lower() == "fatal" else log_level
    _logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))


## ENTRYPOINT ##################################################################

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    logging.debug("Parsed args: %s", args)
    try:
        main(args)

    except RateLimit as e:
        logging.error("Rate limited: %s", e)
        sys.exit(98)

    except NotImplementedError as e:
        logging.error("Not implemented: %s", e)
        sys.exit(99)

    except Exception as e:
        e_str = str(e)
        e_str = e_str or "An unknown error occurred."
        logging.critical(e_str, exc_info=True)
        sys.exit(1)
