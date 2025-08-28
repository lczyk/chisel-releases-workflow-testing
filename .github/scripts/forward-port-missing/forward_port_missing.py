#!/usr/bin/env python3
"""
Check labels on PRs and forward-port if needed.
"""
# spell-checker: ignore Marcin Konowalczyk lczyk
# spell-checker: words levelname
# mypy: disable-error-code="unused-ignore"

from __future__ import annotations

import argparse
import logging
import os
import sys
from html.parser import HTMLParser

__version__ = "0.0.4"
__author__ = "Marcin Konowalczyk"

__changelog__ = [
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


def currently_supported_ubuntu_releases() -> list[tuple[str, str]]:
    code, res = geturl(DISTS_URL)
    handle_code(code, DISTS_URL)
    parser = DistsParser()
    parser.feed(res.decode("utf-8"))
    out = [(get_version(codename), codename) for codename in parser.dists]
    out.sort()  # sort by version
    return out


################################################################################

import json
from dataclasses import dataclass


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
        )


def geturl_github(url: str, params: dict[str, object] | None = None) -> tuple[int, bytes]:
    assert "github.com" in url, "Only GitHub URLs are supported."
    url = url.replace("github.com", "api.github.com/repos") if "api.github.com" not in url else url
    url = url.rstrip("/")
    headers = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
    github_token = os.getenv("GITHUB_TOKEN", None)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return geturl(url, params=params, headers=headers)


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
    return prs


################################################################################


def get_slices_in_pr(pr: PR) -> set[str]:
    """Get the list of files in the /slices directory in the base branch of the PR."""
    assert pr.base.repo_owner == "canonical", "Only PRs into canonical repositories are supported."
    assert pr.base.repo_name == "chisel-releases", "Only PRs into chisel-releases are supported."
    assert pr.base.ref.startswith("ubuntu-"), "Only PRs into branches named 'ubuntu-XX.XX' are supported."

    url = f"https://api.github.com/repos/{pr.base.repo_owner}/{pr.base.repo_name}/contents/slices"
    code, res = geturl_github(
        url,
        params={"ref": pr.base.ref},
    )
    handle_code(code, url)
    parsed_result = json.loads(res.decode("utf-8"))
    assert isinstance(parsed_result, list), "Expected response to be a list of files."

    files = {item["name"] for item in parsed_result if item["type"] == "file"}
    files = {f.removesuffix(".yaml") for f in files if f.endswith(".yaml")}
    return files


## MAIN ########################################################################


def main(args: argparse.Namespace) -> None:
    ubuntu_releases = currently_supported_ubuntu_releases()
    prs = prs_into_chisel_releases()
    logging.info("Found %d open PRs in %s", len(prs), CHISEL_RELEASES_URL)

    prs_by_ubuntu_release: dict[tuple[str, str], set[PR]] = {
        ubuntu_release: set() for ubuntu_release in ubuntu_releases
    }
    for pr in prs:
        branch = pr.base.ref
        assert branch.startswith("ubuntu-"), "Only PRs into branches named 'ubuntu-XX.XX' are supported."
        version = branch.split("-", 1)[1]
        codename = VERSION_TO_CODENAME.get(version, "unknown")
        key = (version, codename)
        if key not in prs_by_ubuntu_release:
            logging.warning(
                "PR #%d is into unsupported Ubuntu release %s (%s). Skipping.", pr.number, version, codename
            )
            continue
        prs_by_ubuntu_release[key].add(pr)

    # filter out releases with no PRs
    prs_by_ubuntu_release = {k: v for k, v in prs_by_ubuntu_release.items() if len(v) > 0}

    # For each PR, get the list of files in the /slices directory in the base branch
    slices_by_pr: dict[PR, set[str]] = {}
    if args.jobs == 1:
        # NOTE: it is much nicer to debug/profile without parallelism
        slices_by_pr = {pr: get_slices_in_pr(pr) for pr in prs}

    else:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = args.jobs if args.jobs != -1 else None  # None = as many as possible
        _prs = list(prs)  # we want list for zipping with results
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(get_slices_in_pr, _prs)
        for pr, slices in zip(_prs, results):
            slices_by_pr[pr] = slices

    logging.info("Fetched slices for %d PRs", len(slices_by_pr))

    # Log some info
    for ubuntu_release, prs in prs_by_ubuntu_release.items():
        logging.info("Found %d open PRs into ubuntu-%s (%s)", len(prs), ubuntu_release[0], ubuntu_release[1])
        for pr in prs:
            slices = slices_by_pr.get(pr, set())
            logging.info("#%d: %s (%d slices)", pr.number, pr.title, len(slices))

    # for ubuntu_release, prs in prs_by_ubuntu_release.items():
    #     future_releases = [r for r in ubuntu_releases if r > ubuntu_release]
    #     for future_release in future_releases:
    #         prs_into_future_release = prs_by_ubuntu_release.get(future_release, set())
    #         if not prs_into_future_release:
    #             continue
    #         for pr in prs:
    #             slices = slices_by_pr.get(pr, set())
    #             for future_pr in prs_into_future_release:
    #                 future_slices = slices_by_pr.get(future_pr, set())
    #                 missing_slices = slices - future_slices

    # for ubuntu_release, prs in prs_by_ubuntu_release.items():
    #     logging.info("Ubuntu release: %s (%s)", ubuntu_release[0], ubuntu_release[1])
    # for pr in prs:
    #     slices = slices_by_pr.get(pr, set())
    #     logging.info("  PR #%d: %s (%d slices)", pr.number, pr.title, len(slices))
    #     for slice in sorted(slices):
    #         logging.info("    - %s", slice)

    raise NotImplementedError("Main logic not implemented yet.")
    # print(ubuntu_releases)


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
