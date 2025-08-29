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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import total_ordering
from html.parser import HTMLParser
from itertools import product
from typing import TYPE_CHECKING, Callable

__version__ = "0.0.7"
__author__ = "Marcin Konowalczyk"

__changelog__ = [
    ("0.0.7", "fetching package lists for FP missing releases", "@lczyk"),
    ("0.0.6", "main forward-porting logic implemented", "@lczyk"),
    ("0.0.5", "slice diff from the merge base, not the branch head", "@lczyk"),
    ("0.0.4", "--jobs for parallel slice fetching", "@lczyk"),
    ("0.0.3", "get_slices_in_pr implemented", "@lczyk"),
    ("0.0.2", "currently_supported_ubuntu_releases implemented", "@lczyk"),
    ("0.0.1", "initial testing", "@lczyk"),
    ("0.0.0", "boilerplate", "@lczyk"),
]

################################################################################


@total_ordering
@dataclass(frozen=True, order=False)
class UbuntuRelease:
    version: str
    codename: str

    def __str__(self) -> str:
        return f"ubuntu-{self.version} ({self.codename})"

    @property
    def version_tuple(self) -> tuple[int, int]:
        return self.version_to_tuple(self.version)

    @staticmethod
    def version_to_tuple(version: str) -> tuple[int, int]:
        year, month = version.split(".")
        return int(year), int(month)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, UbuntuRelease):
            return NotImplemented
        return self.version_tuple < other.version_tuple

    @classmethod
    def from_branch_name(cls, branch: str) -> UbuntuRelease:
        """Create an UbuntuRelease from a branch name like 'ubuntu-20.04'."""
        assert branch.startswith("ubuntu-"), "Branch name must start with 'ubuntu-'"
        version = branch.split("-", 1)[1]
        try:
            _ = cls.version_to_tuple(version)
        except Exception as e:
            raise ValueError(f"Invalid Ubuntu version '{version}' for branch '{branch}': {e}") from None
        codename = _VERSION_TO_CODENAME.get(version)
        if codename is None:
            raise ValueError(f"Unknown Ubuntu version '{version}' for branch '{branch}'")
        return cls(version=version, codename=codename)


## CONSTANTS ###################################################################

# spell-checker: ignore dists utopic yakkety eoan
KNOWN_RELEASES = {
    UbuntuRelease("14.04", "trusty"),
    UbuntuRelease("14.10", "utopic"),
    UbuntuRelease("15.04", "vivid"),
    UbuntuRelease("15.10", "wily"),
    UbuntuRelease("16.04", "xenial"),
    UbuntuRelease("16.10", "yakkety"),
    UbuntuRelease("17.04", "zesty"),
    UbuntuRelease("17.10", "artful"),
    UbuntuRelease("18.04", "bionic"),
    UbuntuRelease("18.10", "cosmic"),
    UbuntuRelease("19.04", "disco"),
    UbuntuRelease("19.10", "eoan"),
    UbuntuRelease("20.04", "focal"),
    UbuntuRelease("20.10", "groovy"),
    UbuntuRelease("21.04", "hirsute"),
    UbuntuRelease("21.10", "impish"),
    UbuntuRelease("22.04", "jammy"),
    UbuntuRelease("22.10", "kinetic"),
    UbuntuRelease("23.04", "lunar"),
    UbuntuRelease("23.10", "mantic"),
    UbuntuRelease("24.04", "noble"),
    UbuntuRelease("24.10", "oracular"),
    UbuntuRelease("25.04", "plucky"),
    UbuntuRelease("25.10", "questing"),
}


ADDITIONAL_VERSIONS_TO_SKIP: set[UbuntuRelease] = {
    UbuntuRelease("24.10", "oracular"),  # EOL
}

_CODENAME_TO_VERSION = {r.codename: r.version for r in KNOWN_RELEASES}
_VERSION_TO_CODENAME = {r.version: r.codename for r in KNOWN_RELEASES}

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


@contextmanager
def CatchTime() -> Iterator[Callable[[], float]]:
    """measure elapsed time of a code block
    Adapted from: https://stackoverflow.com/a/69156219/2531987
    CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/
    """
    t1 = t2 = time.perf_counter()
    yield lambda: t2 - t1
    t2 = time.perf_counter()


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
    if codename in _CODENAME_TO_VERSION:
        return _CODENAME_TO_VERSION[codename]
    return _fallback_get_version(codename)


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


@total_ordering
@dataclass(frozen=True, unsafe_hash=True, order=False)
class PR:
    url: str  # URL of the PR, e.g. http
    number: int  # number of the PR, e.g #601
    title: str  # title of the PR

    user: str  # user who created the PR (usually, but not necessarily, the author)

    head: Commit
    base: Commit

    def __post_init__(self) -> None:
        # Check that head and base are from the same repository
        _ = UbuntuRelease.from_branch_name(self.base.ref)

    @property
    def ubuntu_release(self) -> UbuntuRelease:
        return UbuntuRelease.from_branch_name(self.base.ref)

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

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PR):
            return NotImplemented
        return self.number < other.number


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
        codename = _VERSION_TO_CODENAME.get(version, "unknown")
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

    # filter down to PRs into branches named "ubuntu-XX.XX"
    results = [pr for pr in results if pr["base"]["ref"].startswith("ubuntu-")]
    # filter out draft PRs
    results = [pr for pr in results if not pr.get("draft", False)]

    return set(PR.from_json(pr) for pr in results)


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

    with CatchTime() as elapsed:
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

    logging.info("Fetched merge bases for %d PRs in %.2f seconds.", len(prs), elapsed())
    for pr, mb in merge_bases_by_pr.items():
        if pr.base.commit != mb:
            logging.warning(
                "PR #%d: base branch '%s' has advanced since the PR was created/updated. Consider rebasing.",
                pr.number,
                pr.base.ref,
            )

    return merge_bases_by_pr


def get_slices_by_pr(
    prs: set[PR],
    merge_bases_by_pr: dict[PR, str],
    jobs: int | None = 1,
) -> tuple[dict[PR, set[str]], dict[PR, set[str]]]:
    # For each PR, get the list of files in the /slices directory in the base branch
    slices_in_head_by_pr: dict[PR, set[str]] = {}
    slices_in_base_by_pr: dict[PR, set[str]] = {}
    get_slices_base = lambda pr: get_slices(pr.base.repo_owner, pr.base.repo_name, merge_bases_by_pr[pr])
    get_slices_head = lambda pr: get_slices(pr.head.repo_owner, pr.head.repo_name, pr.head.ref)

    with CatchTime() as elapsed:
        if jobs == 1:
            # NOTE: it is much nicer to debug/profile without parallelism
            slices_in_head_by_pr = {pr: get_slices_head(pr) for pr in prs}
            slices_in_base_by_pr = {pr: get_slices_base(pr) for pr in prs}

        else:
            from concurrent.futures import ThreadPoolExecutor

            _prs = list(prs)  # we want list for zipping with results
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                thread_pool_size = getattr(executor, "_max_workers", -1)
                results_head = list(executor.map(get_slices_head, _prs))
                results_base = list(executor.map(get_slices_base, _prs))

            logging.debug("Used a thread pool of size %d.", thread_pool_size)
            slices_in_head_by_pr = {pr: slices for pr, slices in zip(_prs, results_head)}
            slices_in_base_by_pr = {pr: slices for pr, slices in zip(_prs, results_base)}

    total_slices = sum(len(slices) for slices in slices_in_head_by_pr.values())
    total_slices += sum(len(slices) for slices in slices_in_base_by_pr.values())
    logging.info("Fetched %d slices for %d PRs in %.2f seconds.", total_slices, len(prs), elapsed())

    return slices_in_head_by_pr, slices_in_base_by_pr


class SliceComparisonResult:
    base_pr: PR
    base_slices: set[str]

    _yes: dict[PR, set[str]]
    _no: dict[PR, set[str]]

    def __init__(self, pr: PR, slices: set[str]) -> None:
        self.base_pr = pr
        self.base_slices = slices
        self._yes = {}
        self._no = {}

    def __bool__(self) -> bool:
        return bool(self._yes)

    def compare_with(self, pr: PR, slices: set[str]) -> None:
        if self.base_slices.issubset(slices):
            self.is_forward_ported = True
            self._yes[pr] = slices
        else:
            self._no[pr] = slices

    def __repr__(self) -> str:
        return (
            f"SliceComparisonResult(base_pr=#{self.base_pr.number}, "
            f"base_slices={len(self.base_slices)}, "
            f"comparisons={len(self._yes) + len(self._no)}, "
            f"is_forward_ported={self.is_forward_ported}, "
        )

    def forward_ports(self) -> set[PR]:
        return set(self._yes.keys())


def get_results_by_pr(
    prs_by_ubuntu_release: dict[UbuntuRelease, set[PR]],
    new_slices_by_pr: dict[PR, set[str]],
) -> dict[PR, dict[UbuntuRelease, SliceComparisonResult]]:
    prs: set[PR] = set()
    for prs_in_release in prs_by_ubuntu_release.values():
        prs.update(prs_in_release)

    ubuntu_releases = sorted(prs_by_ubuntu_release.keys())

    # For each PR we have a mapping from ubuntu release to a set of PRs that
    # forward-port the new slices to that release. An empty set means no
    # forward-port found, a set with None means no new slices to forward-port.
    status_by_pr: dict[PR, dict[UbuntuRelease, SliceComparisonResult]] = {pr: {} for pr in prs}

    for ubuntu_release, prs_in_release in prs_by_ubuntu_release.items():
        future_releases = [r for r in ubuntu_releases if r > ubuntu_release]
        if not future_releases:
            logging.debug("No future releases for %s. Skipping all PRs into it.", ubuntu_release)
            continue

        for pr in prs_in_release:
            new_slices = new_slices_by_pr.get(pr, set())
            for future_release in future_releases:
                status_by_pr[pr][future_release] = SliceComparisonResult(pr, new_slices)
                prs_into_future_release = prs_by_ubuntu_release.get(future_release, set())
                if not prs_into_future_release:
                    logging.debug("No PRs into future release %s of PR #%d", future_release, pr.number)
                    # No PRs into this future release
                    continue

                for pr_future in prs_into_future_release:
                    new_slices_in_future = new_slices_by_pr.get(pr_future, set())
                    status_by_pr[pr][future_release].compare_with(pr_future, new_slices_in_future)

    return status_by_pr


def get_packages_by_release(
    releases: set[UbuntuRelease],
    jobs: int | None = 1,
) -> dict[UbuntuRelease, set[str]]:
    package_listings: dict[tuple[UbuntuRelease, str, str], set[str]] = {}

    _releases = list(releases)  # we want list for zipping with results
    _components = ("main", "restricted", "universe", "multiverse")
    _repos = ("", "security", "updates", "backports")
    _product = list(product(_releases, _components, _repos))

    with CatchTime() as elapsed:
        if jobs == 1:
            for release, component, repo in _product:
                package_listings[(release, component, repo)] = get_package_content(release, component, repo)

        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=jobs) as executor:
                thread_pool_size = getattr(executor, "_max_workers", -1)
                results = list(
                    executor.map(lambda args: get_package_content(*args), _product)  # type: ignore
                )
            logging.debug("Used a thread pool of size %d.", thread_pool_size)
            package_listings = {args: pkgs for args, pkgs in zip(_product, results)}

    logging.info("Fetched packages for %d releases in %.2f seconds.", len(releases), elapsed())

    # Union all components and repos
    packages_by_release: dict[UbuntuRelease, set[str]] = {r: set() for r in releases}
    for (release, _component, _repo), packages in package_listings.items():
        packages_by_release[release].update(packages)

    return packages_by_release


if TYPE_CHECKING:
    import re

_PACKAGE_RE: re.Pattern[str] | None = None  # cache for the compiled regex


def get_package_content(release: UbuntuRelease, component: str, repo: str) -> set[str]:
    if component not in ("main", "restricted", "universe", "multiverse"):
        raise ValueError(
            f"Invalid component: {component}. Must be one of 'main', 'restricted', 'universe', or 'multiverse'."
        )
    if repo not in ("", "security", "updates", "backports"):
        raise ValueError(f"Invalid repo: {repo}. Must be one of '', 'security', 'updates', or 'backports'.")

    logging.debug("Fetching packages for %s, component=%s, repo=%s", release, component, repo)

    name = release.codename
    name = f"{name}-{repo}" if repo else name

    package_url = f"https://archive.ubuntu.com/ubuntu/dists/{name}/{component}/binary-amd64/Packages.gz"
    code, res = geturl(package_url)

    if code != 200:
        # retry with old-releases if not found in archive
        package_url = f"https://old-releases.ubuntu.com/ubuntu/dists/{name}/{component}/binary-amd64/Packages.gz"
        code, res = geturl(package_url)

    if code != 200:
        raise RuntimeError(f"Failed to download package list from '{package_url}'. HTTP status code: {code}")

    import gzip
    import io
    import re

    with gzip.GzipFile(fileobj=io.BytesIO(res)) as f:
        content = f.read().decode("utf-8")

    global _PACKAGE_RE  # noqa: PLW0603
    if _PACKAGE_RE:
        compiled_re = _PACKAGE_RE
    else:
        compiled_re = re.compile(r"^Package:\s*(\S+)", re.MULTILINE)
        _PACKAGE_RE = compiled_re  # cache it

    return set(m.group(1) for m in compiled_re.finditer(content))


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

    if ADDITIONAL_VERSIONS_TO_SKIP:
        logging.info("Skipping additional versions: %s", ", ".join(str(r) for r in ADDITIONAL_VERSIONS_TO_SKIP))

    ubuntu_releases = [r for r in ubuntu_releases if r not in ADDITIONAL_VERSIONS_TO_SKIP]

    logging.info(
        "Considering %d supported Ubuntu releases with branches in chisel-releases: %s",
        len(ubuntu_releases),
        ", ".join(str(r) for r in ubuntu_releases),
    )

    prs = get_all_prs(CHISEL_RELEASES_URL)
    logging.info("Found %d open PRs in %s", len(prs), CHISEL_RELEASES_URL)

    prs_by_ubuntu_release: dict[UbuntuRelease, set[PR]] = {ubuntu_release: set() for ubuntu_release in ubuntu_releases}
    _prs = list(sorted(prs))  # we want list for logging
    for pr in _prs:
        ubuntu_release = UbuntuRelease.from_branch_name(pr.base.ref)
        if ubuntu_release not in prs_by_ubuntu_release:
            prs.discard(pr)
            logging.warning("PR #%d is into unsupported Ubuntu release %s. Skipping.", pr.number, ubuntu_release)
            continue
        prs_by_ubuntu_release[ubuntu_release].add(pr)

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
    for pr in sorted(prs):
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

    results_by_pr = get_results_by_pr(prs_by_ubuntu_release, new_slices_by_pr)

    prs_with_no_forward_ports = {pr for pr, results in results_by_pr.items() if not any(results.values())}
    if prs_with_no_forward_ports and logging.getLogger().isEnabledFor(logging.WARNING):
        logging.warning(
            "Found %d PRs with missing forward-ports: %s",
            len(prs_with_no_forward_ports),
            ", ".join(f"#{pr.number}" for pr in sorted(prs_with_no_forward_ports)),
        )

    if prs_with_no_forward_ports:
        # if we have a bunch of PRs with missing forward-ports, they *might* be
        # missing because the package is just not in the newer release.
        # NOTE: we don't need to fetch the packages for all releases, just
        #       for the *future* releases that are missing forward-ports.
        releases_to_fetch: set[UbuntuRelease] = set()
        for pr in prs_with_no_forward_ports:
            future_releases = [r for r in ubuntu_releases if r > pr.ubuntu_release]
            releases_to_fetch.update(future_releases)
        # Sanity check. If we got here, we should have at least one release to fetch.
        assert releases_to_fetch, "Expected at least one release to fetch packages for."
        packages_by_release = get_packages_by_release(releases_to_fetch, args.jobs)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for release, packages in packages_by_release.items():
                logging.debug("Release %s has %d packages.", release, len(packages))

        # for pr in sorted(prs_with_no_forward_ports):
        #     future_releases = [r for r in ubuntu_releases if r > pr.ubuntu_release]

    for pr, results_by_future in results_by_pr.items():
        for future_release, result in results_by_future.items():
            forward_ports = result.forward_ports()
            if not forward_ports:
                logging.warning(
                    "PR #%d into %s: No PRs found into future release %s.",
                    pr.number,
                    pr.base.ref,
                    future_release,
                )
            elif not result.base_slices:
                logging.info(
                    "PR #%d into %s has no new slices, so it's OK for future release %s.",
                    pr.number,
                    pr.base.ref,
                    future_release,
                )
            else:
                logging.info(
                    "PR #%d into %s is forward-ported to future release %s by %d PR(s) (%s).",
                    pr.number,
                    pr.base.ref,
                    future_release,
                    len(forward_ports),
                    ", ".join(f"#{p.number}" for p in sorted(forward_ports)),
                )

    # make sure we didn't drop any PRs
    all_prs_in_results = set(results_by_pr.keys())
    if all_prs_in_results != prs:
        missing_prs = prs - all_prs_in_results
        additional_prs = all_prs_in_results - prs
        logging.error("Some PRs are missing in the results: %s", ", ".join(f"#{pr.number}" for pr in missing_prs))
        if additional_prs:
            logging.error(
                "Some additional PRs are in the results but not in the input: %s",
                ", ".join(f"#{pr.number}" for pr in additional_prs),
            )
        raise Exception("Some PRs are missing in the results.")

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
