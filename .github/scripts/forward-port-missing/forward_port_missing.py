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
import sys
from html.parser import HTMLParser

__version__ = "0.0.2"
__author__ = "Marcin Konowalczyk"

__changelog__ = [
    ("0.0.2", "currently_supported_ubuntu_releases implemented", "@lczyk"),
    ("0.0.1", "initial testing", "@lczyk"),
    ("0.0.0", "boilerplate", "@lczyk"),
]

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
            self.dists.add(dist)


# spell-checker: ignore dists utopic yakkety eoan DEVEL
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

DEVEL = ("25.10", "questing")

CODENAME_TO_VERSION = {v: k for k, v in VERSION_TO_CODENAME.items()}

DISTS_URL = "https://archive.ubuntu.com/ubuntu/dists/"


def _fallback_get_version(codename: str) -> tuple[str, str]:
    logging.warning("Unknown codename %s, trying to fetch version from the web.", codename)
    code, res = geturl(f"{DISTS_URL}{codename}/Release")
    if code != 200:
        raise Exception(f"Failed to fetch {DISTS_URL}{codename}/Release: HTTP {code}")
    content = res.decode("utf-8")
    version, _codename = None, None
    for line in content.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
        elif line.startswith("Codename:"):
            _codename = line.split(":", 1)[1].strip()
        if version and _codename:
            break
    if not version or not _codename:
        raise Exception(f"Failed to parse version or codename from {DISTS_URL}{codename}/Release")
    return (version, _codename)


def get_version_and_codename(codename: str) -> tuple[str, str]:
    if codename in CODENAME_TO_VERSION:
        return (CODENAME_TO_VERSION[codename], codename)
    if codename == "devel":
        logging.debug("Assuming devel is %s %s", DEVEL[0], DEVEL[1])
        return DEVEL
    return _fallback_get_version(codename)


def currently_supported_ubuntu_releases() -> list[tuple[str, str]]:
    code, res = geturl(DISTS_URL)
    if code != 200:
        raise Exception(f"Failed to fetch {DISTS_URL}: HTTP {code}")
    parser = DistsParser()
    parser.feed(res.decode("utf-8"))
    out = [get_version_and_codename(codename) for codename in parser.dists]
    out.sort()  # sort by version
    return out


## MAIN ########################################################################


def main(args: argparse.Namespace) -> None:
    ubuntu_releases = currently_supported_ubuntu_releases()
    print(ubuntu_releases)
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
    return parser.parse_args()


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

    except NotImplementedError as e:
        logging.error("Not implemented: %s", e)
        sys.exit(99)

    except Exception as e:
        e_str = str(e)
        e_str = e_str or "An unknown error occurred."
        logging.critical(e_str, exc_info=True)
        sys.exit(1)
