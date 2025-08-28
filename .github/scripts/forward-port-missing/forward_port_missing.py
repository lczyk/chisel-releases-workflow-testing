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

__version__ = "0.0.1"
__author__ = "Marcin Konowalczyk"

__changelog__ = [
    (__version__, "initial testing", "@lczyk"),
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


################################################################################


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError("Not implemented yet.")
    # ubuntu_releases = currently_supported_ubuntu_releases()
    # print(ubuntu_releases)


################################################################################


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


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    try:
        main(args)

    except Exception as e:
        e_str = str(e)
        e_str = e_str or "An unknown error occurred."
        logging.critical(e_str, exc_info=True)
        sys.exit(1)
