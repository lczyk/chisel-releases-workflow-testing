[[ -z "${__COMMON_SH__:-}" ]] && __COMMON_SH__=1 || return 0

# Find the root of the `forward-port-missing` script files
# add root to PATH so we can source scripts from there
__FORWARD_PORTING_ROOT__="$(dirname "$PWD")"
export PATH="$__FORWARD_PORTING_ROOT__/:$PATH"

# Find the root of the entire repository
# Walk up the directory tree to find the root of the repository
__PROJECT_ROOT__="$__FORWARD_PORTING_ROOT__"
while [[ ! -d "$__PROJECT_ROOT__/.git" && "$__PROJECT_ROOT__" != "/" ]]; do
    __PROJECT_ROOT__="$(dirname "$__PROJECT_ROOT__")"
done
if [[ ! -d "$__PROJECT_ROOT__/.git" ]]; then
    echo "Error: Could not find the root of the repository." >&2
    exit 1
fi

# echo "__FORWARD_PORTING_ROOT__: $__FORWARD_PORTING_ROOT__"
# echo "__PROJECT_ROOT__: $__PROJECT_ROOT__"

# Source the script
source "$__FORWARD_PORTING_ROOT__/forward-port-missing"

# unset pipefail to avoid issues with error handling in tests
set +o pipefail

# Additional test helpers
# (not from bash_unit)
function assert_not_set() {
    local var_name="$1"
    local message="$2"

    if [[ -n "${!var_name:-}" ]]; then
        if [[ -n "$message" ]]; then
            echo "Assertion failed: '$var_name' should not be set. $message" >&2
        else
            echo "Assertion failed: '$var_name' should not be set." >&2
        fi
        assert false
    fi
}

# get the content of stdout and check if it contains the expected text
function assert_stdout_contains() {
    local output
    output=$(timeout 0.1 cat -)
    if [[ $? -ne 0 ]]; then
        echo "Error reading from stdout." >&2
        assert false
    fi
    local expected="$1"
    if [[ "$output" != *"$expected"* ]]; then
        echo "Assertion failed: Output does not contain expected text." >&2
        echo "Expected: '$expected'" >&2
        echo "Actual: '$output'" >&2
        assert false
    fi
}

# function assert_stderr_contains() {
#     local output
#     output=$(timeout 0.1 cat - 2>&1 >/dev/null)
#     if [[ $? -ne 0 ]]; then
#         echo "Error reading from stderr." >&2
#         exit 1
#     fi
#     local expected="$1"
#     if [[ "$output" != *"$expected"* ]]; then
#         echo "Assertion failed: STDERR does not contain expected text." >&2
#         echo "Expected: '$expected'" >&2
#         echo "Actual: '$output'" >&2
#         exit 1
#     fi
# }