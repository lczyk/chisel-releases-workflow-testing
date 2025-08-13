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
        exit 1
    fi
}