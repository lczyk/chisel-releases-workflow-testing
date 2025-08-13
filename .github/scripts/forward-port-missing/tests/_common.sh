[[ -z "${__COMMON_SH__:-}" ]] && __COMMON_SH__=1 || return 0

__PROJECT_ROOT__="$(dirname "$PWD")"
# add root to PATH so we can source scripts from there
export PATH="$__PROJECT_ROOT__/:$PATH"

source "$__PROJECT_ROOT__/forward-port-missing"

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