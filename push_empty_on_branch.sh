#!/bin/bash

# create an empty commit with in a local branch,
# and push it to the remote.
# Useful for quickly sending a commit to a testing PR branch
# with
function _fatal() {
    echo "Error: $1" >&2
    exit 1
}

function main() {
    local branch="$1"
    echo "Pushing an empty commit to branch: $branch"
    if [ -z "$branch" ]; then
        if command -v fzf &>/dev/null; then
            # try to find a branch using fzf
            branches=$(git branch --format='%(refname:short)')
            branch=$(echo "$branches" | fzf --prompt="Select a branch: " --height=10)
        else
            _fatal "No branch specified and fzf is not available. Please specify a branch as an argument or install fzf."
        fi
    fi

    
    [ -z "$branch" ] && \
        _fatal "No branch selected. Exiting."
    echo "Selected branch: $branch"

    git branch | grep -q "\* $branch" &>/dev/null && \
        _fatal "Branch $branch is currently checked out. Please switch to another branch."

    git branch | grep -q "+ $branch" &>/dev/null && \
        _fatal "Branch $branch is checked out in another worktree. Please switch to another branch."

    local worktree_dir=$(mktemp -d -t "worktree-XXXXXX")
    git worktree add "$worktree_dir" "$branch" || \
        _fatal "Failed to create a worktree for branch $branch."
    (
        cd "$worktree_dir" || exit 1
        # message should be empty + date
        local message="auto commit on branch $branch at $(date +'%Y-%m-%d %H:%M:%S')"
        git commit --allow-empty -m "$message"
        git push
    ) || \
        _fatal "Failed to push empty commit to branch $branch."
    
    git worktree remove "$worktree_dir" || \
        _fatal "Failed to remove worktree directory $worktree_dir. You should remove it manually."
}

main "$@"