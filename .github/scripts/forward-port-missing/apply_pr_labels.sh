#!/bin/bash
# usage: cat results.json | ./apply_pr_labels.sh --dry-run/-n

dry_run=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run)
            dry_run=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done


_maybe_dry_run() {
    if [ "$dry_run" = true ]; then
        # squash multiple spaces into one
        echo "> $(echo "$1" | tr -s ' ')"
    else
        eval "$1"
    fi
}

jq -c '.[]' | while read -r pr; do
    number=$(echo "$pr" | jq -r '.number')
    title=$(echo "$pr" | jq -r '.title')
    url=$(echo "$pr" | jq -r '.url')
    base=$(echo "$pr" | jq -r '.base')
    head=$(echo "$pr" | jq -r '.head')
    forward_ported=$(echo "$pr" | jq -r '.forward_ported')
    label=$(echo "$pr" | jq -r '.label')

    echo "PR #$number: $title"
    echo "  $url"
    echo "  $head -> $base"
    echo "  forward_ported: $forward_ported"
    echo "  has label: $label"

    if [ "$forward_ported" = false ] && [ "$label" = false ]; then
        echo "  Adding the 'forward port missing' label."
        _maybe_dry_run "gh pr edit $number --add-label \"forward port missing\""
    elif [ "$forward_ported" = true ] && [ "$label" = true ]; then
        echo "  Removing the 'forward port missing' label."
        _maybe_dry_run "gh pr edit $number --remove-label \"forward port missing\""
    else
        echo "  No label changes needed."
    fi
done