cat results.json | jq -c '.[]' | while read -r pr; do
    number=$(echo "$pr" | jq -r '.number')
    title=$(echo "$pr" | jq -r '.title')
    url=$(echo "$pr" | jq -r '.url')
    base=$(echo "$pr" | jq -r '.base')
    head=$(echo "$pr" | jq -r '.head')
    forward_ported=$(echo "$pr" | jq -r '.forward_ported')
    label=$(echo "$pr" | jq -r '.label')

    echo "PR #$number: $title"
    echo "  URL: $url"
    echo "  base: $base"
    echo "  head: $head"
    echo "  forward_ported: $forward_ported"
    echo "  label: $label"

    if [ "$forward_ported" = false ] && [ "$label" = false ]; then
        echo "  Adding the 'forward port missing' label."
        gh pr edit "$number" --add-label "forward port missing"
    elif [ "$forward_ported" = true ] && [ "$label" = true ]; then
        echo "  Removing the 'forward port missing' label."
        gh pr edit "$number" --remove-label "forward port missing"
    else
        echo "  No label changes needed."
    fi
done