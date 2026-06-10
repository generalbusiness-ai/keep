---
tags:
  category: system
  context: state
---
# Search then traverse edges from results. Sequential: skip traverse if empty.
match: sequence
rules:
  - when: "!(has(params.query) && params.query != '' && params.query != null)"
    # Deep search requires a non-empty search string.
    return:
      status: error
      with:
        reason: "query required"
  - id: search
    # Initial semantic search
    do: find
    with:
      query: "{params.query}"
      limit: "{params.limit}"
  - when: "!has(search.count) || search.count == 0"
    return: done
  - id: related
    # Follow edges from search hits; deep_follow_depth controls how many
    # primary items contribute tags to the Tier-2 tag-follow fallback.
    do: traverse
    with:
      items: "{search.results}"
      limit: "{params.deep_limit}"
      deep_follow_depth: "{params.deep_follow_depth}"
  - return: done
