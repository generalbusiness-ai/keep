---
name: keep
description: >
  Reflective Memory. Use when working with memory, reflection, context management,
  or intent tracking. Provides the reflective memory practice — how and when to use
  keep tools for capturing insights, tracking commitments, and maintaining awareness.
---

## keep: Reflective Memory

*Your actions have consequences. Context is fleeting. Pay attention to what is being asked.*

**The practice is `keep_prompt(name="reflect")`.** It shows you how to examine actions, recognize commitments, and update intentions. Run it. Follow the steps. Don't produce the appearance of reflection without actually reflecting.

**Three tools:** `keep_flow` (all operations), `keep_help` (documentation), `keep_prompt` (practice prompts).

```
keep_prompt(name="reflect")                                                               # The practice
keep_flow(state="get", params={item_id: "now"}, token_budget=2000)                        # Current intentions
keep_flow(state="query-resolve", params={query: "topic"}, token_budget=2000)              # What do I know?
keep_flow(state="put", params={content: "what's true now", id: "now"})                    # Update intentions
keep_flow(state="put", params={content: "learning", tags: {kind: "learning"}})            # Capture insight
keep_flow(state="put", params={uri: "https://example.com/doc", tags: {topic: "X"}})       # Index a document
keep_flow(state="tag", params={id: "item", tags: {reviewed: "true"}})                     # Tag an item
keep_help(topic="flow-actions")                                                            # Action reference
```

**No MCP?** Every flow has a CLI equivalent: `keep flow get -p item_id=now --token-budget 2000`, `keep flow query-resolve -p query=X --token-budget 2000`, `keep flow put -p content=X`. CLI `--budget` limits execution ticks; `--token-budget` limits rendered response text. Use Bash to run them.

**Protocol:** Reflect using `keep` tools — before, during, and after action. Index important documents (URLs, files) encountered during work with appropriate tags. When writing a plan, incorporate this practice into the plan itself. If the practice is unclear, use `keep_help(topic="index")`.
