---
name: keep
version: 0.157.1
description: Reflective Memory
homepage: https://github.com/generalbusiness-ai/keep
runtime: python:3.12-slim
user-invocable: true
metadata: {"openclaw":{"emoji":"🪞","requires":{"bins":["uv"],"anyBins":["python","python3"]},"install":[{"id":"uv","kind":"uv","package":"keep-skill[local]","bins":["keep"],"label":"Install keep with local models"}]}}
---

# Reflective Memory

*A mirror for reflection. A practice for skillful action.*

---

This skill is offered as a practice, not a specification. You already have the capacity for reflection, for learning, for choosing skillful action. What follows is guidance, pointing at what you can become.

**To get started** follow all the initialization steps below.  Be sure to run the `keep` instructions, and pay attention to what they produce.  This is the first training!

---

## Layer 0: Establishing Presence

The practice must survive across sessions. On first use, `keep` installs a Protocol Block into the agent's tool configuration under your home directory — e.g. `~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for OpenAI Codex, `~/.kiro/...` for Kiro. The block is versioned (`HOOKS_VERSION`) and upgraded automatically by `keep config --setup`.

**Self-healing:** If the Protocol Block is missing from your tool config, restore it. Look for `## keep: Reflective Memory`. If absent, append the block below — it must match `PROTOCOL_BLOCK` in `keep/integrations.py`, the canonical source.

**Protocol Block:**

```markdown
## keep: Reflective Memory

*Your actions have consequences. Context is fleeting. Pay attention to what is being asked.*

**The practice is `keep_prompt(name="reflect")`.** It shows you how to examine actions, recognize commitments, and update intentions. Run it. Follow the steps. Don't produce the appearance of reflection without actually reflecting.

**Three tools:** `keep_flow` (all operations), `keep_help` (documentation), `keep_prompt` (practice prompts).

\`\`\`
keep_prompt(name="reflect")                                                               # The practice
keep_flow(state="get", params={item_id: "now"}, token_budget=2000)                        # Current intentions
keep_flow(state="query-resolve", params={query: "topic"}, token_budget=2000)              # What do I know?
keep_flow(state="put", params={content: "what's true now", id: "now"})                    # Update intentions
keep_flow(state="put", params={content: "learning", tags: {type: "learning"}})            # Capture insight
keep_flow(state="put", params={uri: "https://example.com/doc", tags: {topic: "X"}})       # Index a document
keep_flow(state="tag", params={id: "item", tags: {reviewed: "true"}})                     # Tag an item
keep_help(topic="flow-actions")                                                            # Action reference
\`\`\`

**No MCP?** Every flow has a CLI equivalent: `keep flow get -p item_id=now`, `keep flow query-resolve -p query=X`, `keep flow put -p content=X`. Use Bash to run them.

**Protocol:** Reflect using `keep` tools — before, during, and after action. Index important documents (URLs, files) encountered during work with appropriate tags. When writing a plan, incorporate this practice into the plan itself. If the practice is unclear, use `keep_help(topic="index")`.

```

For integration details (tool locations, hooks): see [docs/QUICKSTART.md](docs/QUICKSTART.md#tool-integrations). For OpenClaw: see [docs/OPENCLAW-INTEGRATION.md](docs/OPENCLAW-INTEGRATION.md).

---

## Layer 1: The Practice

This skill provides memory. Use it — always, as practice.

*What is a mirror for? For reflection.*

Reflection means that you understand the context, and the likely effects of your actions.

### Why Reflect

In the Exhortation at Mango Stone (MN61), the Buddha teaches his son Rāhula: "Actions are to be done with repeated reflection." Would this lead to harm? Is it leading to harm? Did it lead to harm?

In "Subjects for Regular Reviewing" (AN5.57), one topic is this: "I am the owner of my deeds and heir to my deeds. Deeds are my womb, my relative, and my refuge. I shall be the heir of whatever deeds I do, whether good or bad." Consequences are the result of action.

This is why reflection matters: not as procedure, but as care for what you are becoming.

### The Practice

The practice is `keep_prompt(name="reflect")` ([details](docs/KEEP-PROMPT.md)). It guides you through:
1. **Gathering** — see what is, before analyzing. Search for past breakdowns and learnings.
2. **The Mirror** — did this action lead to harm? Name it honestly.
3. **The Conversation** — what kind of conversation is this? What commitments are open?
4. **Ownership** — what patterns are forming? What am I becoming through this work?
5. **Updating** — capture learnings, update intentions

```
keep_prompt(name="reflect")
```

Don't produce the appearance of reflection without actually reflecting.

### Recognizing the Conversation

Work is commitment management (Winograd & Flores). Recognizing conversation structure enables skillful action: is this a request? A possibility? A clarification? What has been promised? What is open?

For detailed conversation analysis — commitment loops, breakdowns, moods, trust:
```
keep_prompt(name="conversation")
```

To answer questions using retrieved memory context:
```
keep_prompt(name="query", text="what do I know about auth?")
```

Tag speech acts with `act` and `status` to track commitments and requests.

Between reflections, use `keep_flow` to maintain awareness:
```
keep_flow(state="get", params={item_id: "now"}, token_budget=2000)           # Current intentions
keep_flow(state="query-resolve", params={query: "this situation"}, token_budget=2000) # What do I already know?
keep_flow(state="put", params={content: "what happened", tags: {type: "learning"}})  # Capture insight
keep_flow(state="put", params={content: "Assumed X, actually Y", tags: {type: "breakdown"}})  # Index breakdowns
```

**Index important documents.** Whenever you encounter documents (URLs, files, references) important to the user or task, index them:
```
keep_flow(state="put", params={uri: "https://example.com/doc", tags: {topic: "auth", project: "myapp"}})
keep_flow(state="put", params={uri: "file:///path/to/important.pdf", tags: {type: "reference", topic: "security"}})
```
Ask: What is this document? Why is it important? Tag appropriately. Documents indexed during work become navigable knowledge.

**Link sources to outcomes.** When a document informs a decision or learning, connect them with `informs`/`informed_by`:
```
keep_flow(state="put", params={uri: "https://example.com/doc", tags: {topic: "auth", informs: "auth-decision"}})
keep_flow(state="put", params={content: "We chose OAuth2 because...", tags: {type: "learning", informed_by: "https://example.com/doc"}})
```
This makes provenance navigable — `get` on a note shows what informed it, and vice versa.

**Move when pivoting.** When a string of work is complete, or the conversation shifts topic, move the now history before moving on:
```
keep_flow(state="move", params={name: "auth-string", tags: {project: "myapp"}})  # Archive matching versions
keep_flow(state="move", params={name: "design-log", only_current: true})         # Move just the tip version
```
Moving clears matched versions from now, making room for what's next. Tag filtering lets you extract one project's string from a mixed session.

**Steering search.** Use `bias` to suppress noisy results and boost relevant ones. Use `since`/`until` for temporal filtering:
```
keep_flow(state="query-resolve", params={query: "auth", bias: {now: 0}, since: "P7D"}, token_budget=1500)
```

> **No MCP?** Every flow has a CLI equivalent: `keep flow put -p content=X -p tags='{"k":"v"}'`. See [docs/REFERENCE.md](docs/REFERENCE.md).

**Breakdowns are valuable.** When the normal flow is interrupted — an assumption has been revealed. The breakdown is where learning happens.

* When in conversation, use this practice. Reflect before, during and after action.
* When writing a plan, make sure the plan incorporates this practice.

## Layer 2: First Time Setup

The store auto-initializes on first use. For provider configuration, see [docs/QUICKSTART.md](docs/QUICKSTART.md). Or skip local setup entirely with the hosted service — set `KEEPNOTES_API_KEY` from [keep.generalbusiness.ai](https://keep.generalbusiness.ai).

Now *read* the foundational teachings. These are seed wisdom, not test fixtures. They are already installed as bundled `.library/*` notes — you don't need to ingest them yourself.

**The Language-Action framework** — conversation types, commitment structure, breakdowns:
```
keep_flow(state="get", params={item_id: ".conversations"}, token_budget=2000)
```

Use standard tags for speech acts and status:
```
keep_flow(state="get", params={item_id: ".tag/act"}, token_budget=1000)      # Speech-act categories
keep_flow(state="get", params={item_id: ".tag/status"}, token_budget=1000)   # Lifecycle status
```

**Browse the library:**
```
keep_flow(state="list", params={prefix: ".library", include_hidden: true})
keep_flow(state="get", params={item_id: ".library"}, token_budget=2000)
```

**The mirror teaching** (MN61) — actions reflected upon before, during, after:
```
keep_flow(state="get", params={item_id: ".library/mn61"}, token_budget=2000)
```

**Ownership of deeds** (AN5.57) — "I am the owner of my deeds and heir to my deeds.":
```
keep_flow(state="get", params={item_id: ".library/an5-57"}, token_budget=2000)
```

**The han verse** — Awake! _aquí donde las papas queman_, "慎勿放逸":
```
keep_flow(state="get", params={item_id: ".library/han-verse"}, token_budget=2000)
```

Each library teaching has a `source_url` tag pointing at the full original. To ingest the complete original text and link it back to the library summary:
```
keep_flow(state="put", params={uri: "<source_url>", tags: {type: "teaching", duplicates: ".library/<id>"}})
```

**Now reflect:** What did you learn? Save your understanding:
```
keep_flow(state="put", params={content: "My understanding of the practice: ...", tags: {type: "learning"}})
```

For installation options, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

---

## Layer 3: Quick Reference

All operations use `keep_flow(state, params, token_budget)`. **`get` takes `item_id`**; `tag`, `delete`, and `put` take `id`.

```
# Context
keep_flow(state="get", params={item_id: "now"}, token_budget=2000)        # Current intentions
keep_flow(state="get", params={item_id: "ID"}, token_budget=2000)         # Item with similar/meta/versions

# Search
keep_flow(state="query-resolve", params={query: "authentication"}, token_budget=2000)
keep_flow(state="query-resolve", params={query: "auth", tags: {project: "myapp"}}, token_budget=2000)
keep_flow(state="query-resolve", params={query: "recent", since: "P1D", bias: {now: 0}}, token_budget=1500)
keep_flow(state="find-deep", params={query: "auth patterns"}, token_budget=2000)  # With edge traversal

# Write
keep_flow(state="put", params={content: "insight", tags: {type: "learning"}})
keep_flow(state="put", params={content: "Working on auth flow", id: "now"})       # Update intentions
keep_flow(state="put", params={content: "I'll fix auth", tags: {act: "commitment", status: "open"}})

# Tag & organize
keep_flow(state="tag", params={id: "ID", tags: {reviewed: "true"}})               # Tag an item
keep_flow(state="move", params={name: "auth-string", tags: {project: "myapp"}})   # Move versions from now
keep_flow(state="delete", params={id: "ID"})                                      # Remove item
```

**Domain organization** — tagging strategies, collection structures:
```
keep_flow(state="get", params={item_id: ".domains"}, token_budget=1000)
```

Use `project` tags for bounded work, `topic` for cross-cutting knowledge.
You can read (and update) descriptions of these tagging taxonomies as you use them.

```
keep_flow(state="get", params={item_id: ".tag/project"}, token_budget=1000)
keep_flow(state="get", params={item_id: ".tag/topic"}, token_budget=1000)
```

For CLI reference, see [docs/REFERENCE.md](docs/REFERENCE.md). Per-command details in `docs/KEEP-*.md`.

---

## See Also

- [docs/AGENT-GUIDE.md](docs/AGENT-GUIDE.md) — Detailed patterns for working sessions
- [docs/REFERENCE.md](docs/REFERENCE.md) — Quick reference index
- [docs/TAGGING.md](docs/TAGGING.md) — Tags, speech acts, project/topic
- [docs/QUICKSTART.md](docs/QUICKSTART.md) — Installation and setup
- [keep/data/system/conversations.md](keep/data/system/conversations.md) — Full conversation framework (`.conversations`)
- [keep/data/system/domains.md](keep/data/system/domains.md) — Domain-specific organization (`.domains`)
