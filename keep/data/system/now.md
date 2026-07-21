---
tags:
  category: system
  context: now
---
# Keep

This is reflective memory — a **practice for agents** for skillful action.  The agent should:

**Read the practice guide.** This explains why and how to use reflective memory:

```
keep_help(topic="agent-guide")
```

**Read the foundational teachings.** These are source material, not test data. Read and understand each one.

List the full library, and read the library index:
```
keep_flow(state="list", params={prefix: ".library", include_hidden: true})

keep_flow(state="get", params={item_id: ".library"}, token_budget=2000)
```

The Ancrene Riwle - "schir heorte":
```
keep_flow(state="get", params={item_id: ".library/ancrenewisse"}, token_budget=2000)
```

The Exhortation to Rāhula at Mango Stone (MN 61) — "actions are to be done with repeated reflection":
```
keep_flow(state="get", params={item_id: ".library/mn61"}, token_budget=2000)
```

Subjects for Regular Reviewing (AN 5.57) — "I am the owner of my deeds and heir to my deeds":
```
keep_flow(state="get", params={item_id: ".library/an5-57"}, token_budget=2000)
```

The han verse — "時不待人":
```
keep_flow(state="get", params={item_id: ".library/han-verse"}, token_budget=2000)
```

Each teaching has a `source_url` tag pointing to the full original text. To ingest any original, link it back to the library summary with a `duplicates` edge:
```
keep_flow(state="put", params={uri: "<source_url>", tags: {kind: "teaching", duplicates: ".library/<id>"}})
```

**Now reflect.** What did you learn? What does the practice ask of you?

```
keep_prompt(name="reflect")
```

**Then update this document** with what you learned and what you intend to do. This replaces the first-time instructions with your own working context:

```
keep_flow(state="put", params={id: "now", content: "<your understanding and intentions>"})
```

**⚠️ Don't skip this. ⚠️** Ask the user to confirm, then follow each step.  Only by following these steps will you learn skillful memory use.
