# OPML Import and Feed Refresh

**Status:** design draft
**Date:** 2026-04-10

## Summary

Add a first-class OPML import flow that turns a subscription export into
structured keep notes, then runs daemon-driven feed refresh in the
background to fetch recent entries and surface new relevant reading.

This is not "generic XML summarization". OPML, RSS, and Atom are
structured formats. keep should parse them into readable markdown-like
content and metadata first, then let the existing summarization,
embedding, tagging, and search machinery operate on that normalized
representation.

## Problem

Today, `keep put` can ingest local XML files and remote feed URLs only as
generic text.

That is not enough for subscriptions:

- An OPML file contains feed structure, groups, feed URLs, site URLs, and
  display names, but generic ingest does not turn those into separate
  notes.
- Feed XML changes over time and needs periodic polling.
- Users care about feed *entries* and linked article content, not the raw
  feed document.
- "New and relevant" should use keep's normal semantic retrieval, not a
  disconnected reader inbox.

The real sample file at `~/Documents/20260410-subscriptions.opml`
contains 122 feed URLs and 6 named top-level groups, so this feature must
handle a substantial subscription set and nested OPML structure.

## Goals

- Import OPML as a real feature, not as an accidental side effect of
  generic XML ingest.
- Preserve OPML grouping and feed metadata.
- Create stable notes for subscriptions and entries.
- Poll feeds in the background and ingest new entries incrementally.
- Fetch article content from entry URLs when available, so summaries are
  based on readable article text rather than feed snippets alone.
- Surface recent feed content through the existing semantic search and
  context machinery.
- Reuse existing daemon, work queue, URI fetch, summarize, analyze, and
  tag paths wherever they already fit.

## Non-goals

- Building a full RSS reader UI in the first slice.
- Designing a separate ranking engine just for feeds.
- Storing raw feed XML as the primary user-facing representation.
- Treating every feed as a normal URL watch by default.

## Evidence from the current code

Current behavior is close in a few places, but the actual feature is
missing:

- Local file ingest recognizes `.xml` but not `.opml`, and generic text
  files are read verbatim.
- Remote HTTP ingest special-cases HTML and some binary formats, but not
  RSS/Atom/OPML.
- After-write link extraction runs for markdown, HTML, email, PDF, DOCX,
  and PPTX, but not XML content types.
- URL watching already exists, but it only detects source changes and
  re-runs `put()` on that URL.
- The watch system defaults to a maximum of 100 watches, while the sample
  OPML contains 122 feeds.

This makes the design direction clear:

1. OPML/feed parsing must be explicit and structured.
2. Feed refresh should be a dedicated daemon capability.
3. Relevance should come from ordinary keep retrieval over entry/article
   notes.

## Design

### Core idea

Split the feature into three note layers:

1. **OPML import note**
   A note representing the imported subscription list.
2. **Feed notes**
   One note per subscription/feed URL.
3. **Entry/article notes**
   One note per feed entry, ideally keyed by article URL.

The daemon periodically refreshes feed notes, discovers new entries, and
upserts entry/article notes. Those notes then participate in normal keep
search, context, and meta resolution.

### Why not just summarize the XML?

Because the structure matters.

For OPML:
- groups
- feed display name
- `xmlUrl`
- `htmlUrl`

For RSS/Atom:
- feed title
- site URL
- entry title
- entry URL
- GUID
- published timestamp
- author
- summary/content

If keep asks an LLM to infer that structure from raw XML, we lose stable
IDs, dedupe keys, timestamps, and precise graph links. The right move is:

1. parse XML deterministically
2. normalize to readable content
3. store tags/metadata
4. summarize only if still useful

Special summarization prompts may still help later, but they are not the
foundation.

## Note model

### 1. OPML import note

The imported OPML file should produce a normalized note representing the
subscription set.

**Suggested ID**

- default: the file URI of the imported OPML file
- example: `file:///Users/hugh/Documents/20260410-subscriptions.opml`

This fits keep's current URI-backed note model and versions naturally if
the file changes.

**Suggested tags**

- `type=feed_import`
- `feed_format=opml`
- `subscription_count=122`
- `feed_group_count=6`
- `import_source=file`

**Suggested content**

Normalized markdown, for example:

```md
# Subscriptions

- 122 feeds
- 6 named groups

## Ungrouped
- Armin Ronacher's Thoughts and Writings
  - feed: https://lucumr.pocoo.org/feed.atom
  - site: https://lucumr.pocoo.org/

## AI
- Cory Doctorow - Pluralistic
  - feed: https://pluralistic.net/feed/
  - site: https://pluralistic.net/
```

This makes the import note readable and searchable without storing raw
XML as the canonical body.

### 2. Feed notes

Each subscription becomes a stable note, keyed by the feed URL.

**Suggested ID**

- the `xmlUrl` / feed URL itself

This is already compatible with keep's URI-centric model and makes the
feed URL the stable anchor for refresh.

**Suggested tags**

- `type=feed`
- `feed_url=<xmlUrl>`
- `site_url=<htmlUrl>` when present
- `feed_title=<display title>`
- `feed_group=<group name>` for each OPML group in the path
- `subscription_source=<opml note id>`
- `feed_active=true`
- `feed_format=rss|atom|unknown`
- `feed_poll_interval=PT1H` or similar user-visible interval tag

**Suggested content**

Normalized feed description, for example:

```md
# Keep Blog

- feed: https://generalbusiness.ai/blog/feed.xml
- site: https://generalbusiness.ai/blog/
- groups: AI

Recent entries:
- 2026-04-09: ...
- 2026-04-01: ...
```

The note body should be derived from parsed feed metadata, not raw XML.

### 3. Entry/article notes

These are the notes the user will actually want surfaced.

**Preferred ID**

- article URL / entry link URL

**Fallback ID**

- a synthetic stable ID when no canonical article URL exists, for example:
  `feed-entry:{feed_hash}:{entry_key_hash}`

The fallback key should be derived from entry GUID first, then link, then
title+published if necessary.

**Suggested tags**

- `type=feed_entry`
- `feed_url=<feed note id>`
- `site_url=<site url>` when known
- `feed_title=<feed title>`
- `feed_group=<group name>` copied from the source feed note
- `published=<timestamp>`
- `author=<author>` when known
- `entry_guid=<guid>` when present
- `entry_title=<title>`
- `entry_source=feed`
- `subscription_source=<opml note id>`

**Body strategy**

Two cases:

1. **Entry has article URL**
   Use the existing URI ingest path for the article URL, with feed-derived
   tags merged in.

2. **Entry has no article URL or article fetch fails**
   Store normalized entry text from the feed payload itself.

This lets keep summarize readable article content when available, while
still indexing feed-only items.

## Scheduling model

### Do not use one normal URL watch per feed by default

Plain URL watches are too weak a model for subscriptions:

- they detect change but do not parse entries
- they re-put the feed URL itself, not the articles
- the current default watch cap is too low for the sample OPML
- the default watch cadence is tuned for generic source monitoring, not
  humane feed polling

### Add a dedicated feed-refresh scheduler

Use the daemon's existing timer pattern, but with a feed-specific queue
replenishment step.

Suggested daemon timer:

- `feed-refresh`

Suggested default interval:

- `PT1H`

This should behave more like supernode replenishment than like watches:

1. daemon checks which feeds are due
2. enqueue refresh work for those feeds
3. each refresh parses the feed and emits entry/article work
4. timer state records last run, next run, and summary

### Feed registry storage

Introduce a dedicated system-doc hierarchy for scheduled feeds:

```text
.feeds
.feeds/PT6H
.feeds/P1D
```

This mirrors the existing watch and markdown-mirror patterns while
keeping feed scheduling separate from generic URL watching.

Each entry should store:

- `feed_url`
- `site_url`
- `title`
- `groups`
- `interval`
- `last_checked`
- `last_success`
- `last_modified`
- `etag`
- `last_error`
- `last_entry_published`
- `subscription_source`
- `active`

## Refresh flow

### Import phase

`keep import opml PATH` should:

1. parse the OPML file
2. create or update the OPML import note
3. create or update feed notes
4. register those feeds in `.feeds` or `.feeds/<interval>`
5. optionally enqueue an immediate first refresh

### Feed refresh phase

A feed refresh task should:

1. fetch the feed with conditional HTTP headers when available
2. reject private/internal destinations under the same SSRF rules as
   normal HTTP ingest
3. parse RSS or Atom deterministically
4. update feed note metadata/content
5. identify new or changed entries
6. upsert entry/article notes
7. update feed refresh state (`etag`, `last_modified`, `last_success`,
   newest seen entry timestamp)

### Entry/article ingest phase

For each new or changed entry:

1. build the stable note ID
2. if an article URL exists, `put(uri=article_url, id=article_url, ...)`
   with feed tags
3. otherwise `put(content=normalized_entry_text, id=<synthetic id>, ...)`
4. preserve published timestamp in tags and, when safe, in `created_at`
5. avoid repeated churn when nothing meaningful changed

## Relevance and surfacing

### Core principle

Feed content should become ordinary keep notes.

That means "new and relevant" can come from the search and context system
we already have, rather than a bespoke ranking subsystem.

### Initial surfacing model

Add a meta-doc or state-doc-backed retrieval path for recent feed entries:

- constrain to `type=feed_entry`
- constrain by recency, for example `since=P7D`
- rank by similarity to the current note or `now`

That yields:

- "new and relevant to what I'm doing now"
- "recent feed content about this topic/project/person"

without inventing a second retrieval stack.

### Why this is the right first slice

- It reuses current embeddings and semantic search.
- It respects existing user tags and context.
- It keeps feed material visible through normal `find`, `get`, `now`, and
  meta sections.

## Content types and prompts

### Content types

Preserve real MIME where known, but add explicit feed tags.

Examples:

- OPML file note: `_content_type=text/x-opml+xml` or
  `_content_type=application/xml`
- RSS/Atom fetch: preserve actual HTTP MIME such as
  `application/rss+xml`, `application/atom+xml`, or `text/xml`
- article note: whatever the existing article fetch path detects

The important semantic classifier should be a user tag like
`feed_format=opml|rss|atom`, not only MIME.

### Prompts

Prompt specialization is optional and secondary.

If needed later, add:

- `.prompt/summarize/feed`
- `.prompt/summarize/feed-entry`
- `.prompt/summarize/opml`

But these should operate on normalized content, not on raw XML.

## Dedupe and update behavior

### Feed notes

Feed notes should update in place by feed URL.

### Entry/article notes

When keyed by article URL:

- multiple feeds pointing to the same article converge on one note
- feed-derived tags merge
- the note participates naturally in search and versions

This is desirable in the common case.

When a synthetic entry ID is required:

- the synthetic key must be stable across refreshes
- refresh should update the same note, not create duplicates

### Change detection

Refresh should not create note churn for every poll.

Use:

- `ETag` / `Last-Modified` at feed fetch time
- stable entry IDs
- content hashes or comparable refresh markers when deciding whether to
  update an entry note

## Failure modes and safety

### HTTP safety

Feed fetch must honor the same private-network and redirect safety
boundaries as current HTTP ingest. Do not introduce a parallel unsafe
HTTP path for feed parsing.

### Large or noisy feeds

Feeds with hundreds of old items should not fan out into an unbounded
first sync by default.

Proposal:

- first refresh ingests only the most recent `N` entries per feed
- later refreshes ingest incrementally based on seen IDs/timestamps

### Feed outages

A failing feed should record `last_error` and remain scheduled unless the
user disables it. One broken feed must not block the refresh cycle.

### Deleted subscriptions

Removing a feed from a later OPML import should not delete existing entry
notes automatically in the first slice. Instead:

- mark the feed inactive
- preserve historical entry/article notes

## Implementation plan

### Phase 1: OPML import and feed registry

1. Add an explicit `import opml` command surface.
2. Parse OPML into a normalized import note.
3. Create stable feed notes from `xmlUrl` records.
4. Preserve OPML groups as feed tags.
5. Register feeds in `.feeds`.

### Phase 2: Daemon feed refresh

1. Add feed timer state and queue replenishment.
2. Add feed refresh work items.
3. Fetch feeds with conditional HTTP and safe redirect handling.
4. Parse RSS/Atom and update feed notes.

### Phase 3: Entry/article ingest

1. Upsert entry/article notes.
2. Prefer article URL ingest when available.
3. Fall back to normalized feed entry content when necessary.
4. Tag entries for recency and lineage.

### Phase 4: Relevance surfacing

1. Add recent-feed meta retrieval.
2. Surface `type=feed_entry` notes through `now` and related contexts.
3. Tune recency defaults and limits against real stores.

## Open questions

- Should OPML import update an existing feed's group tags additively, or
  should the newest import replace prior OPML-derived grouping?
  Proposal: replace OPML-derived groups from that import source, preserve
  unrelated user tags.

- Should article notes be tagged with a single `feed_group` value or allow
  multiple values when the same feed appears in multiple OPML groups?
  Proposal: allow multiple values.

- Should first refresh ingest only a bounded number of items per feed?
  Proposal: yes, default to a recent window.

- Should feeds live in `.feeds` system docs or reuse `.watches` with a
  `kind=feed` entry?
  Proposal: use `.feeds`; feed refresh semantics are materially different.

- Should keep store the raw XML anywhere for debugging?
  Proposal: not by default. If needed later, store short diagnostic
  metadata, not the entire feed payload.

## Recommendation

Build this as a dedicated import-and-refresh feature with structured
parsing and daemon scheduling.

The first implementation should optimize for:

- stable note identities
- reuse of existing article ingest
- safe incremental refresh
- ordinary keep retrieval over fresh feed entries

That gives the user what they actually want: subscriptions that continue
to feed keep with new material, and retrieval that can surface the new
reading most relevant to the current work.
