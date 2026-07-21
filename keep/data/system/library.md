---
tags:
  category: system
  type: index
---
# Library

Public domain texts for bootstrapping reflective memory. The content is relevant to the practice — these are seed wisdom, not test fixtures.

## Teachings

| ID | Title | Topic |
|----|-------|-------|
| `.library/mn61` | The Exhortation to Rāhula at Mango Stone (MN 61) | Reflection before, during, and after action |
| `.library/an5-57` | Subjects for Regular Reviewing (AN 5.57) | The five remembrances — ownership of deeds |
| `.library/han-verse` | Han Verse (版の偈) | Impermanence, urgency, heedfulness |
| `.library/true-person-no-rank` | The True Person of No Rank (無位真人) | Direct pointing — who is aware? |
| `.library/fortytwo-chapters` | Sutra of Forty-Two Chapters (佛說四十二章經) | Conduct, desire, and the path |
| `.library/sticks-and-stones` | Sticks and Stones (Lewis Mumford, 1924) | Architecture as civilization — form follows culture |
| `.library/ancrenewisse` | Ancrene Wisse (Ancrene Riwle, c. 1200) | Inner rule and outer rule — conduct of the heart |

## Reading

To read any teaching:
```
keep_flow(state="get", params={item_id: ".library/mn61"}, token_budget=2000)
```

Each teaching has a `source_url` tag pointing to the full original text online. To ingest the full original into your store, link it back to the library summary with a `duplicates` edge:
```
keep_flow(state="put", params={uri: "<source_url>", tags: {kind: "teaching", duplicates: ".library/<id>"}})
```

## Format Diversity

These texts span multiple languages (English, Classical Chinese, Japanese, Middle English, Pali), formats (HTML, JSON, Markdown, plaintext, PDF), lengths (four-line verse to full book), and domains (Buddhist teaching, Zen liturgy, architectural criticism, monastic guidance). This diversity is intentional — it tests document processing while providing foundational content.
