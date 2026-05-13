"""Analyzer implementations for document decomposition.

Two implementations:
- SlidingWindowAnalyzer (default): Token-budgeted sliding windows with
  XML-style target marking. Works well with small local models.
- SinglePassAnalyzer: Single-pass LLM decomposition with JSON output.
  Better for large-context models.
"""

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable

from .compute_context import apply_context_attrs, current_counters
from .providers.base import AnalysisChunk, AnalyzerProvider, get_registry, strip_summary_preamble
from .tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer("flow")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate — chars / 4 for English text."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Line range extraction for URI-sourced documents
# ---------------------------------------------------------------------------

def _extract_line_ranges(source_content: str, parts: list[dict]) -> list[dict]:
    """Add _start_line and _end_line to parts by matching against source content.
    
    Strategy for markdown files:
    - Split source into sections by heading markers (# ## ### etc.)
    - Parts are ordered to match document structure (part 1 = first section, etc.)
    - Match each part summary's leading text against section headings
    - Fallback: assign sections sequentially to parts
    
    For non-markdown or when matching fails, parts get file-level line ranges
    (startLine=1, endLine=total_lines).
    
    Args:
        source_content: The full source document content
        parts: List of part dicts with "summary" keys
    
    Returns:
        The same parts list with "_start_line" and "_end_line" string tags added
    """
    if not parts or not source_content:
        return parts
    
    lines = source_content.splitlines()
    total_lines = len(lines)
    
    # Try to extract markdown sections
    sections = _extract_markdown_sections(lines)
    
    # If no sections found or sections don't match parts count reasonably,
    # fall back to file-level ranges
    if not sections or len(sections) < len(parts):
        logger.info(
            "Line ranges: using file-level fallback (sections=%d, parts=%d)",
            len(sections), len(parts)
        )
        for part in parts:
            if "tags" not in part:
                part["tags"] = {}
            part["tags"]["_start_line"] = "1"
            part["tags"]["_end_line"] = str(total_lines)
        return parts
    
    # Try to match parts to sections
    section_assignments = _match_parts_to_sections(parts, sections)
    
    # Apply line ranges to parts
    for i, part in enumerate(parts):
        if "tags" not in part:
            part["tags"] = {}
        
        if i < len(section_assignments) and section_assignments[i] is not None:
            section = sections[section_assignments[i]]
            part["tags"]["_start_line"] = str(section["start_line"])
            part["tags"]["_end_line"] = str(section["end_line"])
        else:
            # Fallback to file-level range
            part["tags"]["_start_line"] = "1"
            part["tags"]["_end_line"] = str(total_lines)
    
    return parts


def _extract_markdown_sections(lines: list[str]) -> list[dict]:
    """Extract markdown sections defined by headings.
    
    Returns:
        List of section dicts with keys: title, start_line, end_line (1-indexed)
    """
    import re
    
    sections = []
    # Only match headings that start at the beginning of the line (no indentation)
    heading_pattern = re.compile(r'^#{1,6}\s+(.+)$')
    
    current_section = None
    
    for i, line in enumerate(lines, 1):  # 1-indexed
        # Don't strip leading whitespace - we want to check for indentation
        match = heading_pattern.match(line)
        if match:
            # End previous section
            if current_section is not None:
                current_section["end_line"] = i - 1
                sections.append(current_section)
            
            # Start new section
            current_section = {
                "title": match.group(1).strip(),
                "start_line": i,
                "end_line": len(lines)  # Will be updated when next section starts or at EOF
            }
    
    # Close final section
    if current_section is not None:
        current_section["end_line"] = len(lines)
        sections.append(current_section)
    
    return sections


def _match_parts_to_sections(parts: list[dict], sections: list[dict]) -> list[int | None]:
    """Match part summaries to markdown sections.
    
    Strategy:
    1. Check if part summary starts with or contains section title
    2. Sequential fallback: part N -> section N
    
    Returns:
        List of section indices for each part, or None for no match
    """
    assignments = [None] * len(parts)
    used_sections = set()
    
    # First pass: try to match by content similarity
    for i, part in enumerate(parts):
        summary = part.get("summary", "").lower()
        if not summary:
            continue
        
        best_match = None
        best_score = 0
        
        for j, section in enumerate(sections):
            if j in used_sections:
                continue
            
            section_title = section["title"].lower()
            
            # Check if summary starts with or contains section title
            if summary.startswith(section_title):
                score = len(section_title) / len(summary) * 2  # Higher weight for starts-with
            elif section_title in summary:
                score = len(section_title) / len(summary)
            elif any(word in summary for word in section_title.split() if len(word) > 3):
                # Partial word match for longer words
                matching_words = [w for w in section_title.split() if len(w) > 3 and w in summary]
                score = len(matching_words) / len(section_title.split()) * 0.5
            else:
                score = 0
            
            if score > best_score and score > 0.1:  # Minimum threshold
                best_match = j
                best_score = score
        
        if best_match is not None:
            assignments[i] = best_match
            used_sections.add(best_match)
    
    # Second pass: sequential assignment for unmatched parts
    section_idx = 0
    for i, assignment in enumerate(assignments):
        if assignment is None:
            # Find next available section
            while section_idx < len(sections) and section_idx in used_sections:
                section_idx += 1
            if section_idx < len(sections):
                assignments[i] = section_idx
                used_sections.add(section_idx)
                section_idx += 1
    
    return assignments


# ---------------------------------------------------------------------------
# Keyword passage extraction (fallback when no part match)
# ---------------------------------------------------------------------------

def _find_best_passage(
    source_content: str,
    query: str,
    window: int = 8,
    overlap: int = 4,
) -> dict | None:
    """Find the best-matching passage in a file by keyword overlap.

    Used as a fallback when semantic search matches a file but no specific
    part was matched via FTS. Scans the file with a sliding window and
    returns the window with the highest concentration of query terms.

    Args:
        source_content: Full file content
        query: Search query string
        window: Number of lines per passage window
        overlap: Overlap between windows (step = window - overlap)

    Returns:
        Dict with start_line, end_line, snippet (1-indexed lines),
        or None if no terms match at all.
    """
    if not source_content or not query:
        return None

    lines = source_content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return None

    # Extract query terms (lowercase, skip very short words)
    terms = set()
    for t in re.split(r'\W+', query.lower()):
        if len(t) >= 2:
            terms.add(t)
    if not terms:
        return None

    best_score = 0
    best_start = 0
    step = max(1, window - overlap)

    for i in range(0, total_lines, step):
        end = min(i + window, total_lines)
        window_text = " ".join(lines[i:end]).lower()
        score = sum(1 for t in terms if t in window_text)
        if score > best_score:
            best_score = score
            best_start = i

    if best_score == 0:
        return None

    end = min(best_start + window, total_lines)

    # Snap to nearest heading if within 2 lines (prefer section boundaries).
    # Only snap if the heading is within the window — don't pull in a
    # distant top-level heading that isn't part of the match.
    for offset in range(1, min(3, best_start + 1)):
        check = best_start - offset
        if check >= 0 and re.match(r'^#{1,6}\s', lines[check]):
            best_start = check
            end = min(best_start + window, total_lines)
            break

    snippet = "\n".join(lines[best_start:end])
    return {
        "start_line": str(best_start + 1),  # 1-indexed
        "end_line": str(end),
        "snippet": snippet,
    }


# ---------------------------------------------------------------------------
# Sliding-window output parser
# ---------------------------------------------------------------------------

# Preamble patterns that small models emit despite instructions
_PREAMBLE_RE = re.compile(
    r"^here are the (?:significant )?(?:developments|observations|summaries)[^:]*:\s*$",
    re.IGNORECASE,
)


def _parse_parts(text: str) -> list[dict]:
    """Parse one-summary-per-line output from LLM.

    Handles common LLM quirks: preamble lines, leaked XML tags,
    EMPTY sentinel, and very short lines.
    """
    if not text:
        return []

    results = []
    for line in text.strip().splitlines():
        line = strip_summary_preamble(line.strip())
        # Strip leaked XML tags
        line = re.sub(r"</?(?:analyze|content)>", "", line).strip()
        if not line or line == "EMPTY" or len(line) < 20:
            continue
        if _PREAMBLE_RE.match(line):
            continue
        results.append({"summary": line})
    return results


# ---------------------------------------------------------------------------
# Default analysis prompt — fallback when no .prompt/analyze/* doc matches
# ---------------------------------------------------------------------------

DEFAULT_ANALYSIS_PROMPT = """Extract what is *in* this document, section by section. Content is wrapped in <content> tags; only extract from content inside <analyze> tags.

Write ONE LINE per substantive section or distinct topic. Each line states what the section actually contains — the specific things it says — not your judgment of it.

Rules:
- One observation per line, no numbering, no bullets, no preamble
- Synthesize in your own words — never copy or quote the original text
- Be specific: name the actual things discussed (datasets, methods, claims, numbers, entities), not abstract categories
- Use the document's own headings as anchors when visible
- Do not analyze, critique, recommend, or comment on the writing
- Do not start lines with category labels like "Section:", "Topic:", "Theme:", "Recommendation:"
- Do not include XML tags in your output
- Skip front matter, acknowledgments, boilerplate, and routine fragments
- If nothing substantive: EMPTY"""

INCREMENTAL_ANALYSIS_PROMPT = """You are continuing the analysis of an evolving conversation.
Prior context (already analyzed) is shown outside <analyze> tags.
New entries are inside <analyze> tags.

Identify significant NEW developments in the marked section:
- New themes, decisions, or turning points not already established in the context
- Shifts in direction or approach from what came before
- Important commitments, discoveries, or conclusions

Rules:
- One observation per line, no numbering, no bullets, no preamble
- Synthesize in your own words — never copy or quote the original text
- Be specific: name the actual thing that changed, not abstract categories
- Do not start lines with category labels like "Decision:", "Theme:", "Turning point:"
- Do not include XML tags in your output
- Skip entries that continue an established theme without adding new significance
- If nothing genuinely new: EMPTY"""


# ---------------------------------------------------------------------------
# Model → effective context budget mapping
# ---------------------------------------------------------------------------

# Effective analysis budget per model.  This is NOT the raw context window —
# it's how much content the model can usefully analyze in a single window
# while still producing quality output (synthesis, not paraphrase).
# Small models degrade fast; cloud models can handle much more.
MODEL_BUDGETS: dict[str, int] = {
    # Ollama / local (free)
    "llama3.2:1b":       1500,
    "llama3.2:3b":       3000,
    "llama3.2":          3000,
    "llama3.1:8b":       6000,
    "llama3.1":          6000,
    "qwen2.5:3b":        3000,
    "qwen2.5:7b":        6000,
    "qwen2.5:14b":       8000,
    "mistral:7b":        6000,
    "gemma2:9b":         6000,
    "gemma3:1b":         1500,
    "gemma3:4b":         3000,
    "gemma3:12b":        6000,
    "gemma3:27b":        8000,
    # OpenAI
    "gpt-5-nano":        4000,
    "gpt-4o-mini":       6000,
    "gpt-5-mini":       12000,
    "gpt-4.1-mini":     12000,
    "gpt-4.1":          16000,
    "gpt-4o":           16000,
    "gpt-5":            16000,
    # Anthropic
    "claude-3-haiku-20240307":    8000,
    "claude-3-5-haiku-20241022": 12000,
    "claude-haiku-4-5-20251001": 12000,
    "claude-sonnet-4-6":         16000,
    "claude-opus-4-6":           16000,
    # Google Gemini
    "gemini-2.0-flash":          8000,
    "gemini-2.5-flash-lite":     8000,
    "gemini-2.5-flash":         12000,
    "gemini-2.5-pro":           16000,
}

# Fallback budgets by provider name (when model isn't in the table)
_PROVIDER_BUDGET_FALLBACK: dict[str, int] = {
    "ollama": 3000,
    "mlx": 3000,
    "anthropic": 12000,
    "openai": 12000,
    "gemini": 10000,
    "truncate": 6000,
}

# Default when nothing matches
_DEFAULT_BUDGET = 6000


def get_budget_for_model(model: str, provider: str = "") -> int:
    """Look up effective analysis budget for a model, with fallback."""
    if model in MODEL_BUDGETS:
        return MODEL_BUDGETS[model]
    # Prefix match (e.g. "llama3.2:3b-q4_0" → "llama3.2:3b")
    for known, budget in MODEL_BUDGETS.items():
        if model.startswith(known):
            return budget
    budget = _PROVIDER_BUDGET_FALLBACK.get(provider, _DEFAULT_BUDGET)
    if model:
        logger.info(
            "Unknown model %r for analyzer budget; using %d (provider=%s). "
            "Override with [analyzer] context_budget in keep.toml.",
            model, budget, provider or "unknown",
        )
    return budget


# ---------------------------------------------------------------------------
# SlidingWindowAnalyzer — default
# ---------------------------------------------------------------------------

class SlidingWindowAnalyzer:
    """Token-budgeted sliding-window decomposition.

    Processes chunks in windows sized to the model's context budget.
    Each window uses XML-style tags to mark which chunks are analysis
    targets vs. context-only. Deduplicates by content hash across windows.

    Works well with small local models (Ollama, MLX) that have limited
    context windows.
    """

    def __init__(
        self,
        provider=None,
        context_budget: int = 12000,
        target_ratio: float = 0.6,
        prompt: str | None = None,
    ):
        """Initialize.

        Args:
        provider: A SummarizationProvider with generate() support.
        context_budget: Total token budget per window.
        target_ratio: Fraction of budget allocated to target chunks (vs context).
        prompt: Fixed prompt text to use. If None, uses DEFAULT_ANALYSIS_PROMPT
            (override with prompt_override in analyze()).
        """
        self._provider = provider
        self._context_budget = context_budget
        self._target_ratio = target_ratio
        self._fixed_prompt = prompt

    def _resolve_prompt(self, prompt_override: str | None = None) -> str:
        """Return the system prompt text for this analysis call."""
        if prompt_override:
            return prompt_override
        if self._fixed_prompt:
            return self._fixed_prompt
        return DEFAULT_ANALYSIS_PROMPT

    def analyze(
        self,
        chunks: Iterable[AnalysisChunk],
        guide_context: str = "",
        prompt_override: str | None = None,
    ) -> list[dict]:
        """Decompose content chunks into parts using sliding windows."""
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        system_prompt = self._resolve_prompt(prompt_override)

        with tracer.start_as_current_span(
            "analyze.token_estimate",
            attributes={"chunk_count": len(chunk_list)},
        ):
            total_tokens = sum(_estimate_tokens(c.content) for c in chunk_list)

        # Fits in one window — single-pass
        if total_tokens <= self._context_budget:
            return self._single_pass(chunk_list, guide_context, system_prompt)

        target_budget = int(self._context_budget * self._target_ratio)
        context_budget_per_side = (self._context_budget - target_budget) // 2
        all_parts = []
        seen_hashes: set[str] = set()
        pos = 0

        while pos < len(chunk_list):
            # Collect target chunks up to target_budget
            target_end = pos
            target_tokens = 0
            while target_end < len(chunk_list):
                chunk_tokens = _estimate_tokens(chunk_list[target_end].content)
                if target_tokens + chunk_tokens > target_budget and target_end > pos:
                    break  # budget exceeded (but always include at least one)
                target_tokens += chunk_tokens
                target_end += 1

            # Add context before (scan backwards from pos)
            ctx_before = pos
            ctx_before_tokens = 0
            while ctx_before > 0:
                chunk_tokens = _estimate_tokens(chunk_list[ctx_before - 1].content)
                if ctx_before_tokens + chunk_tokens > context_budget_per_side:
                    break
                ctx_before -= 1
                ctx_before_tokens += chunk_tokens

            # Add context after (scan forwards from target_end)
            ctx_after = target_end
            ctx_after_tokens = 0
            while ctx_after < len(chunk_list):
                chunk_tokens = _estimate_tokens(chunk_list[ctx_after].content)
                if ctx_after_tokens + chunk_tokens > context_budget_per_side:
                    break
                ctx_after += 1
                ctx_after_tokens += chunk_tokens

            # Build window and call LLM
            window = chunk_list[ctx_before:ctx_after]
            target_start_in_window = pos - ctx_before
            target_end_in_window = target_end - ctx_before

            with tracer.start_as_current_span(
                "analyze.window",
                attributes={
                    "window_chunks": len(window),
                    "target_chunks": target_end - pos,
                    "target_tokens": target_tokens,
                    "context_before_tokens": ctx_before_tokens,
                    "context_after_tokens": ctx_after_tokens,
                },
            ):
                raw = self._analyze_window(
                    window, target_start_in_window, target_end_in_window,
                    guide_context, system_prompt,
                )

            # Dedup across windows by summary hash
            with tracer.start_as_current_span(
                "analyze.window.dedup",
                attributes={"candidate_parts": len(raw)},
            ):
                for part in raw:
                    h = hashlib.md5(part["summary"].encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_parts.append(part)

            pos = target_end

        return all_parts

    def _single_pass(self, chunks: list[AnalysisChunk], guide_context: str,
                     system_prompt: str) -> list[dict]:
        """Content fits in one window — send all as targets."""
        provider = self._provider
        if provider is None:
            return []

        if hasattr(provider, '_provider') and provider._provider is not None:
            provider = provider._provider

        with tracer.start_as_current_span(
            "analyze.single_pass.prompt",
            attributes={"chunk_count": len(chunks)},
        ):
            content = "\n\n---\n\n".join(c.content for c in chunks)
            truncated = content[:80000] if len(content) > 80000 else content

            user_prompt = f"<content>\n<analyze>\n{truncated}\n</analyze>\n</content>"
            if guide_context:
                user_prompt = f"{guide_context}\n\n---\n\n{user_prompt}"

        try:
            with tracer.start_as_current_span(
                "analyze.single_pass.provider",
                attributes={"prompt_chars": len(user_prompt)},
            ) as _gen_span:
                apply_context_attrs(_gen_span)
                _t0 = time.monotonic()
                result = provider.generate(system_prompt, user_prompt, max_tokens=4096)
                _gen_ms = (time.monotonic() - _t0) * 1000.0
            if (c := current_counters()) is not None:
                c.add_analyze(ms=_gen_ms)
            if result:
                with tracer.start_as_current_span(
                    "analyze.single_pass.parse",
                    attributes={"result_chars": len(result)},
                ):
                    return _parse_parts(result)
            logger.warning("Provider returned no result for analysis")
        except Exception as e:
            logger.warning("Analysis failed: %s", e)
        return []

    def _analyze_window(
        self,
        window: list[AnalysisChunk],
        target_start: int,
        target_end: int,
        guide_context: str,
        system_prompt: str,
    ) -> list[dict]:
        """Analyze a single window with XML-tagged target marking."""
        provider = self._provider
        if provider is None:
            return []

        if hasattr(provider, '_provider') and provider._provider is not None:
            provider = provider._provider

        with tracer.start_as_current_span(
            "analyze.window.prompt",
            attributes={"window_chunks": len(window)},
        ):
            prompt = self._build_window_prompt(window, target_start, target_end)

            if guide_context:
                prompt = f"{guide_context}\n\n---\n\n{prompt}"

        try:
            with tracer.start_as_current_span(
                "analyze.window.provider",
                attributes={"prompt_chars": len(prompt)},
            ) as _gen_span:
                apply_context_attrs(_gen_span)
                _t0 = time.monotonic()
                result = provider.generate(system_prompt, prompt, max_tokens=4096)
                _gen_ms = (time.monotonic() - _t0) * 1000.0
            if (c := current_counters()) is not None:
                c.add_analyze(ms=_gen_ms)
            if result:
                with tracer.start_as_current_span(
                    "analyze.window.parse",
                    attributes={"result_chars": len(result)},
                ):
                    return _parse_parts(result)
            logger.warning("Sliding window: provider returned no result")
            return []
        except Exception as e:
            logger.warning("Sliding window LLM call failed: %s", e)
            return []

    @staticmethod
    def _build_window_prompt(
        window: list[AnalysisChunk],
        target_start: int,
        target_end: int,
    ) -> str:
        """Build XML-tagged prompt with <content> and <analyze> markers."""
        parts = ["<content>"]

        for i, chunk in enumerate(window):
            if i == target_start:
                parts.append("<analyze>")
            parts.append(chunk.content)
            if i == target_end - 1:
                parts.append("</analyze>")

        parts.append("</content>")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SinglePassAnalyzer — legacy, available as "single-pass" in registry
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_PROMPT = """You are a document analysis assistant. Your task is to decompose a document into its meaningful structural sections.

For each section, provide:
- "summary": A concise summary of the section (1-3 sentences)
- "tags": A dict of relevant tags for this section (optional)

Return a JSON array of section objects. Example:
```json
[
  {"summary": "Introduction and overview of the topic", "tags": {"topic": "overview"}},
  {"summary": "Detailed analysis of the main argument", "tags": {"topic": "analysis"}}
]
```

Guidelines:
- Identify natural section boundaries (headings, topic shifts, structural breaks)
- Each section should be a coherent unit of meaning
- Keep summaries concise but descriptive
- Tags should capture the essence of each section's subject matter
- Return valid JSON only, no commentary outside the JSON array"""


def _parse_decomposition_json(text: str) -> list[dict]:
    """Parse JSON from LLM decomposition output.

    Handles code fences, wrapper objects, and direct JSON arrays.
    """
    if not text:
        return []

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse decomposition JSON")
        return []

    if isinstance(data, dict):
        for key in ("sections", "parts", "chunks", "result", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            return []

    if not isinstance(data, list):
        return []

    result = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if not entry.get("summary"):
            continue
        section = {
            "summary": str(entry.get("summary", "")),
        }
        if entry.get("tags") and isinstance(entry["tags"], dict):
            section["tags"] = {str(k): str(v) for k, v in entry["tags"].items()}
        result.append(section)

    return result


class SinglePassAnalyzer:
    """Single-pass LLM decomposition with JSON output.

    Concatenates all chunks, sends to the configured summarization provider's
    generate() method, and parses the resulting JSON into part dicts.
    Better for large-context models that can handle the full document at once.
    """

    def __init__(self, provider=None):
        self._provider = provider

    def analyze(
        self,
        chunks: Iterable[AnalysisChunk],
        guide_context: str = "",
    ) -> list[dict]:
        """Decompose content chunks into parts."""
        chunk_list = list(chunks)
        content = "\n\n---\n\n".join(c.content for c in chunk_list)
        return self._call_llm(content, guide_context)

    def _call_llm(self, content: str, guide_context: str = "") -> list[dict]:
        """Call the LLM to decompose content into sections."""
        provider = self._provider
        if provider is None:
            return []

        if hasattr(provider, '_provider') and provider._provider is not None:
            provider = provider._provider

        truncated = content[:80000] if len(content) > 80000 else content

        user_prompt = truncated
        if guide_context:
            user_prompt = (
                f"Decompose this document into meaningful sections.\n\n"
                f"Use these tag definitions to guide your tagging:\n\n"
                f"{guide_context}\n\n"
                f"---\n\n"
                f"Document to analyze:\n\n{truncated}"
            )

        try:
            with tracer.start_as_current_span(
                "analyze.decomposition.provider",
                attributes={"prompt_chars": len(user_prompt)},
            ) as _gen_span:
                apply_context_attrs(_gen_span)
                _t0 = time.monotonic()
                result = provider.generate(
                    DECOMPOSITION_SYSTEM_PROMPT,
                    user_prompt,
                    max_tokens=4096,
                )
                _gen_ms = (time.monotonic() - _t0) * 1000.0
            if (c := current_counters()) is not None:
                c.add_analyze(ms=_gen_ms)
            if result:
                return _parse_decomposition_json(result)
            logger.warning(
                "Provider %s returned no result for decomposition",
                type(provider).__name__,
            )
            return []
        except Exception as e:
            logger.warning("LLM decomposition failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# TagClassifier — data-driven post-analysis classification
# ---------------------------------------------------------------------------

# Regex to parse classifier output lines like:
#   3: act=commitment(0.9) status=open(0.85)
# Tolerant: handles "NUMBER:" prefix echo from small models
_CLASSIFY_LINE_RE = re.compile(
    r"^(?:NUMBER:\s*)?(\d+):\s*(.*)$"
)
# Tolerant: handles spaces around parens (e.g. "act=assertion (0.9)")
_TAG_ASSIGNMENT_RE = re.compile(
    r"(\w+)=(\w+)\s*\(\s*([0-9.]+)\s*\)"
)
# Extract ## Prompt section from document content
_PROMPT_SECTION_RE = re.compile(
    r"^## Prompt\s*\n(.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def extract_prompt_section(content: str) -> str:
    """Extract ## Prompt section from document content, if present."""
    m = _PROMPT_SECTION_RE.search(content)
    return m.group(1).strip() if m else ""


_DEFAULT_TAG_PROMPT = """\
Classify each numbered text fragment.

{taxonomy}

Output one line per fragment. Format — one or more tag=value(confidence) pairs:
NUMBER: tag1=value1(CONFIDENCE) tag2=value2(CONFIDENCE)

CONFIDENCE is 0.0 to 1.0. If no tags apply, write:
NUMBER: NONE

Examples:
{examples}

Rules:
- ONLY use these values — {valid_values}
- Do NOT invent new values
- If a fragment is just a preamble, heading, or meta-commentary with no substantive content, output NONE
- 0.9+ = unambiguous, 0.7-0.9 = likely"""


class TagClassifier:
    """Data-driven tag classification for analyzed parts.

    Reads constrained tag specifications from the store and builds a
    classification prompt dynamically. Each part summary is classified
    against the taxonomy; tags are applied only above a confidence threshold.

    This is a post-processing step — run after SlidingWindowAnalyzer
    produces parts, before tags are written to the store.
    """

    def __init__(
        self,
        provider=None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize.

        Args:
        provider: A SummarizationProvider with generate() support.
        confidence_threshold: Minimum confidence (0-1) to apply a tag.
        """
        self._provider = provider
        self._confidence_threshold = confidence_threshold
        self._tag_specs: list[dict] | None = None

    @staticmethod
    def _extract_prompt_section(content: str) -> str:
        """Extract ## Prompt section from document content, if present."""
        return extract_prompt_section(content)

    def load_specs(self, keeper) -> list[dict]:
        """Load constrained tag specifications from the store.

        Scans for .tag/* documents with _constrained=true, then loads
        their sub-documents to build the classification taxonomy.

        If a document contains a ``## Prompt`` section, that section is
        extracted and used as the classifier-facing description (the
        ``prompt`` field). The human-readable description (``description``)
        is always taken from the document summary.

        Args:
            keeper: A Keeper instance to read tag specs from.

        Returns:
            List of tag spec dicts, each with:
              - key: tag name (e.g. "act")
              - description: from the parent doc summary
              - prompt: from the parent doc's ## Prompt section (or "")
              - values: list of {value, description, prompt} dicts
        """
        specs = []
        # Find all .tag/* parent docs
        doc_coll = keeper._resolve_doc_collection()
        tag_docs = keeper._document_store.query_by_id_prefix(doc_coll, ".tag/")

        # Group: parent docs vs value docs
        parents = {}  # key -> record
        children = {}  # key -> [record, ...]
        for rec in tag_docs:
            doc_id = rec.id if hasattr(rec, 'id') else rec.get("id", "")
            # .tag/act -> parent; .tag/act/commitment -> child
            parts = doc_id.split("/")
            if len(parts) == 2:
                # Parent: .tag/KEY
                key = parts[1]
                parents[key] = rec
            elif len(parts) == 3:
                # Child: .tag/KEY/VALUE
                key = parts[1]
                children.setdefault(key, []).append(rec)

        for key, parent_rec in parents.items():
            tags = parent_rec.tags if hasattr(parent_rec, 'tags') else parent_rec.get("tags", {})
            if tags.get("_constrained") != "true":
                continue

            parent_summary = parent_rec.summary if hasattr(parent_rec, 'summary') else parent_rec.get("summary", "")
            parent_prompt = self._extract_prompt_section(parent_summary)

            # Build value list from children
            values = []
            for child_rec in children.get(key, []):
                child_id = child_rec.id if hasattr(child_rec, 'id') else child_rec.get("id", "")
                value_name = child_id.split("/")[-1]
                summary = child_rec.summary if hasattr(child_rec, 'summary') else child_rec.get("summary", "")
                child_prompt = self._extract_prompt_section(summary)
                values.append({
                    "value": value_name,
                    "description": summary,
                    "prompt": child_prompt,
                })

            specs.append({
                "key": key,
                "description": parent_summary,
                "prompt": parent_prompt,
                "values": sorted(values, key=lambda v: v["value"]),
            })

        self._tag_specs = specs
        return specs

    def build_prompt(
        self,
        specs: list[dict] | None = None,
        *,
        template: str | None = None,
    ) -> str:
        """Build a classification system prompt from tag specs.

        The taxonomy section is generated from store data — no hardcoded
        tag names or values.  If a spec or value has a ``prompt`` field
        (from a ``## Prompt`` section in the tag doc), it is used as the
        classifier-facing description.

        Args:
            specs: Tag specs (uses loaded specs if not provided).
            template: Optional prompt template from ``.prompt/tag/*``.
                Supports ``{taxonomy}``, ``{examples}``, ``{valid_values}``
                placeholders. If None, uses a built-in default.
        """
        specs = specs or self._tag_specs
        if not specs:
            return ""

        taxonomy = self._build_taxonomy(specs)
        valid_values = self._build_valid_values(specs)
        examples = self._build_examples(specs)

        if template:
            return template.format(
                taxonomy=taxonomy,
                examples=examples,
                valid_values=valid_values,
            )

        return _DEFAULT_TAG_PROMPT.format(
            taxonomy=taxonomy,
            examples=examples,
            valid_values=valid_values,
        )

    @staticmethod
    def _build_taxonomy(specs: list[dict]) -> str:
        sections = []
        for spec in specs:
            lines = [f"## Tag: `{spec['key']}`"]
            desc = spec.get("prompt") or spec.get("description", "")
            if desc:
                lines.append(desc)
            lines.append("")
            lines.append("Values (pick at most one):")
            for v in spec["values"]:
                vdesc = v.get("prompt") or v.get("description", "")
                if vdesc:
                    lines.append(f"- `{v['value']}` — {vdesc}")
                else:
                    lines.append(f"- `{v['value']}`")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    @staticmethod
    def _build_valid_values(specs: list[dict]) -> str:
        valid_per_key = {}
        for spec in specs:
            valid_per_key[spec["key"]] = [v["value"] for v in spec["values"]]
        return "; ".join(
            f"{k}: {', '.join(vs)}" for k, vs in valid_per_key.items()
        )

    @staticmethod
    def _build_examples(specs: list[dict]) -> str:
        keys = [s["key"] for s in specs]
        examples = []
        if "act" in keys:
            examples.extend([
                "1: act=assertion(0.9)",
                "2: act=commitment(0.8) status=open(0.9)",
                "3: NONE",
                "4: act=request(0.7) status=open(0.8)",
                "5: act=commitment(0.9) status=fulfilled(0.8)",
                "6: act=assessment(0.8)",
            ])
        else:
            k0 = keys[0] if keys else "tag"
            v0 = specs[0]["values"][0]["value"] if specs and specs[0]["values"] else "value"
            examples.extend([
                f"1: {k0}={v0}(0.9)",
                "2: NONE",
            ])
        return "\n".join(examples)

    def classify(
        self,
        parts: list[dict],
        specs: list[dict] | None = None,
        *,
        prompt_template: str | None = None,
    ) -> list[dict]:
        """Classify parts and add tags above the confidence threshold.

        Args:
            parts: List of part dicts (must have "summary" key).
            specs: Tag specs (uses loaded specs if not provided).
            prompt_template: Optional prompt template from ``.prompt/tag/*``.

        Returns:
            The same parts list, with "tags" dicts added/updated in place.
        """
        specs = specs or self._tag_specs
        if not parts or not specs:
            return parts

        provider = self._provider
        if provider is None:
            return parts

        if hasattr(provider, '_provider') and provider._provider is not None:
            provider = provider._provider

        with tracer.start_as_current_span(
            "analyze.classify.prompt",
            attributes={"part_count": len(parts), "spec_count": len(specs)},
        ):
            system_prompt = self.build_prompt(specs, template=prompt_template)
        if not system_prompt:
            return parts

        # Build user message: numbered fragments
        with tracer.start_as_current_span(
            "analyze.classify.fragments",
            attributes={"part_count": len(parts)},
        ):
            fragment_lines = []
            for i, part in enumerate(parts, 1):
                summary = part.get("summary", "").strip()
                if summary:
                    fragment_lines.append(f"{i}: {summary}")

        if not fragment_lines:
            return parts

        user_prompt = "Classify these fragments:\n\n" + "\n".join(fragment_lines)

        try:
            with tracer.start_as_current_span(
                "analyze.classify.provider",
                attributes={"prompt_chars": len(user_prompt)},
            ) as _gen_span:
                apply_context_attrs(_gen_span)
                _t0 = time.monotonic()
                result = provider.generate(system_prompt, user_prompt, max_tokens=2048)
                _gen_ms = (time.monotonic() - _t0) * 1000.0
            if (c := current_counters()) is not None:
                c.add_analyze(ms=_gen_ms)
            if result:
                with tracer.start_as_current_span(
                    "analyze.classify.apply",
                    attributes={"result_chars": len(result)},
                ):
                    self._apply_classifications(parts, result, specs)
        except Exception as e:
            logger.warning("Tag classification failed: %s", e)

        return parts

    def _apply_classifications(
        self,
        parts: list[dict],
        llm_output: str,
        specs: list[dict] | None = None,
    ) -> None:
        """Parse LLM output and apply tags to parts above threshold.

        Validates assigned values against the taxonomy specs to reject
        invented values from small models.
        """
        # Build valid value sets for validation
        valid_values: dict[str, set[str]] = {}
        if specs:
            for spec in specs:
                valid_values[spec["key"]] = {v["value"] for v in spec["values"]}

        for line in llm_output.strip().splitlines():
            line = line.strip()
            m = _CLASSIFY_LINE_RE.match(line)
            if not m:
                continue

            part_num = int(m.group(1))
            assignments_str = m.group(2).strip()

            if assignments_str == "NONE" or not assignments_str:
                continue

            # 1-indexed → 0-indexed
            if part_num < 1 or part_num > len(parts):
                continue
            part = parts[part_num - 1]

            # Parse tag=value(confidence) assignments
            tags = part.get("tags", {})
            for tm in _TAG_ASSIGNMENT_RE.finditer(assignments_str):
                key = tm.group(1)
                value = tm.group(2)

                # Validate against taxonomy
                if key in valid_values and value not in valid_values[key]:
                    logger.info("Rejected invented value: %s=%s", key, value)
                    continue

                try:
                    confidence = float(tm.group(3))
                except ValueError:
                    continue

                if confidence >= self._confidence_threshold:
                    tags[key] = value

            if tags:
                part["tags"] = tags


# ---------------------------------------------------------------------------
# Register with the provider registry
# ---------------------------------------------------------------------------

get_registry().register_analyzer("default", SlidingWindowAnalyzer)
get_registry().register_analyzer("sliding-window", SlidingWindowAnalyzer)
get_registry().register_analyzer("single-pass", SinglePassAnalyzer)
