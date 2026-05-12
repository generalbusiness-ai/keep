"""State-doc loader, compiler, and evaluator.

Loads `.state/*` documents from the keep store, compiles CEL predicates,
and evaluates rules against a context to produce action dispatches and
state transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class AsyncActionEncountered(Exception):
    """Raised by run_action when a foreground flow encounters an async action.

    The flow runtime catches this to produce a cursor and delegate the
    remaining work to the background queue.
    """

    def __init__(self, action_name: str, action_params: dict[str, Any]) -> None:
        super().__init__(f"async action: {action_name}")
        self.action_name = action_name
        self.action_params = action_params


class SubflowExecutionError(Exception):
    """Raised when a child state-doc flow fails and should fail the parent."""


# ---------------------------------------------------------------------------
# Compiled rule and state-doc types
# ---------------------------------------------------------------------------


@dataclass
class CompiledRule:
    """A single rule from a state doc with its predicate pre-compiled."""

    id: Optional[str] = None
    when: Optional[Any] = None  # compiled CEL program, or None (unconditional)
    when_source: str = ""  # original predicate string for debugging
    do: Optional[str] = None  # action name
    with_params: Optional[dict[str, Any]] = None  # action params (may contain templates)
    then: Optional[str | dict[str, Any]] = None  # state transition
    return_status: Optional[str] = None  # terminal: "done", "error", "stopped"
    return_with: Optional[dict[str, Any]] = None  # data attached to return


@dataclass
class StateDoc:
    """A parsed and compiled state document."""

    name: str
    match: str  # "sequence" or "all"
    rules: list[CompiledRule] = field(default_factory=list)
    post: list[CompiledRule] = field(default_factory=list)  # post-block for match:all


@dataclass
class RuleResult:
    """Result of evaluating a single rule."""

    rule: CompiledRule
    fired: bool
    output: Optional[dict[str, Any]] = None
    transition: Optional[str | dict[str, Any]] = None
    terminal: Optional[str] = None  # "done", "error", "stopped"
    terminal_data: Optional[dict[str, Any]] = None


@dataclass
class EvalResult:
    """Result of evaluating a complete state doc."""

    actions: list[dict[str, Any]]  # [{action, params, rule_id}, ...]
    bindings: dict[str, dict[str, Any]]  # rule_id -> output dict
    transition: Optional[str | dict[str, Any]] = None
    terminal: Optional[str] = None
    terminal_data: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# CEL compilation
# ---------------------------------------------------------------------------

import cel as _cel


def _compile_predicate(source: str) -> Any:
    """Compile a CEL predicate string into a reusable program."""
    return _cel.compile(source)


def _eval_predicate(program: Any, context: dict[str, Any], source: str = "") -> bool:
    """Evaluate a compiled CEL predicate against a context dict."""
    try:
        result = program.execute(context)
    except Exception as exc:
        # CEL raises for absent map keys, but `when:` predicates are filters:
        # a note missing the referenced tag simply does not match the rule.
        # NOTE: substring match tracks the cel library's error wording —
        # revisit if the upstream message format changes.
        if "No such key:" in str(exc):
            logger.debug("Predicate missing key in %r: %s", source or "<unknown>", exc)
            return False
        logger.warning("Predicate eval error in %r: %s", source or "<unknown>", exc)
        return False
    if isinstance(result, bool):
        return result
    return bool(result)


# ---------------------------------------------------------------------------
# Template interpolation for `with:` params
# ---------------------------------------------------------------------------

def _resolve_template(value: Any, context: dict[str, Any]) -> Any:
    """Resolve template references in `with:` parameter values.

    Handles three cases:
    - Pure template string: "{params.query}" -> resolved value
    - Mixed template string: "prefix {x.y} suffix" -> interpolated string
    - Non-string values: returned as-is
    - Nested dicts/lists: resolved recursively
    """
    if isinstance(value, str) and "{" in value and "}" in value:
        # Pure template: entire string is one reference
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}") and stripped.count("{") == 1:
            path = stripped[1:-1].strip()
            resolved = _resolve_path(path, context)
            if resolved is None:
                logger.debug("Template reference %r resolved to None", path)
            return resolved
        # Mixed template: interpolate each {ref}
        import re

        def _replace(match: re.Match) -> str:
            resolved = _resolve_path(match.group(1).strip(), context)
            return str(resolved) if resolved is not None else ""

        return re.sub(r"\{([^}]+)\}", _replace, value)
    if isinstance(value, dict):
        return {k: _resolve_template(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_template(v, context) for v in value]
    return value


def _resolve_path(path: str, context: dict[str, Any]) -> Any:
    """Resolve a dotted path like 'params.query' against a context dict."""
    parts = path.split(".")
    current: Any = context
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            current = getattr(current, part, None)
            if current is None:
                return None
    return current


# ---------------------------------------------------------------------------
# State-doc parsing
# ---------------------------------------------------------------------------

def _parse_rule(raw: dict[str, Any]) -> CompiledRule:
    """Parse and compile a single rule from YAML."""
    rule = CompiledRule()

    rule.id = raw.get("id")
    if isinstance(rule.id, str):
        rule.id = rule.id.strip() or None

    when_src = raw.get("when")
    if isinstance(when_src, str) and when_src.strip():
        rule.when_source = when_src.strip()
        try:
            rule.when = _compile_predicate(rule.when_source)
        except (ValueError, RuntimeError) as exc:
            rule_label = f"rule {rule.id!r}" if rule.id else "rule"
            raise ValueError(f"{rule_label}: failed to compile when: {rule.when_source!r}: {exc}") from exc

    rule.do = raw.get("do")
    if isinstance(rule.do, str):
        rule.do = rule.do.strip() or None

    with_params = raw.get("with")
    if isinstance(with_params, dict):
        rule.with_params = dict(with_params)

    then = raw.get("then")
    if isinstance(then, str):
        rule.then = then.strip() or None
    elif isinstance(then, dict):
        rule.then = dict(then)

    ret = raw.get("return")
    if isinstance(ret, str):
        rule.return_status = ret.strip() or None
    elif isinstance(ret, dict):
        # return: {status: stopped, with: {reason: "..."}}
        rule.return_status = str(ret.get("status") or ret.get("return") or "").strip() or None

    if isinstance(raw.get("return"), dict):
        ret_with = raw["return"].get("with")
        if isinstance(ret_with, dict):
            rule.return_with = dict(ret_with)

    return rule


def parse_state_doc(name: str, body: str) -> StateDoc:
    """Parse a state doc from its YAML body text.

    Args:
        name: State doc name (e.g. "after-write").
        body: YAML text body of the state doc.
    """
    parsed = yaml.safe_load(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"state doc {name!r} must be a YAML mapping")

    if "match" not in parsed:
        raise ValueError(
            f"state doc {name!r}: missing 'match' field "
            f"(fragments use parse_fragment instead)"
        )
    match = str(parsed["match"]).strip().lower()
    if match not in ("sequence", "all"):
        raise ValueError(f"state doc {name!r}: match must be 'sequence' or 'all', got {match!r}")

    raw_rules = parsed.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError(f"state doc {name!r}: rules must be a list")

    rules = []
    for i, r in enumerate(raw_rules):
        if isinstance(r, dict):
            rules.append(_parse_rule(r))
        else:
            logger.warning("state doc %r: rules[%d] is not a mapping, skipping", name, i)

    post: list[CompiledRule] = []
    raw_post = parsed.get("post")
    if isinstance(raw_post, list):
        for i, r in enumerate(raw_post):
            if isinstance(r, dict):
                post.append(_parse_rule(r))
            else:
                logger.warning("state doc %r: post[%d] is not a mapping, skipping", name, i)

    return StateDoc(name=name, match=match, rules=rules, post=post)


# ---------------------------------------------------------------------------
# Fragment parsing and composition
# ---------------------------------------------------------------------------

@dataclass
class StateDocFragment:
    """A parsed fragment that contributes rules to a base state doc."""

    name: str
    order: str  # "before", "after", "before:{id}", "after:{id}"
    rules: list[CompiledRule]


def parse_fragment(name: str, body: str) -> StateDocFragment:
    """Parse a state doc fragment from its YAML body.

    Fragments contribute rules to a base state doc. They have the same
    ``rules:`` list as a full state doc but no ``match:`` or ``post:``.

    Args:
        name: Fragment name (e.g. "obsidian-links").
        body: YAML text body.
    """
    parsed = yaml.safe_load(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"fragment {name!r} must be a YAML mapping")

    order = str(parsed.get("order", "after")).strip()

    raw_rules = parsed.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError(f"fragment {name!r}: rules must be a list")

    rules = []
    for i, r in enumerate(raw_rules):
        if isinstance(r, dict):
            rules.append(_parse_rule(r))
        else:
            logger.warning("fragment %r: rules[%d] is not a mapping, skipping", name, i)

    return StateDocFragment(name=name, order=order, rules=rules)


def merge_fragments(base: StateDoc, fragments: list[StateDocFragment]) -> StateDoc:
    """Merge fragment rules into a base state doc.

    Fragments are inserted into the base rule list according to their
    ``order`` field:
    - ``after`` (default): appended after base rules
    - ``before``: prepended before base rules
    - ``after:{rule_id}``: inserted after the rule with that id
    - ``before:{rule_id}``: inserted before the rule with that id

    If a positional target is not found, falls back to ``after`` with
    a warning.
    """
    if not fragments:
        return base

    # Start with a mutable copy of the base rules
    merged = list(base.rules)

    before_rules: list[CompiledRule] = []
    after_rules: list[CompiledRule] = []

    for frag in fragments:
        order = frag.order

        if order == "before":
            before_rules.extend(frag.rules)
        elif order == "after":
            after_rules.extend(frag.rules)
        elif ":" in order:
            position, target_id = order.split(":", 1)
            target_id = target_id.strip()
            position = position.strip()

            # Find the target rule index
            idx = None
            for i, rule in enumerate(merged):
                if rule.id == target_id:
                    idx = i
                    break

            if idx is None:
                logger.warning(
                    "fragment %r: order target %r not found, appending",
                    frag.name, target_id,
                )
                after_rules.extend(frag.rules)
            elif position == "after":
                # Insert after the target
                for j, rule in enumerate(frag.rules):
                    merged.insert(idx + 1 + j, rule)
            elif position == "before":
                # Insert before the target
                for j, rule in enumerate(frag.rules):
                    merged.insert(idx + j, rule)
            else:
                logger.warning(
                    "fragment %r: unknown order position %r, appending",
                    frag.name, position,
                )
                after_rules.extend(frag.rules)
        else:
            logger.warning(
                "fragment %r: unknown order %r, appending",
                frag.name, order,
            )
            after_rules.extend(frag.rules)

    merged = before_rules + merged + after_rules
    return StateDoc(name=base.name, match=base.match, rules=merged, post=list(base.post))


# ---------------------------------------------------------------------------
# Unified loader with fragment composition
# ---------------------------------------------------------------------------

@dataclass
class NoteStub:
    """Minimal note interface for the loader."""
    id: str
    summary: str
    tags: dict[str, Any]


def load_state_doc(
    name: str,
    *,
    get_note: Any,
    list_children: Any,
) -> Optional[StateDoc]:
    """Load a state doc by name with fragment composition.

    This is the single loader used by both the write path (background
    processing) and the read path (flow runtime).

    Args:
        name: State doc name without ``.state/`` prefix (e.g. "after-write").
        get_note: Callable ``(id: str) -> note | None`` returning a note-like
            object with ``.summary`` and ``.tags`` attributes.
        list_children: Callable ``(prefix: str) -> list[note]`` returning
            child notes matching the prefix, each with ``.id``, ``.summary``,
            and ``.tags``.
    """
    base: Optional[StateDoc] = None

    # Load the authoritative store version. Bundled state docs are migrated
    # into the store up front; runtime does not fall back to package copies.
    try:
        note = get_note(f".state/{name}")
        if note is not None:
            body = str(getattr(note, "summary", "") or "").strip()
            if body:
                try:
                    base = parse_state_doc(name, body)
                except (ValueError, RuntimeError) as exc:
                    logger.warning("Failed to compile state doc %r: %s", name, exc)
    except Exception:
        pass

    if base is None:
        return None

    # Discover and merge child fragments from the store.
    fragments: list[StateDocFragment] = []

    try:
        prefix = f".state/{name}/"
        children = list_children(prefix)
        if children:
            for child in sorted(children, key=lambda c: getattr(c, "id", "")):
                tags = getattr(child, "tags", {}) or {}
                if tags.get("active") == "false":
                    logger.debug("Skipping inactive fragment: %s", getattr(child, "id", ""))
                    continue
                body = str(getattr(child, "summary", "") or "").strip()
                if not body:
                    continue
                frag_name = getattr(child, "id", "").removeprefix(prefix)
                try:
                    fragments.append(parse_fragment(frag_name, body))
                except (ValueError, RuntimeError) as exc:
                    logger.warning("Failed to parse fragment %s: %s",
                                   getattr(child, "id", ""), exc)
                    continue
    except Exception as exc:
        logger.debug("Fragment discovery for %s skipped: %s", name, exc)

    if fragments:
        # Sort by name for stable ordering before merging
        fragments.sort(key=lambda f: f.name)
        base = merge_fragments(base, fragments)
        logger.debug("Merged %d fragment(s) into %s", len(fragments), name)

    return base


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_state_doc(
    doc: StateDoc,
    context: dict[str, Any],
    *,
    run_action: Optional[Any] = None,
) -> EvalResult:
    """Evaluate a state doc's rules against a context.

    Args:
        doc: Compiled state document.
        context: Evaluation context with item, params, budget, flow, and
            any prior rule output bindings.
        run_action: Optional callback ``(action_name, params) -> output_dict``
            for synchronous action execution. If None, actions are collected
            but not executed (caller dispatches them).

    Returns:
        EvalResult with actions to dispatch, output bindings, and
        optional transition or terminal status.
    """
    if doc.match == "sequence":
        return _eval_sequence(doc, context, run_action=run_action)
    return _eval_all(doc, context, run_action=run_action)


def _eval_sequence(
    doc: StateDoc,
    context: dict[str, Any],
    *,
    run_action: Optional[Any] = None,
) -> EvalResult:
    """Evaluate rules in sequence, short-circuiting on transition/terminal."""
    bindings: dict[str, dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    eval_ctx = dict(context)

    for rule in doc.rules:
        # Check predicate
        if rule.when is not None:
            if not _eval_predicate(rule.when, eval_ctx, rule.when_source):
                continue

        # Terminal
        if rule.return_status is not None:
            if rule.return_with:
                data = _resolve_template(rule.return_with, eval_ctx)
                data = data if isinstance(data, dict) else None
            else:
                # Auto-passthrough: expose all bindings when return.with is omitted
                data = dict(bindings) if bindings else None
            return EvalResult(
                actions=actions,
                bindings=bindings,
                terminal=rule.return_status,
                terminal_data=data,
            )

        # Action
        if rule.do is not None:
            params = _resolve_template(rule.with_params, eval_ctx) if rule.with_params else {}
            if not isinstance(params, dict):
                params = {}

            action_entry = {"action": rule.do, "params": params, "rule_id": rule.id}
            actions.append(action_entry)

            # Execute synchronously if callback provided (sequence needs outputs)
            if run_action is not None:
                try:
                    output = run_action(rule.do, params)
                    if isinstance(output, dict) and rule.id:
                        bindings[rule.id] = output
                        eval_ctx[rule.id] = output
                except (AsyncActionEncountered, SubflowExecutionError):
                    raise  # flow runtime handles delegation
                except Exception as exc:
                    logger.warning("Action %s failed: %s", rule.do, exc)
                    error_output = {"error": str(exc)}
                    if rule.id:
                        bindings[rule.id] = error_output
                        eval_ctx[rule.id] = error_output

        # Transition
        if rule.then is not None:
            transition = rule.then
            if isinstance(transition, dict):
                with_data = transition.get("with")
                if isinstance(with_data, dict):
                    transition = dict(transition)
                    transition["with"] = _resolve_template(with_data, eval_ctx)
            return EvalResult(
                actions=actions,
                bindings=bindings,
                transition=transition,
            )

    # Fell through all rules with no terminal or transition
    return EvalResult(actions=actions, bindings=bindings, terminal="done")


def _eval_all(
    doc: StateDoc,
    context: dict[str, Any],
    *,
    run_action: Optional[Any] = None,
) -> EvalResult:
    """Evaluate all matching rules (potentially parallel), then post block."""
    bindings: dict[str, dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    eval_ctx = dict(context)

    for rule in doc.rules:
        # Check predicate
        if rule.when is not None:
            if not _eval_predicate(rule.when, eval_ctx, rule.when_source):
                continue

        # Action
        if rule.do is not None:
            params = _resolve_template(rule.with_params, eval_ctx) if rule.with_params else {}
            if not isinstance(params, dict):
                params = {}

            action_entry = {"action": rule.do, "params": params, "rule_id": rule.id}
            actions.append(action_entry)

            # In match:all, actions could run in parallel; for now, sequential
            if run_action is not None:
                try:
                    output = run_action(rule.do, params)
                    if isinstance(output, dict) and rule.id:
                        bindings[rule.id] = output
                        eval_ctx[rule.id] = output
                except (AsyncActionEncountered, SubflowExecutionError):
                    raise  # flow runtime handles delegation
                except Exception as exc:
                    logger.warning("Action %s failed: %s", rule.do, exc)
                    error_output = {"error": str(exc)}
                    if rule.id:
                        bindings[rule.id] = error_output
                        eval_ctx[rule.id] = error_output

    # Post block
    if doc.post:
        for rule in doc.post:
            if rule.when is not None:
                if not _eval_predicate(rule.when, eval_ctx):
                    continue
            if rule.return_status is not None:
                if rule.return_with:
                    data = _resolve_template(rule.return_with, eval_ctx)
                    data = data if isinstance(data, dict) else None
                else:
                    data = dict(bindings) if bindings else None
                return EvalResult(
                    actions=actions,
                    bindings=bindings,
                    terminal=rule.return_status,
                    terminal_data=data,
                )
            if rule.then is not None:
                transition = rule.then
                if isinstance(transition, dict):
                    with_data = transition.get("with")
                    if isinstance(with_data, dict):
                        transition = dict(transition)
                        transition["with"] = _resolve_template(with_data, eval_ctx)
                return EvalResult(
                    actions=actions,
                    bindings=bindings,
                    transition=transition,
                )

    # Default terminal for match:all with no post block
    return EvalResult(actions=actions, bindings=bindings, terminal="done")
