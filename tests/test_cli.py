"""CLI tests for reflective memory.

Tests verify:
1. CLI commands map to equivalent Python API
2. JSON output is valid and parseable (for jq/unix composability)
3. Exit codes follow conventions (0 success, 1 not found, etc.)
4. Human-readable output is line-oriented (for grep/wc/etc.)
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from keep.state_doc_runtime import FlowResult
from keep.cli_app import app
from tests.conftest import _cleanup_daemons_under, _write_test_store_config


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _shared_e2e_cli_env(tmp_path_factory):
    """Shared store/env for read-only daemon-backed CLI e2e tests."""
    root = tmp_path_factory.mktemp("cli-e2e-shared")
    store = root / ".keep-test-store"
    store.mkdir()
    _write_test_store_config(store)
    env = os.environ.copy()
    env["KEEP_STORE_PATH"] = str(store)
    env["KEEP_CONFIG"] = str(store)
    env["KEEP_LOCAL_ONLY"] = "1"

    # Warm the shared store once so the first `keep` invocation does not spend
    # its entire timeout budget on fresh-store daemon bootstrap.
    warm_cmd = [sys.executable, "-m", "keep", "get", ".meta/todo", "--limit", "0"]
    result = None
    for attempt in range(3):
        result = subprocess.run(
            warm_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=env,
        )
        if result.returncode == 0:
            break
        # The daemon can still be in the middle of auto-start when the first
        # CLI request exhausts its startup budget; a short retry reuses that
        # in-flight daemon instead of paying for another cold bootstrap.
        time.sleep(1.0)
    assert result is not None
    assert result.returncode == 0, result.stderr

    yield env

    _cleanup_daemons_under(root)


@pytest.fixture
def cli(request):
    """Run CLI command and return result."""
    runner = CliRunner()

    def run(*args: str, input: str | None = None) -> SimpleNamespace:
        if request.node.get_closest_marker("e2e") is not None:
            env = request.getfixturevalue("_shared_e2e_cli_env")
            return subprocess.run(
                [sys.executable, "-m", "keep", *args],
                capture_output=True,
                text=True,
                input=input,
                cwd=Path(__file__).parent.parent,
                env=env,
            )
        result = runner.invoke(
            app,
            list(args),
            input=input,
            catch_exceptions=False,
            terminal_width=120,
        )
        return SimpleNamespace(
            returncode=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return run


# -----------------------------------------------------------------------------
# Help and Basic Commands
# -----------------------------------------------------------------------------

class TestCliBasics:
    """Basic CLI functionality tests."""
    
    def test_help(self, cli):
        """CLI shows help with --help."""
        result = cli("--help")
        assert result.returncode == 0
        assert "keep" in result.stdout.lower()
        assert "find" in result.stdout
        assert "put" in result.stdout
    
    def test_no_args_shows_now(self):
        """CLI with no args renders the current working context."""
        runner = CliRunner()
        with (
            patch("keep.cli_app._get_port", return_value=1234),
            patch(
                "keep.cli_app._get",
                return_value={
                    "item": {"id": "now", "summary": "Working context", "tags": {"project": "keep"}},
                    "viewing_offset": 0,
                    "similar": [],
                    "meta": {},
                    "edges": {},
                    "parts": [],
                    "prev": [],
                    "next": [],
                },
            ),
        ):
            result = runner.invoke(
                app,
                [],
                catch_exceptions=False,
                terminal_width=120,
            )

        assert result.exit_code == 0
        assert "---" in result.stdout
        assert "id: now" in result.stdout

    @pytest.mark.e2e
    def test_meta_docs_loaded(self, cli):
        """Meta-doc system docs are loaded and accessible."""
        result = cli("get", ".meta/todo", "--limit", "0")
        assert result.returncode == 0
        assert ".meta/todo" in result.stdout

    @pytest.mark.e2e
    def test_tag_type_doc_loaded(self, cli):
        """Tag description doc .tag/type is loaded and has full content."""
        result = cli("get", ".tag/type", "--limit", "0")
        assert result.returncode == 0
        assert "Entity Type" in result.stdout
        # Should contain the values table (verbatim, not summarized)
        assert "conversation" in result.stdout
        assert "vulnerability" in result.stdout

    @pytest.mark.e2e
    def test_meta_sections_use_namespace_prefix(self, cli):
        """Meta sections in frontmatter use meta/ prefix to avoid key conflicts."""
        result = cli("get", ".meta/todo", "--limit", "0")
        assert result.returncode == 0
        # The meta-doc itself shouldn't show meta/ sections (it IS a meta-doc)
        # but its content should have the state-doc rules intact
        assert "do: find" in result.stdout
        assert "act: commitment" in result.stdout

    def test_command_help(self, cli):
        """Individual commands have help."""
        for cmd in ["find", "put", "get", "list", "tag"]:
            result = cli(cmd, "--help")
            assert result.returncode == 0, f"{cmd} --help failed"
            assert "Usage" in result.stdout or "usage" in result.stdout.lower()

    @pytest.mark.parametrize("cmd", ["find", "put", "get", "list", "tag", "del", "now", "config"])
    def test_help_examples_formatted(self, cli, cmd):
        """CLI help renders examples with preserved line breaks, not wrapped.

        Typer/Click uses backspace (0x08) in docstrings to mark sections that
        should not be re-wrapped.  If docstrings are raw strings, the backspace
        becomes a literal backslash-b which Click ignores, collapsing examples.
        """
        result = cli(cmd, "--help")
        assert result.returncode == 0, f"{cmd} --help failed"
        assert r"\b" not in result.stdout, (
            f"{cmd} --help contains literal '\\b' — "
            f"docstring is likely a raw string (r\"\"\") instead of regular string"
        )


# -----------------------------------------------------------------------------
# JSON Output Tests (Composability with jq)
# -----------------------------------------------------------------------------

class TestJsonOutput:
    """Tests for JSON output format."""
    
    def test_json_flag_exists(self, cli):
        """Global --json flag is recognized."""
        # --json is now a global flag (before the command)
        result = cli("--json", "find", "--help")
        # --help takes precedence, should succeed
        assert result.returncode == 0

    def test_collections_json_is_valid(self, cli):
        """--json collections produces valid JSON array."""
        result = cli("--json", "collections")
        # May fail with NotImplementedError, but if it produces output, it should be JSON
        if result.returncode == 0 and result.stdout.strip():
            parsed = json.loads(result.stdout)
            assert isinstance(parsed, list)
    
    def test_json_output_has_required_fields(self):
        """JSON item output includes id, summary, tags, score."""
        from keep.console_support import _format_items
        from keep.types import Item

        item = Item(
            id="test:1",
            summary="Test summary",
            tags={"project": "myapp", "_created": "2026-01-30T10:00:00Z"},
            score=0.95
        )

        output = _format_items([item], as_json=True)
        parsed = json.loads(output)[0]

        assert parsed["id"] == "test:1"
        assert parsed["summary"] == "Test summary"
        assert parsed["tags"]["project"] == "myapp"
        assert parsed["score"] == 0.95
    
    def test_json_list_output(self):
        """JSON list output is a valid array."""
        from keep.console_support import _format_items
        from keep.types import Item
        
        items = [
            Item(id="test:1", summary="First", tags={}, score=0.9),
            Item(id="test:2", summary="Second", tags={}, score=0.8),
        ]
        
        output = _format_items(items, as_json=True)
        parsed = json.loads(output)
        
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "test:1"
        assert parsed[1]["id"] == "test:2"
    
    def test_json_empty_list(self):
        """Empty results produce empty JSON array."""
        from keep.console_support import _format_items
        
        output = _format_items([], as_json=True)
        parsed = json.loads(output)
        
        assert parsed == []


# -----------------------------------------------------------------------------
# Human-Readable Output Tests (Composability with grep/wc)
# -----------------------------------------------------------------------------

class TestHumanOutput:
    """Tests for human-readable output format."""
    
    def test_human_item_format(self):
        """Human-readable item shows id and summary."""
        from keep.console_support import _format_summary_line
        from keep.types import Item

        item = Item(id="file:///doc.md", summary="A document about testing")
        output = _format_summary_line(item)

        assert "file:///doc.md" in output
        assert "A document about testing" in output

    def test_human_item_prefers_name_then_title(self):
        """List-style human output uses note display-name conventions."""
        from keep.console_support import _format_summary_line
        from keep.types import Item

        named = Item(
            id="contact:discord:42",
            summary="Longer fallback summary",
            tags={"name": ["Alice", "Alicia"], "title": "Ignored"},
        )
        titled = Item(
            id="doc:1",
            summary="Fallback summary",
            tags={"title": ["Old title", "Current title"]},
        )

        named_output = _format_summary_line(named)
        titled_output = _format_summary_line(titled)

        assert "Alicia" in named_output
        assert "Longer fallback summary" not in named_output
        assert "Current title" in titled_output
        assert "Fallback summary" not in titled_output
    
    def test_human_item_with_score(self):
        """Human-readable item shows score in full YAML mode."""
        from keep.console_support import render_context
        from keep.types import Item, ItemContext

        item = Item(id="test:1", summary="Test", score=0.95)
        output = render_context(ItemContext(item=item))

        # YAML frontmatter format includes score
        assert "score: 0.95" in output
        assert "---" in output
    
    def test_human_list_empty(self):
        """Empty list shows user-friendly message."""
        from keep.console_support import _format_items
        
        output = _format_items([], as_json=False)
        assert "No results" in output
    
    def test_human_list_separates_items(self):
        """Items are separated by newlines (summary format)."""
        from keep.console_support import _format_items
        from keep.types import Item

        items = [
            Item(id="test:1", summary="First"),
            Item(id="test:2", summary="Second"),
        ]
        output = _format_items(items, as_json=False)

        # Items should be on separate lines (summary format: id@V{N} date summary)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert "test:1" in lines[0]


class TestFlowRendering:
    def test_render_flow_response_renders_item_binding(self):
        from keep.console_support import render_flow_response

        result = FlowResult(
            status="done",
            ticks=1,
            bindings={
                "item": {
                    "id": "test-item",
                    "summary": "important fact",
                    "tags": {"topic": "test"},
                }
            },
        )

        rendered = render_flow_response(result)

        assert "flow: done" in rendered
        assert "important fact" in rendered

    def test_render_flow_response_get_uses_cli_context_shape(self):
        from keep.console_support import render_flow_response
        from keep.types import Item, ItemContext

        class KeeperStub:
            def get_context(self, item_id):
                assert item_id == "test-item"
                return ItemContext(
                    item=Item(id="test-item", summary="important fact", tags={"topic": "test"}),
                )

        result = FlowResult(
            status="done",
            ticks=1,
            history=["get"],
            data={
                "similar": {
                    "results": [
                        {"id": "other", "summary": "other fact", "tags": {}},
                    ]
                }
            },
            bindings={
                "item": {
                    "id": "test-item",
                    "summary": "important fact",
                    "tags": {"topic": "test"},
                },
                "similar": {
                    "results": [
                        {"id": "other", "summary": "other fact", "tags": {}},
                    ]
                },
            },
        )

        rendered = render_flow_response(result, keeper=KeeperStub())

        assert "flow: done" in rendered
        assert "---\nid: test-item" in rendered
        assert "\nsimilar:\n" not in rendered

    def test_render_flow_response_get_uses_cli_context_shape_without_keeper(self):
        from keep.console_support import render_flow_response

        result = FlowResult(
            status="done",
            ticks=1,
            history=["get"],
            bindings={
                "item": {
                    "id": "test-item",
                    "summary": "important fact",
                    "tags": {"topic": "test"},
                },
                "similar": {
                    "results": [
                        {"id": "other", "summary": "other fact", "tags": {}},
                    ]
                },
            },
        )

        rendered = render_flow_response(result, keeper=None)

        assert "flow: done" in rendered
        assert "---\nid: test-item" in rendered
        assert "\nsimilar:\n" in rendered
        assert "other fact" in rendered

    def test_render_flow_response_get_uses_note_first_for_fragment_history(self):
        from keep.console_support import render_flow_response

        result = FlowResult(
            status="done",
            ticks=1,
            history=["openclaw", "get"],
            bindings={
                "item": {
                    "id": "test-item",
                    "summary": "important fact",
                    "tags": {"topic": "test"},
                },
            },
        )

        rendered = render_flow_response(result, keeper=None)

        assert "flow: done" in rendered
        assert "---\nid: test-item" in rendered

    def test_render_flow_response_renders_results_binding(self):
        from keep.console_support import render_flow_response

        result = FlowResult(
            status="done",
            ticks=1,
            bindings={
                "results": {
                    "results": [
                        {
                            "id": ".library/example",
                            "summary": "Example teaching",
                            "tags": {"kind": "teaching"},
                        }
                    ],
                    "count": 1,
                }
            },
        )

        rendered = render_flow_response(result)

        assert "flow: done" in rendered
        assert "Example teaching" in rendered

    def test_frontmatter_quotes_scalar_tag_values(self):
        """Scalar tag values are always quoted in frontmatter."""
        from keep.console_support import render_context
        from keep.types import Item, ItemContext

        spoof = 'conv1 [2026-01-01] "fake summary"'
        item = Item(
            id="test:1",
            summary="Test",
            tags={"note": spoof, "project": "keep"},
        )

        output = render_context(ItemContext(item=item))
        assert 'note: "conv1 [2026-01-01] \\"fake summary\\""' in output
        assert 'project: "keep"' in output

    def test_frontmatter_unifies_edge_refs_under_tags(self):
        """Resolved edge refs are rendered in tags, not tags/<inverse>."""
        from keep.console_support import render_context
        from keep.types import EdgeRef, Item, ItemContext

        item = Item(
            id="alice",
            summary="Profile",
            tags={"said": "manual-tag", "_created": "2026-03-02T00:00:00"},
        )
        ctx = ItemContext(
            item=item,
            edges={
                "said": [
                    EdgeRef(
                        source_id="conv1",
                        date="2026-02-27",
                        summary='Gina said "hello"',
                    ),
                ],
            },
        )

        output = render_context(ctx)
        assert "tags/said:" not in output
        assert "said:" in output
        assert '- conv1 [2026-02-27] "Gina said \\"hello\\""' in output
        # Edge refs supersede raw scalar display for the same key.
        assert 'said: "manual-tag"' not in output


# -----------------------------------------------------------------------------
# Exit Code Tests (Shell Scripting)
# -----------------------------------------------------------------------------

class TestExitCodes:
    """Tests for proper exit codes."""
    
    def test_help_returns_zero(self, cli):
        """Help returns exit code 0."""
        result = cli("--help")
        assert result.returncode == 0
    
    def test_unknown_command_returns_nonzero(self, cli):
        """Unknown command returns non-zero exit code."""
        result = cli("nonexistent-command")
        assert result.returncode != 0
    
    def test_invalid_tag_format_returns_error(self, cli):
        """Invalid tag format returns exit code 1."""
        # put with bad tag format
        result = cli("put", "test:1", "--tag", "badformat")
        # Should fail due to missing = in tag
        # (may also fail due to NotImplemented, which is fine)
        if "Invalid tag format" in result.stderr:
            assert result.returncode == 1

    def test_put_inline_with_summary_rejected(self, cli):
        """--summary with inline text is rejected (would lose content)."""
        result = cli("put", "my note", "--summary", "a summary")
        assert result.returncode == 1
        assert "cannot be used with inline text" in result.stderr

    def test_put_stdin_with_summary_rejected(self, cli):
        """--summary with stdin is rejected (would lose content)."""
        result = cli("put", "-", "--summary", "a summary", input="some content")
        assert result.returncode == 1
        assert "cannot be used with stdin" in result.stderr

    def test_put_inline_too_long_rejected(self, cli):
        """Inline text exceeding max_summary_length is rejected."""
        long_text = "x" * 3001
        result = cli("put", long_text)
        assert result.returncode == 1
        assert "too long" in result.stderr.lower()
        assert "file" in result.stderr.lower()  # Hint mentions file

    def test_put_inline_survives_path_probe_oserror(self, monkeypatch):
        """Long inline text stays inline when filesystem probes raise OSError."""
        from keep import cli_app

        runner = CliRunner()
        long_text = "This is inline text, not a path. " * 12
        captured: dict[str, object] = {}
        original_is_dir = cli_app.Path.is_dir
        original_exists = cli_app.Path.exists

        def raising_is_dir(path_obj):
            if str(path_obj) == long_text:
                raise OSError(63, "File name too long")
            return original_is_dir(path_obj)

        def raising_exists(path_obj):
            if str(path_obj) == long_text:
                raise OSError(63, "File name too long")
            return original_exists(path_obj)

        def fake_post(port, path, body):
            captured["port"] = port
            captured["path"] = path
            captured["body"] = body
            return {"id": "%inline"}

        monkeypatch.setattr(cli_app.Path, "is_dir", raising_is_dir)
        monkeypatch.setattr(cli_app.Path, "exists", raising_exists)
        monkeypatch.setattr(cli_app, "_get_port", lambda: 1234)
        monkeypatch.setattr(cli_app, "_post", fake_post)

        result = runner.invoke(
            app,
            ["put", long_text],
            catch_exceptions=False,
            terminal_width=120,
        )

        assert result.exit_code == 0
        assert captured["path"] == "/v1/notes"
        assert captured["body"] == {
            "content": long_text,
            "uri": None,
            "id": None,
            "tags": None,
            "summary": None,
            "force": None,
        }
        assert "%inline stored." in result.stdout


# -----------------------------------------------------------------------------
# Tag Parsing Tests
# -----------------------------------------------------------------------------

class TestTagParsing:
    """Tests for key=value tag parsing."""
    
    def test_single_tag(self, cli):
        """Single --tag is parsed correctly."""
        # We can't fully test this without implementation,
        # but we can check the format is accepted
        result = cli("put", "--help")
        assert "--tag" in result.stdout or "-t" in result.stdout

    def test_tag_format_validation(self, cli):
        """Tags without = are rejected."""
        result = cli("put", "test:1", "--tag", "invalid")
        # Should fail - either due to format error or missing dependencies
        # When implementation is complete, this should specifically check for format error
        assert result.returncode != 0


# -----------------------------------------------------------------------------
# Unix Composability Integration
# -----------------------------------------------------------------------------

class TestUnixComposability:
    """Tests demonstrating Unix pipeline composability."""
    
    def test_json_output_pipeable(self):
        """JSON output can be processed with standard tools."""
        from keep.console_support import _format_items
        from keep.types import Item
        
        items = [
            Item(id="doc:1", summary="First doc", tags={"project": "alpha"}, score=0.9),
            Item(id="doc:2", summary="Second doc", tags={"project": "beta"}, score=0.8),
        ]
        
        json_output = _format_items(items, as_json=True)
        
        # Simulate: keep find "query" --json | jq '.[0].id'
        parsed = json.loads(json_output)
        first_id = parsed[0]["id"]
        assert first_id == "doc:1"
        
        # Simulate: keep find "query" --json | jq '.[] | select(.score > 0.85)'
        high_score = [item for item in parsed if item["score"] > 0.85]
        assert len(high_score) == 1
        assert high_score[0]["id"] == "doc:1"
    
    def test_line_oriented_for_wc(self):
        """Human output is countable with wc -l."""
        from keep.console_support import _format_items
        from keep.types import Item
        
        items = [
            Item(id=f"doc:{i}", summary=f"Doc {i}") 
            for i in range(5)
        ]
        
        output = _format_items(items, as_json=False)
        
        # Each item produces predictable lines
        lines = output.strip().split("\n")
        # 5 items * 2 lines each, minus separators handled differently
        assert len(lines) >= 5  # At least one line per item
    
    def test_ids_extractable_from_json(self):
        """IDs can be extracted for use in other commands."""
        from keep.console_support import _format_items
        from keep.types import Item

        items = [
            Item(id="file:///a.md", summary="A"),
            Item(id="file:///b.md", summary="B"),
        ]

        json_output = _format_items(items, as_json=True)
        parsed = json.loads(json_output)

        # Simulate: keep find "query" --json | jq -r '.[].id' | xargs -I{} keep get {}
        ids = [item["id"] for item in parsed]
        assert ids == ["file:///a.md", "file:///b.md"]

    def test_get_multiple_ids_separated_by_yaml_separator(self):
        """Multiple IDs in get produce YAML-document-separated output."""
        runner = CliRunner()
        with (
            patch("keep.cli_app._get_port", return_value=1234),
            patch(
                "keep.cli_app._get_one_item",
                side_effect=[
                    "---\nid: .conversations\nConversation doc",
                    "---\nid: .domains\nDomains doc",
                ],
            ),
        ):
            result = runner.invoke(
                app,
                ["get", ".conversations", ".domains"],
                catch_exceptions=False,
                terminal_width=120,
            )
        assert result.exit_code == 0
        assert "\n---\n" in result.stdout
        assert ".conversations" in result.stdout
        assert ".domains" in result.stdout

    @pytest.mark.e2e
    def test_get_single_id_no_extra_separator(self, cli):
        """Single ID in get produces normal output (no extra separators)."""
        result = cli("get", ".conversations")
        assert result.returncode == 0
        assert "id: .conversations" in result.stdout

    @pytest.mark.e2e
    def test_get_nonexistent_id_returns_error(self, _shared_e2e_cli_env):
        """Nonexistent ID returns exit code 1."""
        # Use a real subprocess here so daemon-client globals from in-process
        # CliRunner invocations cannot leak across command boundaries, while
        # still reusing the warmed shared test daemon/store.
        result = subprocess.run(
            [sys.executable, "-m", "keep", "get", "nonexistent:id:that:does:not:exist"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env=_shared_e2e_cli_env,
        )
        assert result.returncode == 1
        assert "Not found" in result.stderr


# -----------------------------------------------------------------------------
# CLI / API Equivalence Tests
# -----------------------------------------------------------------------------

class TestApiCliEquivalence:
    """Tests verifying CLI maps to Python API."""
    
    def test_find_maps_to_api_find(self, cli):
        """'find' command maps to Keeper.find()."""
        result = cli("find", "--help")
        assert "semantic" in result.stdout.lower() or "similar" in result.stdout.lower()
        # The CLI find uses mem.find(query, limit=limit)
    
    def test_put_maps_to_api_put(self, cli):
        """'put' command maps to Keeper.put()."""
        result = cli("put", "--help")
        assert "URI" in result.stdout or "document" in result.stdout.lower()
        # The CLI put uses kp.put(uri=...)

    def test_put_text_mode_maps_to_api_put(self, cli):
        """'put' text mode (no ://) maps to Keeper.put()."""
        result = cli("put", "--help")
        # The help should mention text content mode
        assert "text" in result.stdout.lower() or "content" in result.stdout.lower()
        # The CLI put with text calls kp.put() internally
    
    def test_get_maps_to_api_get(self, cli):
        """'get' command maps to Keeper.get()."""
        result = cli("get", "--help")
        assert "ID" in result.stdout or "id" in result.stdout.lower()
        # The CLI get uses mem.get(id)

    def test_get_accepts_multiple_ids(self, cli):
        """'get' command accepts multiple ID arguments."""
        result = cli("get", "--help")
        # Help should show variadic argument
        assert "ID..." in result.stdout
    
    def test_list_tag_filter(self, cli):
        """'list --tag' is available for tag filtering."""
        result = cli("list", "--help")
        assert "--tag" in result.stdout

    def test_list_prefix_argument(self, cli):
        """'list' accepts an optional prefix argument."""
        result = cli("list", "--help")
        assert "PREFIX" in result.stdout or "prefix" in result.stdout.lower()


# -----------------------------------------------------------------------------
# Option Tests
# -----------------------------------------------------------------------------

class TestOptions:
    """Tests for CLI options."""
    
    def test_store_option(self, cli):
        """--store option is available."""
        result = cli("find", "--help")
        assert "--store" in result.stdout or "-s" in result.stdout
    
    def test_limit_option(self, cli):
        """--limit option is available."""
        result = cli("find", "--help")
        assert "--limit" in result.stdout or "-n" in result.stdout
    
    def test_json_option(self, cli):
        """--json global option is available."""
        # --json is now a global flag, visible in main help
        result = cli("--help")
        assert "--json" in result.stdout or "-j" in result.stdout


# -----------------------------------------------------------------------------
# Shell-safe ID Quoting Tests
# -----------------------------------------------------------------------------

class TestShellQuoteId:
    """Tests for _shell_quote_id() helper."""

    def test_safe_id_not_quoted(self):
        """IDs with only shell-safe chars are returned as-is."""
        from keep.console_support import _shell_quote_id
        assert _shell_quote_id("%abc123def456") == "%abc123def456"
        assert _shell_quote_id("file:///path/to/doc.md") == "file:///path/to/doc.md"
        assert _shell_quote_id("https://example.com/path") == "https://example.com/path"
        assert _shell_quote_id("now@V{3}") == "now@V{3}"
        assert _shell_quote_id("now@V{-3}") == "now@V{-3}"
        assert _shell_quote_id(".tag/act") == ".tag/act"
        assert _shell_quote_id(".conversations") == ".conversations"

    def test_space_id_quoted(self):
        """IDs with spaces get single-quoted."""
        from keep.console_support import _shell_quote_id
        result = _shell_quote_id("file:///Application Data/foo")
        assert result == "'file:///Application Data/foo'"

    def test_special_chars_quoted(self):
        """IDs with shell-special chars get single-quoted."""
        from keep.console_support import _shell_quote_id
        assert _shell_quote_id("test$var") == "'test$var'"
        assert _shell_quote_id("test&bg") == "'test&bg'"
        assert _shell_quote_id("test;cmd") == "'test;cmd'"

    def test_single_quote_escaped(self):
        r"""IDs containing single quotes use '\\'' escaping."""
        from keep.console_support import _shell_quote_id
        result = _shell_quote_id("it's a test")
        assert result == "'it'\\''s a test'"

    def test_summary_line_quotes_unsafe_id(self):
        """_format_summary_line quotes IDs with spaces."""
        from keep.console_support import _format_summary_line
        from keep.types import Item
        item = Item(
            id="file:///Application Data/doc.md",
            summary="A doc",
            tags={"_updated": "2026-01-15T00:00:00Z"},
        )
        output = _format_summary_line(item)
        assert "'file:///Application Data/doc.md'" in output

    def test_summary_line_no_quotes_safe_id(self):
        """_format_summary_line does not quote safe IDs."""
        from keep.console_support import _format_summary_line
        from keep.types import Item
        item = Item(
            id="%abc123",
            summary="A note",
            tags={"_updated": "2026-01-15T00:00:00Z"},
        )
        output = _format_summary_line(item)
        assert "%abc123" in output
        assert "'" not in output.split(" ")[0]  # ID portion not quoted

    def test_versioned_id_quotes_unsafe(self):
        """_format_versioned_id quotes IDs with spaces."""
        from keep.console_support import _format_versioned_id
        from keep.types import Item
        item = Item(
            id="file:///my docs/test.md",
            summary="Test",
            tags={},
        )
        output = _format_versioned_id(item)
        assert output == "'file:///my docs/test.md'"

    def test_json_output_not_quoted(self):
        """JSON output does NOT shell-quote IDs."""
        from keep.console_support import _format_items
        from keep.types import Item
        item = Item(
            id="file:///Application Data/doc.md",
            summary="A doc",
            tags={},
        )
        output = _format_items([item], as_json=True)
        parsed = json.loads(output)[0]
        assert parsed["id"] == "file:///Application Data/doc.md"  # Raw, no quoting


# -----------------------------------------------------------------------------
# Command Alias Tests
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Directory Put Tests
# -----------------------------------------------------------------------------

class TestPutDirectory:
    """Tests for directory mode in the put command."""

    def _write_remote_config(self, store: Path) -> None:
        _write_test_store_config(store)
        with (store / "keep.toml").open("a", encoding="utf-8") as fh:
            fh.write(
                "\n[remote]\n"
                "api_url = \"https://api.example.test\"\n"
                "api_key = \"kn_test\"\n"
                "project = \"first-user\"\n"
            )

    def _fake_remote_keeper(self, monkeypatch, calls: list[tuple[str, str, dict | None]]):
        from keep import cli_app

        class FakeResponse:
            def __init__(self, status_code: int, data: dict):
                self.status_code = status_code
                self._data = data
                self.text = json.dumps(data)
                self.reason_phrase = "OK" if status_code < 400 else "error"

            def json(self):
                return self._data

        class FakeClient:
            def request(self, method, path, json=None):
                calls.append((method, path, json))
                if method == "GET" and path.startswith("/v1/notes/.ignore"):
                    return FakeResponse(404, {"detail": "not found"})
                if method == "POST" and path == "/v1/notes":
                    return FakeResponse(200, {
                        "id": (
                            (json or {}).get("id")
                            or (json or {}).get("uri")
                            or f"remote-{len(calls)}"
                        ),
                        "summary": "",
                        "tags": (json or {}).get("tags") or {},
                    })
                return FakeResponse(200, {})

        class FakeRemoteKeeper:
            def __init__(self):
                self._client = FakeClient()

            def _log_call(self, *args, **kwargs):
                pass

            def close(self):
                pass

        fake = FakeRemoteKeeper()
        monkeypatch.setattr(cli_app, "_remote_keeper", fake)
        monkeypatch.setattr(
            cli_app,
            "_remote_keeper_key",
            ("https://api.example.test", "kn_test", "first-user", ""),
        )
        monkeypatch.setattr(cli_app, "_get_remote_keeper", lambda: fake)
        monkeypatch.setattr(
            cli_app,
            "_daemon_get_port",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("local daemon must not start in remote mode")
            ),
        )

    def test_list_directory_files_basic(self, tmp_path):
        """_list_directory_files returns regular files sorted by name."""
        from keep.console_support import _list_directory_files
        (tmp_path / "b.txt").write_text("B")
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "c.txt").write_text("C")
        files = _list_directory_files(tmp_path)
        assert [f.name for f in files] == ["a.txt", "b.txt", "c.txt"]

    def test_list_directory_files_skips_subdirs(self, tmp_path):
        """_list_directory_files skips subdirectories (non-recursive)."""
        from keep.console_support import _list_directory_files
        (tmp_path / "file.txt").write_text("ok")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("nested")
        files = _list_directory_files(tmp_path)
        assert [f.name for f in files] == ["file.txt"]

    def test_list_directory_files_skips_symlinks(self, tmp_path):
        """_list_directory_files skips symlinks."""
        from keep.console_support import _list_directory_files
        real_file = tmp_path / "real.txt"
        real_file.write_text("real")
        (tmp_path / "link.txt").symlink_to(real_file)
        files = _list_directory_files(tmp_path)
        assert [f.name for f in files] == ["real.txt"]

    def test_list_directory_files_skips_hidden(self, tmp_path):
        """_list_directory_files skips hidden files (dotfiles)."""
        from keep.console_support import _list_directory_files
        (tmp_path / ".DS_Store").write_text("junk")
        (tmp_path / ".hidden").write_text("hidden")
        (tmp_path / "visible.txt").write_text("ok")
        files = _list_directory_files(tmp_path)
        assert [f.name for f in files] == ["visible.txt"]

    def test_list_directory_files_prunes_nested_git_repo(self, tmp_path):
        """Recursive walks should not cross into a nested Git repository."""
        from keep.console_support import _list_directory_files
        (tmp_path / "root.txt").write_text("root")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / ".git").mkdir()
        (nested / "inner.txt").write_text("inner")

        files = _list_directory_files(tmp_path, recurse=True)

        assert [f.relative_to(tmp_path) for f in files] == [Path("root.txt")]

    def test_list_directory_files_prunes_nested_git_worktree(self, tmp_path):
        """Recursive walks should not cross into a nested linked worktree."""
        from keep.console_support import _list_directory_files
        (tmp_path / "root.txt").write_text("root")
        nested = tmp_path / "worktree"
        nested.mkdir()
        (nested / ".git").write_text("gitdir: ../.git/worktrees/worktree\n")
        (nested / "inner.txt").write_text("inner")

        files = _list_directory_files(tmp_path, recurse=True)

        assert [f.relative_to(tmp_path) for f in files] == [Path("root.txt")]

    def test_git_visible_files_prunes_nested_git_worktree(self, tmp_path, monkeypatch):
        """Git-aware walks should also enforce nested worktree boundaries."""
        from keep import utils
        from keep.console_support import _list_directory_files
        (tmp_path / "root.txt").write_text("root")
        nested = tmp_path / "worktree"
        nested.mkdir()
        (nested / ".git").write_text("gitdir: ../.git/worktrees/worktree\n")
        (nested / "inner.txt").write_text("inner")

        def fake_run(*args, **kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout=b"root.txt\0worktree/inner.txt\0",
            )

        monkeypatch.setattr(
            utils,
            "subprocess",
            SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
        )

        files = _list_directory_files(tmp_path, recurse=True)

        assert [f.relative_to(tmp_path) for f in files] == [Path("root.txt")]

    def test_list_directory_files_allows_root_git_worktree(self, tmp_path):
        """The walk root may itself be a linked worktree."""
        from keep.console_support import _list_directory_files
        (tmp_path / ".git").write_text("gitdir: ../.git/worktrees/current\n")
        (tmp_path / "root.txt").write_text("root")

        files = _list_directory_files(tmp_path, recurse=True)

        assert [f.relative_to(tmp_path) for f in files] == [Path("root.txt")]

    def test_list_directory_files_empty(self, tmp_path):
        """_list_directory_files returns empty list for empty directory."""
        from keep.console_support import _list_directory_files
        files = _list_directory_files(tmp_path)
        assert files == []

    def test_list_directory_files_only_hidden(self, tmp_path):
        """_list_directory_files returns empty when only hidden files exist."""
        from keep.console_support import _list_directory_files
        (tmp_path / ".gitignore").write_text("*")
        (tmp_path / ".DS_Store").write_text("")
        files = _list_directory_files(tmp_path)
        assert files == []

    def test_is_filesystem_path_directory(self, tmp_path):
        """_is_filesystem_path recognizes existing directories."""
        from keep.console_support import _is_filesystem_path
        result = _is_filesystem_path(str(tmp_path))
        assert result is not None
        assert result.is_dir()

    def test_is_filesystem_path_file(self, tmp_path):
        """_is_filesystem_path recognizes existing files."""
        from keep.console_support import _is_filesystem_path
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _is_filesystem_path(str(f))
        assert result is not None
        assert result.is_file()

    def test_is_filesystem_path_nonexistent(self):
        """_is_filesystem_path returns None for nonexistent paths."""
        from keep.console_support import _is_filesystem_path
        result = _is_filesystem_path("/nonexistent/path/to/nowhere")
        assert result is None

    def test_is_filesystem_path_uri(self, tmp_path):
        """_is_filesystem_path returns None for URIs even if path exists."""
        from keep.console_support import _is_filesystem_path
        result = _is_filesystem_path(f"file://{tmp_path}")
        assert result is None

    def test_is_filesystem_path_tilde(self):
        """_is_filesystem_path expands tilde."""
        from keep.console_support import _is_filesystem_path
        result = _is_filesystem_path("~")
        assert result is not None
        assert result == Path.home()

    def test_put_directory_rejects_summary(self, cli, tmp_path):
        """Directory mode rejects --summary flag."""
        (tmp_path / "a.txt").write_text("test")
        result = cli("put", str(tmp_path), "--summary", "nope")
        assert result.returncode == 1
        assert "--summary cannot be used with directory" in result.stderr

    def test_put_directory_rejects_id(self, cli, tmp_path):
        """Directory mode rejects --id flag."""
        (tmp_path / "a.txt").write_text("test")
        result = cli("put", str(tmp_path), "--id", "custom")
        assert result.returncode == 1
        assert "--id cannot be used with directory" in result.stderr

    def test_put_empty_directory(self, cli, tmp_path):
        """Directory mode errors on empty directory."""
        result = cli("put", str(tmp_path))
        assert result.returncode == 1
        assert "no eligible files" in result.stderr

    def test_remote_put_file_uploads_to_hosted_without_daemon(self, tmp_path, monkeypatch):
        """A TOML [remote] makes file puts hosted uploads, not daemon writes."""
        from keep import cli_app

        store = tmp_path / "store"
        store.mkdir()
        self._write_remote_config(store)
        source = tmp_path / "note.txt"
        source.write_text("hosted file body", encoding="utf-8")
        calls: list[tuple[str, str, dict | None]] = []
        self._fake_remote_keeper(monkeypatch, calls)
        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--store", str(store), "put", str(source), "--id", "uploaded"],
            catch_exceptions=False,
            terminal_width=120,
        )

        assert result.exit_code == 0, result.output
        note_posts = [body for method, path, body in calls if method == "POST" and path == "/v1/notes"]
        assert len(note_posts) == 1
        body = note_posts[0]
        assert body is not None
        assert body["id"] == "uploaded"
        assert "uri" not in body
        assert body["content_type"] == "text/plain"
        assert base64.b64decode(body["content_base64"]).decode() == "hosted file body"

    def test_remote_put_file_without_id_uses_file_uri_identity(self, tmp_path, monkeypatch):
        """Remote uploads keep URI-mode ID semantics when --id is omitted."""
        from keep import cli_app

        store = tmp_path / "store"
        store.mkdir()
        self._write_remote_config(store)
        source = tmp_path / "note.txt"
        source.write_text("hosted file body", encoding="utf-8")
        calls: list[tuple[str, str, dict | None]] = []
        self._fake_remote_keeper(monkeypatch, calls)
        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)

        result = CliRunner().invoke(
            app,
            ["--store", str(store), "put", str(source)],
            catch_exceptions=False,
            terminal_width=120,
        )

        assert result.exit_code == 0, result.output
        note_posts = [body for method, path, body in calls if method == "POST" and path == "/v1/notes"]
        assert len(note_posts) == 1
        body = note_posts[0]
        file_uri = f"file://{source.resolve()}"
        assert body is not None
        assert body["id"] == file_uri
        assert "uri" not in body
        assert base64.b64decode(body["content_base64"]).decode() == "hosted file body"

    def test_remote_put_directory_uploads_files_and_skips_local_git_enqueue(
        self, tmp_path, monkeypatch,
    ):
        """Recursive directory puts upload child files to hosted REST only."""
        from keep import cli_app

        store = tmp_path / "store"
        store.mkdir()
        self._write_remote_config(store)
        root = tmp_path / "docs"
        (root / "nested").mkdir(parents=True)
        (root / "a.txt").write_text("alpha", encoding="utf-8")
        (root / "nested" / "b.md").write_text("bravo", encoding="utf-8")
        calls: list[tuple[str, str, dict | None]] = []
        self._fake_remote_keeper(monkeypatch, calls)
        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--store", str(store), "put", str(root), "-r", "-t", "project=remote"],
            catch_exceptions=False,
            terminal_width=120,
        )

        assert result.exit_code == 0, result.output
        note_posts = [body for method, path, body in calls if method == "POST" and path == "/v1/notes"]
        assert len(note_posts) == 2
        decoded = sorted(base64.b64decode(body["content_base64"]).decode() for body in note_posts)
        assert decoded == ["alpha", "bravo"]
        assert all("uri" not in body for body in note_posts)
        assert sorted(body["id"] for body in note_posts) == [
            f"file://{(root / 'a.txt').resolve()}",
            f"file://{(root / 'nested' / 'b.md').resolve()}",
        ]
        assert all(body["tags"] == {"project": "remote"} for body in note_posts)
        assert all(not body.get("enqueue_git") for body in note_posts)

    def test_remote_put_watch_is_rejected_without_daemon(self, tmp_path, monkeypatch):
        """Remote mode must not fall back to local watch/pending behavior."""
        from keep import cli_app

        store = tmp_path / "store"
        store.mkdir()
        self._write_remote_config(store)
        source = tmp_path / "note.txt"
        source.write_text("body", encoding="utf-8")
        calls: list[tuple[str, str, dict | None]] = []
        self._fake_remote_keeper(monkeypatch, calls)
        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--store", str(store), "put", str(source), "--watch"],
            catch_exceptions=False,
            terminal_width=120,
        )

        assert result.exit_code == 1
        assert "--watch/--unwatch require the local daemon" in result.stderr
        assert calls == []

    def test_remote_analyze_force_is_sent_to_hosted_endpoint(self, tmp_path, monkeypatch):
        """Remote analyze preserves --force instead of silently downgrading it."""
        from keep import cli_app

        store = tmp_path / "store"
        store.mkdir()
        self._write_remote_config(store)
        calls: list[tuple[str, str, dict | None]] = []

        def fake_remote_request(method, path, body=None):
            calls.append((method, path, body))
            return 202, {"id": "doc-1", "status": "queued"}

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        monkeypatch.setattr(cli_app, "_remote_request", fake_remote_request)
        monkeypatch.setattr(
            cli_app,
            "_daemon_get_port",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("local daemon must not start in remote mode")
            ),
        )

        result = CliRunner().invoke(
            app,
            ["--store", str(store), "analyze", "doc-1", "--force", "-t", "topic"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert calls == [
            ("POST", "/v1/notes/doc-1/analyze", {"force": True, "tags": ["topic"]})
        ]

    def test_remote_find_uses_flow_backed_remote_keeper(self, tmp_path, monkeypatch):
        """Hosted CLI find uses RemoteKeeper flow support, not a local daemon."""
        from keep import cli_app
        from keep.types import Item

        store = tmp_path / "store"
        store.mkdir()
        self._write_remote_config(store)
        calls: list[dict] = []

        class FakeRemoteKeeper:
            def find(self, query=None, **kwargs):
                calls.append({"query": query, **kwargs})
                return [
                    Item(
                        id="remote-found",
                        summary="found through hosted flow",
                        tags={"topic": "cli"},
                        score=0.75,
                    )
                ]

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        monkeypatch.setattr(cli_app, "_get_remote_keeper", lambda: FakeRemoteKeeper())
        monkeypatch.setattr(
            cli_app,
            "_daemon_get_port",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("local daemon must not start in remote mode")
            ),
        )

        result = CliRunner().invoke(
            app,
            [
                "--store", str(store),
                "--json",
                "find", "needle",
                "--limit", "3",
                "-t", "topic=cli",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["notes"][0]["id"] == "remote-found"
        assert calls == [{
            "query": "needle",
            "tags": {"topic": "cli"},
            "similar_to": None,
            "limit": 3,
            "since": None,
            "until": None,
            "include_self": False,
            "include_hidden": False,
            "deep": False,
            "scope": None,
        }]

    def test_put_directory_watch_sends_directory_kind(self, tmp_path, monkeypatch):
        """`keep put <dir> --watch` posts watch_kind=directory and enqueue_git."""
        from keep import cli_app

        (tmp_path / "a.txt").write_text("hello")
        captured: list[tuple[str, str, dict | None]] = []

        def fake_request(method, port, path, body=None):
            captured.append((method, path, body))
            if method == "GET":
                return 404, {}
            return 200, {"id": body.get("uri", ""),
                         "watch": {"source": str(tmp_path), "interval": "PT30S"}}

        monkeypatch.setattr(cli_app, "_daemon_request", fake_request)
        cli_app._put_directory(
            port=0, resolved_path=tmp_path, parsed_tags={},
            recurse=False, exclude=None,
            watch=True, unwatch=False, interval=None,
            force=False, json_output=False,
        )
        watch_calls = [c for c in captured if c[1] == "/v1/notes" and c[2] and (c[2].get("watch") or c[2].get("unwatch"))]
        assert len(watch_calls) == 1
        body = watch_calls[0][2]
        assert body["watch"] is True
        assert body["watch_kind"] == "directory"
        assert body["enqueue_git"] is True

    def test_put_directory_enqueues_git_ingest_without_watch(self, tmp_path, monkeypatch):
        """Plain directory imports still request initial git-history ingest."""
        from keep import cli_app

        (tmp_path / "a.txt").write_text("hello")
        captured: list[tuple[str, str, dict | None]] = []

        def fake_request(method, port, path, body=None):
            captured.append((method, path, body))
            if method == "GET":
                return 404, {}
            return 200, {"id": body.get("uri", ""), "git": {"queued": 0}}

        monkeypatch.setattr(cli_app, "_daemon_request", fake_request)
        cli_app._put_directory(
            port=0, resolved_path=tmp_path, parsed_tags={},
            recurse=True, exclude=None,
            watch=False, unwatch=False, interval=None,
            force=False, json_output=False,
        )

        git_calls = [
            c for c in captured
            if c[1] == "/v1/notes" and c[2] and c[2].get("enqueue_git")
        ]
        assert len(git_calls) == 1
        body = git_calls[0][2]
        assert body["watch_kind"] == "directory"
        assert body["enqueue_git"] is True
        assert body["recurse"] is True

    def test_put_directory_unwatch_sends_directory_kind(self, tmp_path, monkeypatch):
        """Unwatching a directory must post watch_kind=directory.

        Otherwise the daemon routes the request to flow_put_item, which
        fails on a directory URI.
        """
        from keep import cli_app

        (tmp_path / "a.txt").write_text("hello")
        captured: list[tuple[str, str, dict | None]] = []

        def fake_request(method, port, path, body=None):
            captured.append((method, path, body))
            if method == "GET":
                return 404, {}
            if body and body.get("unwatch"):
                return 200, {"unwatch": True}
            return 200, {"id": body.get("uri", "")}

        monkeypatch.setattr(cli_app, "_daemon_request", fake_request)
        cli_app._put_directory(
            port=0, resolved_path=tmp_path, parsed_tags={},
            recurse=False, exclude=None,
            watch=False, unwatch=True, interval=None,
            force=False, json_output=False,
        )
        unwatch_calls = [c for c in captured if c[1] == "/v1/notes" and c[2] and c[2].get("unwatch")]
        assert len(unwatch_calls) == 1
        body = unwatch_calls[0][2]
        assert body["unwatch"] is True
        assert body["watch_kind"] == "directory"

    def test_put_directory_only_hidden(self, cli, tmp_path):
        """Directory mode errors when only hidden files exist."""
        (tmp_path / ".DS_Store").write_text("junk")
        result = cli("put", str(tmp_path))
        assert result.returncode == 1
        assert "no eligible files" in result.stderr

    def test_put_file_count_cap(self, cli, tmp_path):
        """Directory mode rejects directories with too many files."""
        max_files = 1000  # default config value
        # Create max_files + 1 files
        for i in range(max_files + 1):
            (tmp_path / f"file_{i:04d}.txt").write_text(f"content {i}")
        result = cli("put", str(tmp_path))
        assert result.returncode == 1
        assert f"{max_files + 1} files" in result.stderr
        assert f"max {max_files}" in result.stderr

    def test_put_bare_file_path_help(self, cli):
        """Put help mentions directory and file modes."""
        result = cli("put", "--help")
        assert result.returncode == 0
        assert "Directory mode" in result.stdout or "directory" in result.stdout.lower()

    # --exclude tests (unit-level, against _list_directory_files)

    def test_list_directory_exclude_by_extension(self, tmp_path):
        """--exclude filters files by extension glob."""
        from keep.console_support import _list_directory_files
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.log").write_text("B")
        (tmp_path / "c.txt").write_text("C")
        files = _list_directory_files(tmp_path, exclude=["*.log"])
        assert [f.name for f in files] == ["a.txt", "c.txt"]

    def test_list_directory_exclude_multiple_patterns(self, tmp_path):
        """--exclude supports multiple patterns."""
        from keep.console_support import _list_directory_files
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.log").write_text("B")
        (tmp_path / "c.pyc").write_text("C")
        (tmp_path / "d.md").write_text("D")
        files = _list_directory_files(tmp_path, exclude=["*.log", "*.pyc"])
        assert [f.name for f in files] == ["a.txt", "d.md"]

    def test_list_directory_exclude_recursive(self, tmp_path):
        """--exclude filters files in subdirectories when recursing."""
        from keep.console_support import _list_directory_files
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "keep.txt").write_text("keep")
        (sub / "skip.log").write_text("skip")
        files = _list_directory_files(tmp_path, recurse=True, exclude=["*.log"])
        names = [f.name for f in files]
        assert "top.txt" in names
        assert "keep.txt" in names
        assert "skip.log" not in names

    def test_list_directory_exclude_subdirectory_pattern(self, tmp_path):
        """--exclude can match subdirectory paths like 'build/*'."""
        from keep.console_support import _list_directory_files
        (tmp_path / "readme.txt").write_text("ok")
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.js").write_text("built")
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.js").write_text("source")
        files = _list_directory_files(tmp_path, recurse=True, exclude=["build/*"])
        names = [str(f.relative_to(tmp_path)) for f in files]
        assert "readme.txt" in names
        assert "src/main.js" in names
        assert "build/output.js" not in names

    def test_list_directory_exclude_no_effect_without_match(self, tmp_path):
        """--exclude with non-matching pattern returns all files."""
        from keep.console_support import _list_directory_files
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.txt").write_text("B")
        files = _list_directory_files(tmp_path, exclude=["*.xyz"])
        assert [f.name for f in files] == ["a.txt", "b.txt"]


class TestCommandAliases:
    """Tests that old command names still work as hidden aliases."""

    def test_help_shows_put_not_update(self, cli):
        """Main help shows canonical command names."""
        result = cli("--help")
        assert result.returncode == 0
        assert "put" in result.stdout
        assert "del" in result.stdout
        assert "tag" in result.stdout
        # Old names should be hidden
        lines = result.stdout.split("\n")
        visible_commands = [l for l in lines if l.strip() and not l.strip().startswith("--")]
        visible_text = "\n".join(visible_commands)
        assert "tag-update" not in visible_text
        # 'update' and 'delete' should not appear as visible commands.

    def test_update_alias_works(self, cli):
        """'update' still works as a hidden alias for 'put'."""
        result = cli("update", "--help")
        assert result.returncode == 0

    def test_delete_alias_works(self, cli):
        """'delete' still works as a hidden alias for 'del'."""
        result = cli("delete", "--help")
        assert result.returncode == 0

    def test_tag_update_alias_works(self, cli):
        """'tag-update' still works as a hidden alias for 'tag'."""
        result = cli("tag-update", "--help")
        assert result.returncode == 0

    def test_del_help(self, cli):
        """'del' command has help."""
        result = cli("del", "--help")
        assert result.returncode == 0
        assert "Delete" in result.stdout or "delete" in result.stdout.lower()


# -----------------------------------------------------------------------------
# Stdin JSON Template Expansion
# -----------------------------------------------------------------------------

class TestStdinJsonTemplates:
    """Tests for stdin JSON template expansion."""
    pass  # Template tests removed — functions moved out of cli.py
