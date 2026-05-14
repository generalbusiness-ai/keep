"""Centralized constants that determine system behavior.

Import from here instead of hardcoding magic numbers.
Add new constants here when a value is used in more than one module.
"""

# -- Database filenames (relative to store_path) --
WORK_QUEUE_DB = "continuation.db"
PENDING_SUMMARIES_DB = "pending_summaries.db"
DOCUMENTS_DB = "documents.db"

# -- Daemon --
DAEMON_PORT = 5337
DAEMON_PORT_FILE = ".daemon.port"
DAEMON_TOKEN_FILE = ".daemon.token"
DAEMON_PID_FILE = ".daemon.pid"
OPS_LOG_FILE = "keep-ops.log"

# Seconds the daemon may run with zero processed work, zero failed work, zero
# flow activity, and no active watches/mirrors/timers before it exits and
# releases its resources. Set to 0 to disable idle-exit entirely (the daemon
# only stops on signal or version-restart). Override at runtime with the
# KEEP_DAEMON_IDLE_SECONDS environment variable.
DAEMON_IDLE_EXIT_SECONDS = 600

# -- SQLite --
SQLITE_BUSY_TIMEOUT_MS = 5000

# -- Work queue / processing --
DEFAULT_LEASE_SECONDS = 120
BACKOFF_BASE_SECONDS = 30
BACKOFF_MAX_SECONDS = 3600

# -- Logging --
OPS_LOG_MAX_BYTES = 1_000_000
OPS_LOG_BACKUP_COUNT = 3

# -- Flow state names --
# Public named states and private compatibility states are centralized here
# so flow-level contracts stay visible and reviewable in one place.
#
# WARNING:
# If a state-doc name needs special handling in code, treat that as design
# debt, not as a normal extension mechanism. A small centralized allowlist is
# acceptable as a temporary compatibility bridge, but each such case should be
# reviewed for refactoring into a more explicit and extensible contract. If
# special-casing proliferates, the flow system is failing its extensibility
# goal.

# Public flow state names.
STATE_PUT = "put"
STATE_STUB = "stub"
STATE_ASSESS = "assess"
STATE_TAG = "tag"
STATE_DELETE = "delete"
STATE_MOVE = "move"
STATE_LIST = "list"
STATE_GET_CONTEXT = "get"
STATE_FIND_DEEP = "find-deep"
STATE_QUERY_RESOLVE = "query-resolve"
STATE_PROMPT = "prompt"

# Private compatibility flow names.
# These are adapter labels used by flow_client helpers together with inline
# state docs. They preserve the older exact-get / exact-find method contracts
# while the public named `get` flow remains context-assembling.
STATE_COMPAT_GET_ITEM = "compat-get-item"
STATE_COMPAT_FIND = "compat-find"

# -- Flow rendering --
# TODO: Remove this state-name allowlist once note-first rendering is driven
# by an explicit flow/render contract rather than implicit state identity.
# State docs listed here have explicit note-first rendering on the text
# tool/CLI surface. Keep these special cases centralized and reviewable;
# do not infer behavior from ad hoc state-name checks in renderers.
FLOW_NOTE_FIRST_RENDER_STATES = frozenset({STATE_GET_CONTEXT})
