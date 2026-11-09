"""Run the memx MCP server over stdio, for clients that start it as a local process:

    python -m memx_mcp            # reads MEMX_DSN, OPENROUTER_API_KEY etc. from the environment or .env

Inserts run in this process unless MEMX_INSERT_MODE=queued (then a `memx worker` must be running).
"""

from memx_mcp.server import mcp

def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
