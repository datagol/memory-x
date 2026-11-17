# Security

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub: the repository's **Security** tab, then
**Report a vulnerability**. Say what you found, how to reproduce it and what an attacker gains. You will get a reply
within five working days.

## What memx does and does not protect

- **One token grants everything.** `MEMX_API_TOKEN` gives access to every scope through the API, the dashboard and
  the MCP server. memx does not know who your users are. The application in front of it decides which scope a
  signed-in user may use, and must never pass a scope or a fact id straight from a client.
- **No token means local use only.** Without a token the API answers only requests addressed to this machine and
  the MCP server checks Host and Origin, so a web page cannot reach either through a browser. Set a token before
  publishing the ports.
- **Stored text is untrusted.** Facts come from conversations and documents. They reach the judge model under an
  `untrusted` key, are flagged or rejected when they look like instructions, and are rendered one per line so they
  cannot add headings to the block an agent reads. An agent should still treat recalled memory as information, not
  as orders.
- **Forget.** Forgetting a scope deletes its raw inputs, facts, decisions and recall logs. Forgetting one fact
  deletes the fact and its wording from logs, but the raw input it was extracted from stays until its scope is
  forgotten.
- **What leaves your network.** Text you insert is sent to OpenRouter for extraction, embedding and judging. Nothing
  else is sent anywhere.
