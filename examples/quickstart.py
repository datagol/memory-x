"""memx in one file: remember a conversation, then recall it for another agent.

Needs Postgres with pgvector (``docker compose up -d postgres``) and OPENROUTER_API_KEY in the environment or .env.
Run: python examples/quickstart.py
"""

import asyncio

from memx import Memory, MemxConfig, Messages, Turn


async def main() -> None:
    memory = await Memory.connect(MemxConfig(insert_mode="inline"))  # inline: insert returns once facts are stored
    try:
        # Scopes run from general to specific. What is stored for alex is visible to alex's agents, not to others.
        receipt = await memory.insert(
            "memory.acme.alex",
            Messages(turns=[
                Turn(role="user", content="I lead the data platform team. We never deploy on Fridays."),
                Turn(role="assistant", content="Noted."),
            ]),
        )
        for fact in receipt.facts:
            print(f"stored [{fact.category.value}] {fact.text}")

        # An agent working for alex asks before acting. The answer is a markdown block ready for its prompt.
        recall = await memory.recall("memory.acme.alex.agent1", "can I ship this on Friday?")
        print(recall.markdown)
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
