"""HopProbe: questions whose answer sits in a fact that shares no content word with the question.

Each answer needs two facts: a bridge that names the question's subject ("Orion is owned by Ingrid Halvorsen") and an
answer fact about the bridge entity ("Ingrid Halvorsen lives in Trondheim"). Every answer fact has look-alikes about
other people or teams ("Kofi Mensah lives in Accra"), so search alone finds a pile of candidates and cannot tell which
one the question means; only the entity link can. This is the case the walk is for.

Scoring is exact and needs no judge: an answer is correct when it names the expected value and none of the
look-alike values of the same kind.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

OBSERVED = datetime(2026, 9, 15, 12, tzinfo=UTC)
QUESTION_DATE = datetime(2026, 10, 20, 12, tzinfo=UTC)
SCOPE = "memory.hop2"

PEOPLE = [  # name, city, favourite language, manager
    ("Ingrid Halvorsen", "Trondheim", "Rust", "Tobias Lindgren"),
    ("Kofi Mensah", "Accra", "Elixir", "Amara Okafor"),
    ("Mei Tanaka", "Sapporo", "OCaml", "Hiroshi Sato"),
    ("Lucas Ferreira", "Porto Alegre", "Kotlin", "Beatriz Costa"),
    ("Anya Petrova", "Kazan", "Haskell", "Dmitri Volkov"),
    ("Rahul Iyer", "Pune", "Scala", "Kavya Menon"),
    ("Sofia Rossi", "Bologna", "Clojure", "Marco Bianchi"),
    ("Liam O'Brien", "Galway", "Go", "Siobhan Walsh"),
    ("Nadia Haddad", "Beirut", "Julia", "Omar Khalil"),
    ("Pablo Ruiz", "Valencia", "Zig", "Lucia Navarro"),
]
MANAGERS = [  # name, city, favourite language
    ("Tobias Lindgren", "Uppsala", "Ada"), ("Amara Okafor", "Lagos", "Erlang"), ("Hiroshi Sato", "Kyoto", "Lisp"),
    ("Beatriz Costa", "Recife", "Lua"), ("Dmitri Volkov", "Novosibirsk", "Fortran"),
    ("Kavya Menon", "Kochi", "Perl"), ("Marco Bianchi", "Turin", "Pascal"), ("Siobhan Walsh", "Cork", "Prolog"),
    ("Omar Khalil", "Amman", "Smalltalk"), ("Lucia Navarro", "Seville", "Racket"),
]
SYSTEMS = [  # system, owner
    ("Orion", "Ingrid Halvorsen"), ("Helios", "Kofi Mensah"), ("Vega", "Mei Tanaka"), ("Atlas", "Lucas Ferreira"),
    ("Nimbus", "Anya Petrova"), ("Quasar", "Rahul Iyer"), ("Pulsar", "Sofia Rossi"), ("Zephyr", "Liam O'Brien"),
    ("Cobalt", "Nadia Haddad"), ("Tundra", "Pablo Ruiz"), ("Meridian", "Tobias Lindgren"), ("Lattice", "Amara Okafor"),
    ("Halcyon", "Hiroshi Sato"), ("Juniper", "Beatriz Costa"),
]
TEAMS = [  # team, pager number, service it runs
    ("Falcon team", "555-0141", "the invoicing service"), ("Heron team", "555-0172", "the search service"),
    ("Kestrel team", "555-0118", "the notifications service"), ("Osprey team", "555-0163", "the payments gateway"),
    ("Plover team", "555-0195", "the analytics pipeline"), ("Wren team", "555-0127", "the identity service"),
    ("Merlin team", "555-0154", "the email relay"), ("Swift team", "555-0186", "the data warehouse"),
    ("Tern team", "555-0109", "the mobile backend"), ("Egret team", "555-0133", "the feature-flag service"),
]


def handbook() -> str:
    lines = ["Platform directory (September 2026)."]
    for name, city, lang, manager in PEOPLE:
        lines += [f"{name} lives in {city}.", f"{name}'s favourite programming language is {lang}.",
                  f"{name} reports to {manager}."]
    for name, city, lang in MANAGERS:
        lines += [f"{name} lives in {city}.", f"{name}'s favourite programming language is {lang}."]
    for system, owner in SYSTEMS:
        lines.append(f"{system} is owned by {owner}.")
    for team, pager, service in TEAMS:
        lines += [f"The {team}'s pager number is {pager}.", f"{service[0].upper()}{service[1:]} is run by the {team}."]
    return "\n\n".join(lines)


def questions() -> list[dict]:
    cities = [p[1] for p in PEOPLE] + [m[1] for m in MANAGERS]
    langs = [p[2] for p in PEOPLE] + [m[2] for m in MANAGERS]
    pagers = [t[1] for t in TEAMS]
    city_of = {p[0]: p[1] for p in PEOPLE} | {m[0]: m[1] for m in MANAGERS}
    lang_of = {m[0]: m[2] for m in MANAGERS}
    out = []
    for system, owner in SYSTEMS:  # 14: system -> owner -> town
        out.append({"id": f"city-{system}", "q": f"In which town is the person responsible for {system} based?",
                    "answer": city_of[owner], "lookalikes": cities})
    for name, _, _, manager in PEOPLE:  # 10: person -> manager -> language
        first = name.split()[0]
        out.append({"id": f"lang-{first}", "q": f"Which coding language does {first}'s boss like best?",
                    "answer": lang_of[manager], "lookalikes": langs})
    for name, _, _, manager in PEOPLE:  # 10: person -> manager -> town
        first = name.split()[0]
        out.append({"id": f"bosstown-{first}", "q": f"Which town is {first}'s boss based in?",
                    "answer": city_of[manager], "lookalikes": cities})
    for team, pager, service in TEAMS:  # 10: service -> team -> pager
        out.append({"id": f"pager-{team.split()[0]}", "q": f"What number do I call to page whoever operates {service}?",
                    "answer": pager, "lookalikes": pagers})
    return out


def _says(text: str, value: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", text, re.IGNORECASE) is not None


def correct(response: str, q: dict) -> bool:
    if not _says(response, q["answer"]):
        return False
    return not any(_says(response, v) for v in q["lookalikes"] if v != q["answer"])
