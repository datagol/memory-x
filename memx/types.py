from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Category(StrEnum):
    identity = "identity"
    preference = "preference"
    goal = "goal"
    rule = "rule"
    relationship = "relationship"
    work_context = "work_context"
    event = "event"
    procedure = "procedure"
    other = "other"


CATEGORY_DEFINITIONS: dict[str, str] = {
    "identity": "A stable attribute of a person or organization: name, role, title, location, language, background, expertise.",
    "preference": "What someone likes, dislikes, or how they want things done or presented.",
    "goal": "An outcome someone wants, a plan, an intention, or a project they are pursuing.",
    "rule": "A constraint, policy, requirement, or prohibition: must, must not, always, never.",
    "relationship": "A connection between people, teams, or organizations (reports to, works with, married to, client of).",
    "work_context": "Tools, systems, datasets, repositories, environments, or the current state of ongoing work.",
    "event": "Something that happened or is scheduled to happen at a particular time.",
    "procedure": "A method, workflow, or sequence of steps for doing something.",
    "other": "None of the above, or too vague to tell.",
}

# Order sections appear in when rendered.
RENDER_ORDER: list[Category] = [
    Category.identity,
    Category.rule,
    Category.preference,
    Category.goal,
    Category.relationship,
    Category.work_context,
    Category.procedure,
    Category.event,
    Category.other,
]

SECTION_TITLES: dict[Category, str] = {
    Category.identity: "Identity",
    Category.rule: "Rules",
    Category.preference: "Preferences",
    Category.goal: "Goals",
    Category.relationship: "Relationships",
    Category.work_context: "Work context",
    Category.procedure: "Procedures",
    Category.event: "Events",
    Category.other: "Other",
}

FactStatus = Literal["active", "superseded", "expired"]
RecallMode = Literal["full", "ranked"]

ENTITY_TYPES = ("person", "organization", "team", "system", "project", "place", "product", "other")

# Topics: one theme per fact, chosen by Jev from this fixed list inside the sort request. Fixed so that "diet",
# "food" and "cuisine" become one node instead of three. Keys are stored on the fact; definitions go to Jev.
TOPIC_DEFINITIONS: dict[str, str] = {
    "food_and_drink": "Food, cooking, restaurants, diet, drinks.",
    "health_and_fitness": "Health, medical matters, exercise, sleep, wellbeing.",
    "family_and_home": "Family members, partner, children, pets, housing, household.",
    "travel": "Trips, destinations, transport, commuting, holidays.",
    "hobbies_and_leisure": "Hobbies, sports as a fan or player, games, crafts, outdoor activities.",
    "arts_and_media": "Books, films, TV, music, podcasts, art.",
    "shopping_and_products": "Purchases, brands, devices owned or wanted, consumer products.",
    "finance": "Money, budgets, salaries, investments, prices, invoices, contracts.",
    "education_and_learning": "Study, courses, degrees, skills being learned, certifications.",
    "career_and_role": "Job title, employer, career moves, responsibilities, performance.",
    "people_and_relationships": "Friends, colleagues, reporting lines, clients, who knows whom.",
    "communication_style": "How someone wants to be addressed, tone, language, format, length of replies.",
    "scheduling_and_availability": "Working hours, time zones, meetings, availability, deadlines, calendar.",
    "leadership_and_decisions": "Approvals, strategy, decisions, ownership, who decides what.",
    "process_and_workflow": "How work is done: procedures, reviews, rituals, ways of working.",
    "software_development": "Code, repositories, languages, frameworks, testing, releases.",
    "data_and_analytics": "Datasets, databases, pipelines, dashboards, metrics, reporting.",
    "cloud_and_infrastructure": "Cloud providers, servers, networking, environments, hosting, migrations.",
    "security_and_compliance": "Access, permissions, policies, privacy, regulation, audits.",
    "product_and_design": "Product features, roadmap, UX, design choices, customers' needs.",
    "sales_and_marketing": "Deals, pipeline, campaigns, pricing, customers, partners.",
    "operations_and_logistics": "Supply, facilities, equipment, vendors, physical operations.",
    "legal": "Contracts, terms, intellectual property, disputes, legal obligations.",
    "ai_and_automation": "AI tools, models, agents, automation, prompts.",
    "documents_and_knowledge": "Files, notes, documentation, archives, where information lives.",
    "location_and_environment": "Where someone lives or works, offices, geography, weather, local matters.",
    "identity_and_background": "Name, age, nationality, languages spoken, personal history.",
    "goals_and_plans": "Ambitions, targets, intentions, upcoming projects.",
    "events_and_milestones": "Specific happenings: launches, incidents, celebrations, anniversaries.",
    "general": "No clear theme, or several at once.",
}
TOPICS = list(TOPIC_DEFINITIONS)


# ---- Input ---------------------------------------------------------------


class Turn(BaseModel):
    role: str = Field(description="user | assistant | system | tool, or a participant name")
    content: str
    at: datetime | None = None
    name: str | None = None


class Messages(BaseModel):
    kind: Literal["messages"] = "messages"
    turns: list[Turn]


class Text(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    source: str = "document"
    trust: float | None = Field(default=None, ge=0, le=1)


IngestInput = Annotated[Messages | Text, Field(discriminator="kind")]


# ---- Pipeline ------------------------------------------------------------


class EntityMention(BaseModel):
    """A named thing the extractor saw in a candidate, before resolution against stored entities."""

    name: str
    type: str = "other"
    aka: list[str] = Field(default_factory=list)  # other names the extractor is confident mean the same thing


class EntityRef(BaseModel):
    """A stored entity attached to a fact."""

    id: UUID
    name: str
    type: str = "other"
    surface: str | None = None


class Candidate(BaseModel):
    """An extracted statement not yet stored."""

    index: int = 0
    window: int = 0
    text: str
    speaker: str
    category_hint: Category | None = None
    attribute: str | None = None  # the single attribute the fact states ("role", "reading progress")
    is_event: bool = False
    time_text: str | None = None
    event_date: date | None = None
    confidence: float = 1.0
    entities: list[EntityMention] = Field(default_factory=list)
    outcome: str | None = None  # set once committed: added | dropped | rejected | superseded-on-arrival
    fact_id: UUID | None = None


class Fact(BaseModel):
    id: UUID
    scope: str
    category: Category
    text: str
    status: FactStatus
    superseded_by: UUID | None = None
    superseded_at: datetime | None = None
    speaker: str | None = None
    source: str | None = None
    trust: float
    sort_confidence: float
    observed_at: datetime
    event_at: datetime | None = None
    stale_after: datetime | None = None
    reinforced_count: int = 0
    last_reinforced_at: datetime | None = None
    access_count: int = 0
    last_accessed_at: datetime | None = None
    flags: dict[str, Any] = Field(default_factory=dict)
    ingest_id: UUID | None = None
    created_at: datetime
    token_count: int = 0
    topic: str | None = None
    topic_confidence: float | None = None
    entities: list[EntityRef] = Field(default_factory=list)


class PathStep(BaseModel):
    """One hop of the walk that reached a fact: from a fact, through an entity or topic."""

    from_fact: UUID
    via: str
    kind: Literal["entity", "topic"] = "entity"


class ScoredFact(Fact):
    relevance: float | None = None
    score: float
    found_by: Literal["search", "entity_hop", "topic_hop", "full"] = "search"
    path: list[PathStep] = Field(default_factory=list)


class InsertReceipt(BaseModel):
    ingest_id: UUID
    status: Literal["received", "extracted", "committed", "failed"]
    mode: str
    added: int = 0
    dropped: int = 0
    historical: int = 0
    superseded: int = 0
    shadowed: int = 0
    rejected: int = 0
    held_back: int = 0  # replacements Jev suggested but the policy held back; both facts stay, linked as related
    facts: list[Fact] | None = None
    error: str | None = None


class Recall(BaseModel):
    recall_id: UUID
    scope: str
    mode: RecallMode
    truncated: bool = False
    markdown: str
    facts: list[ScoredFact]
    token_count: int
    explain: dict[str, Any] | None = None


class ForgetReport(BaseModel):
    facts_removed: int
    ingests_removed: int
    scopes_removed: int


class HistoryEntry(BaseModel):
    fact: Fact
    decisions: list[dict[str, Any]]


class FactHistory(BaseModel):
    fact_id: UUID
    versions: list[HistoryEntry]
    links: list[dict[str, Any]]


class SweepReport(BaseModel):
    expired: int
    purged: int
    decisions_pruned: int
    jobs_pruned: int


class ExportReport(BaseModel):
    files: int
    scope: str
