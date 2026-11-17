from __future__ import annotations
from datetime import UTC, datetime
import pytest
from memx.config import MemxConfig
from memx.types import Text


CFG = MemxConfig(_env_file=None)


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


async def test_truncated_extraction_splits_the_window_or_fails_clearly():
    from types import SimpleNamespace

    from memx.pipeline import extract as px
    from memx.providers.extractor import ExtractResult
    from memx.providers.http import ProviderError

    class Extractor:
        model = "fake"

        async def extract(self, window, *, observed_at, total_windows):
            if len(window.units) > 1:  # pretend the whole window is too much for one response
                return ExtractResult(facts=[], truncated=True)
            return ExtractResult(facts=[{"text": window.units[0].text}])

    ctx = SimpleNamespace(config=CFG, extractor=Extractor())
    [window] = px.windows_for(ctx, Text(text="First paragraph.\n\nSecond paragraph.").model_dump())
    assert len(window.units) == 2
    facts, _, _ = await px._extract_window(ctx, window, NOW, 1)
    assert [f["text"] for f in facts] == ["First paragraph.", "Second paragraph."]

    [single] = px.windows_for(ctx, Text(text="One paragraph only.").model_dump())

    class Always(Extractor):
        async def extract(self, window, *, observed_at, total_windows):
            return ExtractResult(facts=[], truncated=True)

    with pytest.raises(ProviderError, match="more facts than fit"):
        await px._extract_window(SimpleNamespace(config=CFG, extractor=Always()), single, NOW, 1)


def test_extractor_keeps_the_attribute_label():
    from memx.providers.extractor import TOOL, _clean

    item = TOOL["function"]["parameters"]["properties"]["facts"]["items"]
    assert "attribute" in item["required"]
    [f] = _clean([{"text": "Dana Whitfield's spouse/guest is Marisol.", "attribute": "spouse", "speaker": "document",
                   "category_hint": "relationship", "is_event": False, "confidence": 1, "entities": []}])
    assert f["attribute"] == "spouse"


async def test_a_capped_answer_is_continued_not_re_extracted():
    from types import SimpleNamespace

    from memx import usage
    from memx.pipeline import extract as px
    from memx.providers.extractor import ExtractResult

    class Extractor:
        model = "fake"
        supports_continue = True

        def __init__(self):
            self.calls = []

        async def extract(self, window, *, observed_at, total_windows, already=None):
            self.calls.append(list(already or []))
            start = len(already or [])
            facts = [{"text": f"Fact {i}."} for i in range(start, min(start + 25, 30))]
            return ExtractResult(facts=facts, tokens_in=100, tokens_out=10 * len(facts))

    ex = Extractor()
    ctx = SimpleNamespace(config=CFG, extractor=ex)
    [window] = px.windows_for(ctx, Text(text="First paragraph.\n\nSecond paragraph.").model_dump())
    with usage.track() as u:
        facts, _, tout = await px._extract_window(ctx, window, NOW, 1)
    assert [f["text"] for f in facts] == [f"Fact {i}." for i in range(30)]
    assert len(ex.calls) == 2 and len(ex.calls[1]) == 25  # one follow-up that saw the first 25
    assert u.extract_continuations == 1 and u.extract_splits == 0
    assert tout == 300  # no output paid twice


def test_slim_extractor_schema_and_continuation_prompt():
    from memx.pipeline.windowing import Unit, Window
    from memx.providers.extractor import SYSTEM_PROMPT, TOOL, _clean, render_window

    item = TOOL["function"]["parameters"]["properties"]["facts"]["items"]
    assert set(item["required"]) == {"text", "attribute", "speaker"}
    assert "category_hint" not in item["properties"] and "confidence" not in item["properties"]
    [f] = _clean([{"text": "The user ran a 5k on 2026-10-04.", "attribute": "race", "speaker": "user",
                   "event_date": "2026-10-04"}])
    assert f["is_event"] and f["entities"] == [] and f["confidence"] == 1.0
    assert "one fact per item" in SYSTEM_PROMPT and "changed from" in SYSTEM_PROMPT
    w = Window(0, "messages", [Unit("user", "Hi.")])
    body = render_window(w, observed_at=NOW, total_windows=1, already=["The user likes tea."])
    assert "<already_recorded>" in body and "- The user likes tea." in body


async def test_unusable_extractor_output_splits_the_window():
    from types import SimpleNamespace

    from memx.pipeline import extract as px
    from memx.providers.extractor import ExtractResult

    class Extractor:
        model = "fake"

        async def extract(self, window, *, observed_at, total_windows):
            if len(window.units) > 1:
                return ExtractResult(facts=[], failed="extractor returned unusable output twice")
            return ExtractResult(facts=[{"text": window.units[0].text}])

    ctx = SimpleNamespace(config=CFG, extractor=Extractor())
    [window] = px.windows_for(ctx, Text(text="First paragraph.\n\nSecond paragraph.").model_dump())
    facts, _, _ = await px._extract_window(ctx, window, NOW, 1)
    assert [f["text"] for f in facts] == ["First paragraph.", "Second paragraph."]
