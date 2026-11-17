from __future__ import annotations
from memx.pipeline.windowing import make_windows, split_window
from memx.types import Messages, Text, Turn


def test_short_input_is_one_window():
    w = make_windows(Messages(turns=[Turn(role="user", content="hi")]), window_tokens=2500, context_turns=2,
                     chars_per_token=3.6)
    assert len(w) == 1 and w[0].context == []


def test_long_transcript_splits_on_turns_with_context():
    turns = [Turn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i} " + "word " * 200)
             for i in range(40)]
    windows = make_windows(Messages(turns=turns), window_tokens=500, context_turns=2, chars_per_token=3.6)
    assert len(windows) > 5
    max_chars = 500 * 3.6
    flat = [u.text for w in windows for u in w.units]
    assert flat == [t.content.strip() for t in turns]  # nothing lost, order kept, no turn split
    for prev, cur in zip(windows, windows[1:]):
        assert sum(len(u.text) for u in cur.units) <= max_chars or len(cur.units) == 1
        assert [c.text for c in cur.context] == [u.text[-500:] for u in (prev.units)[-2:]] or len(prev.units) < 2


def test_giant_turn_and_document_are_split():
    big = "\n\n".join(f"Paragraph {i}. " + "Sentence here. " * 60 for i in range(10))
    w1 = make_windows(Messages(turns=[Turn(role="user", content=big)]), window_tokens=300, context_turns=2,
                      chars_per_token=3.6)
    assert len(w1) > 3 and all(len(u.text) <= 1080 for w in w1 for u in w.units)
    w2 = make_windows(Text(text=big, source="wiki"), window_tokens=300, context_turns=2, chars_per_token=3.6)
    assert len(w2) > 3 and w2[1].context and w2[1].source == "wiki"
    a, b = split_window(w2[0], 2)
    if len(w2[0].units) > 1:
        assert a.units + b.units == w2[0].units
    else:  # one long unit is split inside the unit, at a sentence boundary, losing nothing
        assert a.units and b.units
        assert " ".join(u.text for u in a.units + b.units).split() == w2[0].units[0].text.split()


def test_a_single_long_unit_can_be_split():
    from memx.pipeline.windowing import Unit, Window, can_split

    long = Window(0, "messages", [Unit("user", "One fact here. " * 60)])
    assert can_split(long) and not can_split(Window(0, "messages", [Unit("user", "Short.")]))
    a, b = split_window(long, 2)
    assert len(a.units) == len(b.units) == 1 and a.units[0].role == "user"
    assert " ".join([a.units[0].text, b.units[0].text]).split() == long.units[0].text.split()
