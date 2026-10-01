"""Conversation-layout checks over CDP: does the chat behave like a chat?

check_ui_live.py only ever sees the landing page and check_ui_interactive.py
clicks through a single turn. Neither of them can catch the class of bug this
script exists for, because the bug only appears *after* an answer arrives:

    the newest answer is hidden behind the question box, and the user has to
    scroll to find it.

That is not hypothetical. The deployed build pinned the question box with
`position: fixed; bottom: 0`. A fixed box is positioned against the viewport,
so it reserves no space in the scroll content, and two things broke at once:
Streamlit's scroll-to-bottom stopped 279px short of the end, and the answer's
last line sat 19px underneath the bar. Measured A/B on that build versus the
fix is in the summary of the change.

So this script asks real questions and then measures the geometry:

  * the bar is sticky, never fixed
  * after each answer the view is at the bottom of the conversation
  * the newest answer is entirely above the bar, by a sane margin -- close
    enough to read as attached, far enough not to look cut off
  * the same holds at desktop, laptop, tablet and phone widths
  * the send button is centred on the text box
  * turns are compact, and the conversation does not grow a dead zone
  * no assistant turn came back as an error alert

That last one is not a layout check, but it belongs here: a layout gate that
asserts the answer box is well-formed will happily pass while every answer in
the app is an error message. It is exactly what caught the
`UnserializableReturnValueError` from `@st.cache_data` on RetrievedChunk.

Usage
-----
    streamlit run src/app.py --server.port 8501 &
    python scripts/check_ui_chat.py

Needs Chrome, `pip install websockets`, the built index and GROQ_API_KEY: it
makes real model calls. `websockets` is deliberately NOT in requirements.txt --
these are developer checks, and a dev-only dependency in the app manifest would
change the Render build. Exits non-zero on any failure. Override with
CHROME_PATH / UI_URL / UI_CDP_PORT.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

CHROME = os.environ.get(
    "CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
URL = os.environ.get("UI_URL", "http://localhost:8501")
PORT = int(os.environ.get("UI_CDP_PORT", "9335"))

# How many questions to ask. Three is enough to prove the conversation grows,
# that the rhythm holds past the first turn, and that the newest answer is the
# one on screen rather than an earlier one.
TURNS = 3

QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap?",
    "What is the benchmark of HDFC Flexi Cap?",
    "What is the minimum SIP amount for HDFC ELSS Tax Saver?",
]

# The clearance band between the bottom of the newest answer and the top of the
# input bar, in CSS pixels.
#
# The lower bound is the overlap guard: the deployed build measured -19px, i.e.
# the answer really was under the bar. The upper bound is the dead-zone guard:
# rendering the disclaimer footer after every turn measured 163px on desktop
# and 202px on a phone, which reads as "the answer has been cut off" and is what
# made people scroll. Comfortable is a small, deliberate gap.
CLEARANCE_MIN = 8
CLEARANCE_MAX = 110

# Vertical centre difference between the send button and the text box, in px.
SEND_CENTRE_TOLERANCE = 1

# Example-card budget on the landing page. Four across gave 188px per card for a
# 40-44 character question, so each wrapped to 4-5 lines and stood 99px tall --
# a content card, not a suggestion. A 2x2 grid gives ~385px and two lines.
EXAMPLE_MIN_WIDTH = 300
EXAMPLE_MAX_HEIGHT = 72

SIDEBAR_MAX_WIDTH = 280

# Widths re-measured once a conversation is on screen. Nothing here triggers a
# model call; the transcript lives in session state, so resizing is free.
RESPONSIVE = [(1440, 900), (1280, 800), (1024, 768), (820, 1180), (390, 844)]

_PROFILE_DIR = None
failures = []


def check(label, ok, detail=""):
    if ok:
        print(f"PASS: {label}")
    else:
        failures.append(f"{label} :: {detail}")
        print(f"FAIL: {label}  {detail}")


def start_chrome():
    global _PROFILE_DIR
    if not os.path.exists(CHROME):
        raise RuntimeError(
            f"Chrome not found at {CHROME}. Set CHROME_PATH to your binary."
        )
    _PROFILE_DIR = tempfile.mkdtemp(prefix="hdfc-ui-chat-")
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={_PROFILE_DIR}",
            "--window-size=1440,900",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/json/version", timeout=1
            ):
                return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("chrome devtools endpoint never came up")


class Session:
    def __init__(self, ws):
        self.ws, self._id, self.errs = ws, 0, []

    async def send(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method,
                                       "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=90))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def ev(self, js):
        """Evaluate and insist on a value.

        A probe that returns None is treated as a failure, never as a pass. An
        earlier version of this family of checks swallowed a JavaScript
        exception and reported PASS on a measurement that never happened.
        """
        r = await self.send("Runtime.evaluate", expression=js,
                            returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            d = r["exceptionDetails"]
            raise RuntimeError(d.get("text", "js error") + " " + str(
                d.get("exception", {}).get("description", ""))[:300])
        value = r["result"].get("value")
        if value is None:
            raise RuntimeError("probe returned None (JS error or missing node)")
        return value


async def connect():
    raw = urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/json/list", timeout=10
    ).read()
    page = next(t for t in json.loads(raw) if t["type"] == "page")
    ws = await websockets.connect(
        page["webSocketDebuggerUrl"], max_size=40 * 1024 * 1024
    )
    session = Session(ws)
    await session.send("Runtime.enable")
    return ws, session


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

LANDING = """
(() => {
  const R = e => { if (!e) return null; const b = e.getBoundingClientRect();
    return {t: Math.round(b.top), b: Math.round(b.bottom),
            h: Math.round(b.height), l: Math.round(b.left),
            w: Math.round(b.width)}; };
  const block = document.querySelector('[data-testid="stHorizontalBlock"]');
  const cards = block
    ? [...block.querySelectorAll('.stButton > button')].map(b => ({
        ...R(b), txt: (b.innerText || '').trim()}))
    : [];
  const bottom = document.querySelector('[data-testid="stBottom"]');
  const side = document.querySelector('[data-testid="stSidebar"]');
  const de = document.documentElement;
  return JSON.stringify({
    cards,
    cardRows: [...new Set(cards.map(c => c.t))].length,
    blockW: block ? Math.round(block.getBoundingClientRect().width) : null,
    bottomPosition: bottom ? getComputedStyle(bottom).position : null,
    sidebarW: side ? Math.round(side.getBoundingClientRect().width) : null,
    hasFooter: !!document.querySelector('.footer-legal'),
    hasStrip: !!document.querySelector('.disclaimer-strip'),
    hasPill: !!document.querySelector('.facts-pill'),
    hasBrand: !!document.querySelector('.app-brand'),
    overflowX: de.scrollWidth - de.clientWidth,
  });
})()
"""

# Geometry after an answer has landed. `bottom` is the top of the input bar;
# `clearance` is positive when the newest answer finishes above it.
CONVERSATION = """
(() => {
  const R = e => { if (!e) return null; const b = e.getBoundingClientRect();
    return {t: Math.round(b.top), b: Math.round(b.bottom),
            h: Math.round(b.height), l: Math.round(b.left),
            w: Math.round(b.width)}; };
  const scroll = document.querySelector('[data-testid="ScrollToBottomContainer"]');
  const msgs = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
  const last = msgs.length ? msgs[msgs.length - 1] : null;
  const bar = document.querySelector('[data-testid="stBottom"]');
  const ta = document.querySelector('[data-testid="stChatInputTextArea"]');
  const sb = document.querySelector('[data-testid="stChatInputSubmitButton"]');
  const tr = ta ? ta.getBoundingClientRect() : null;
  const br = sb ? sb.getBoundingClientRect() : null;
  const de = document.documentElement;
  const msgs_ = msgs.map((m, i) => {
    const r = m.getBoundingClientRect();
    const cs = getComputedStyle(m);
    const next = msgs[i + 1] ? msgs[i + 1].getBoundingClientRect() : null;
    const alert = m.querySelector('[data-testid="stAlert"]');
    return {
      role: m.querySelector('.user-bubble') ? 'user' : 'assistant',
      h: Math.round(r.h || r.height),
      padTop: cs.paddingTop,
      padBottom: cs.paddingBottom,
      marginBottom: cs.marginBottom,
      gapToNext: next ? Math.round(next.top - r.bottom) : null,
      alertText: alert ? (alert.innerText || '').trim().slice(0, 160) : '',
      text: (m.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 90),
    };
  });
  return JSON.stringify({
    msgCount: msgs.length,
    turns: msgs_,
    atBottom: scroll
      ? Math.abs((scroll.scrollHeight - scroll.clientHeight) - scroll.scrollTop) < 6
      : null,
    scrollTop: scroll ? Math.round(scroll.scrollTop) : null,
    scrollMax: scroll ? scroll.scrollHeight - scroll.clientHeight : null,
    lastBottom: last ? Math.round(last.getBoundingClientRect().bottom) : null,
    barTop: bar ? Math.round(bar.getBoundingClientRect().top) : null,
    clearance: (last && bar)
      ? Math.round(bar.getBoundingClientRect().top
                   - last.getBoundingClientRect().bottom) : null,
    barPosition: bar ? getComputedStyle(bar).position : null,
    taCentre: tr ? Math.round((tr.top + tr.bottom) / 2) : null,
    sbCentre: br ? Math.round((br.top + br.bottom) / 2) : null,
    sbHeight: br ? Math.round(br.height) : null,
    sbColour: sb ? getComputedStyle(sb).backgroundColor : null,
    hasFooter: !!document.querySelector('.footer-legal'),
    hasStrip: !!document.querySelector('.disclaimer-strip'),
    overflowX: de.scrollWidth - de.clientWidth,
  });
})()
"""

ALERTS = """
(() => {
  const msgs = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
  const bad = [];
  msgs.forEach((m, i) => {
    const a = m.querySelector('[data-testid="stAlert"]');
    if (a) bad.push({i, text: (a.innerText || '').replace(/\\s+/g, ' ').trim()
                                 .slice(0, 200)});
  });
  const empty = [];
  msgs.forEach((m, i) => {
    if (!m.querySelector('.user-bubble') && !(m.innerText || '').trim()) {
      empty.push(i);
    }
  });
  return JSON.stringify({bad, empty});
})()
"""


async def wait_for(session, js, tries=90):
    """Wait for an expression to be truthy, without raising on the way.

    Probing before Streamlit has finished its first paint is the classic way to
    write a check that measures nothing: a null testid reads as "the thing is
    absent" and gets reported as a failure of the page rather than of the wait.
    """
    for _ in range(tries):
        try:
            if await session.ev(
                "(() => { try { return !!(" + js + "); }"
                " catch (e) { return false; } })()"
            ):
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def resize(session, width, height, mobile):
    await session.send(
        "Emulation.setDeviceMetricsOverride", width=width, height=height,
        deviceScaleFactor=1, mobile=(width < 500),
    )
    await asyncio.sleep(1.6)


async def ask(session, text):
    """Type into the real chat input and press Enter, as a user would."""
    # The probe has to yield a value: Session.ev treats a None result as a JS
    # error, and `.focus()` returns undefined, which arrives over the wire as
    # null. That is the probe misfiring, not the page.
    await session.ev(
        "(() => { const t = document.querySelector("
        "'[data-testid=\\'stChatInputTextArea\\']');"
        " if (!t) return false; t.focus(); return true; })()"
    )
    await asyncio.sleep(0.4)
    await session.send("Input.insertText", text=text)
    await asyncio.sleep(0.8)
    # `text` belongs to keyDown only. Passing an explicit null on keyUp is
    # rejected by CDP with -32602 "expected string value".
    await session.send(
        "Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter",
        windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, text="\r",
    )
    await session.send(
        "Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter",
        windowsVirtualKeyCode=13, nativeVirtualKeyCode=13,
    )


async def wait_for_answer(session, previous, tries=100):
    """Wait until a new assistant turn exists and the page has settled.

    Waits on the message count, not on a timer: the wait has to cover a real
    model call, which in testing has ranged from 0.3s to well over 8s.
    """
    for _ in range(tries):
        await asyncio.sleep(1.5)
        try:
            data = json.loads(await session.ev(CONVERSATION))
        except Exception:
            continue
        if data["msgCount"] > previous and data["msgCount"] % 2 == 0:
            await asyncio.sleep(2.5)
            return True
    return False


async def run(session):
    await session.send("Page.enable")
    await session.send("Page.navigate", url=URL)

    # First paint is not one event. The wordmark arrives first, the sidebar and
    # the chat input stream in after it, and a probe in between measures a
    # half-built page. Wait for the parts this script measures.
    ready = all([
        await wait_for(session, "document.querySelector('.app-brand')"),
        await wait_for(session,
                       "document.querySelector('[data-testid=\"stSidebar\"]')"),
        await wait_for(session,
                       "document.querySelector('[data-testid=\"stBottom\"]')"),
        await wait_for(session,
                       "document.querySelector('[data-testid=\"stChatInput"
                       "TextArea\"]')"),
    ])
    if not ready:
        check("app hydrates", False,
              f"page never finished painting at {URL}: missing a header, "
              "sidebar, question bar or text box")
        return
    await asyncio.sleep(2.5)
    check("app hydrates", True, f"{URL}")

    # ---- landing ----------------------------------------------------------
    print()
    print("Landing page:")
    land = json.loads(await session.ev(LANDING))
    check("wordmark present", bool(land["hasBrand"]))
    check("facts-only pill present", bool(land["hasPill"]))
    check("short disclaimer strip present", bool(land["hasStrip"]))

    # The reported bug, asserted on the computed style of the real element.
    check(
        "question bar is sticky, not fixed to the viewport",
        land["bottomPosition"] == "sticky",
        f"computed position = {land['bottomPosition']!r}; a fixed bar reserves "
        "no space in the scroll content, which is what hid the answers",
    )

    check(
        "sidebar is not oversized",
        land["sidebarW"] is not None and 0 < land["sidebarW"] <= SIDEBAR_MAX_WIDTH,
        f"{land['sidebarW']}px (limit {SIDEBAR_MAX_WIDTH}px)",
    )
    check(
        "example suggestions are compact",
        bool(land["cards"])
        and all(c["h"] <= EXAMPLE_MAX_HEIGHT for c in land["cards"]),
        f"heights {[c['h'] for c in land['cards']]} "
        f"(limit {EXAMPLE_MAX_HEIGHT}px each)",
    )
    check(
        "example suggestions are wide enough to read on two lines",
        bool(land["cards"])
        and all(c["w"] >= EXAMPLE_MIN_WIDTH for c in land["cards"]),
        f"widths {[c['w'] for c in land['cards']]} "
        f"(need >= {EXAMPLE_MIN_WIDTH}px; four-across gave 188px)",
    )
    check(
        "example suggestions form a 2x2 grid",
        land["cardRows"] == 2 and len(land["cards"]) == 4,
        f"{len(land['cards'])} cards in {land['cardRows']} rows",
    )
    check(
        "full disclaimer shown on the landing page",
        land["hasFooter"],
        "landing page is the one place the long form appears",
    )
    check("no horizontal overflow on the landing page",
          land["overflowX"] == 0, f"{land['overflowX']}px")

    # ---- conversation -----------------------------------------------------
    print()
    print(f"Conversation ({TURNS} typed questions):")
    for index, question in enumerate(QUESTIONS, 1):
        before = json.loads(await session.ev(CONVERSATION))["msgCount"]
        await ask(session, question)
        landed = await wait_for_answer(session, before)
        if not landed:
            check(f"turn {index}: answer rendered", False,
                  f"no assistant turn after 150s: {question!r}")
            return
        data = json.loads(await session.ev(CONVERSATION))
        check(f"turn {index}: answer rendered", True, data["turns"][-1]["text"])

        alerts = json.loads(await session.ev(ALERTS))
        check(
            f"turn {index}: answer is not an error notice",
            not alerts["bad"] and not alerts["empty"],
            json.dumps(alerts)[:260],
        )

        check(
            f"turn {index}: view scrolled to the newest answer",
            data["atBottom"],
            f"scrollTop {data['scrollTop']} of {data['scrollMax']}",
        )
        check(
            f"turn {index}: newest answer clears the question bar",
            data["clearance"] >= CLEARANCE_MIN,
            f"clearance {data['clearance']}px (was -19px, answer under the bar)",
        )
        check(
            f"turn {index}: no dead zone above the question bar",
            data["clearance"] <= CLEARANCE_MAX,
            f"clearance {data['clearance']}px (was 163px desktop / 202px phone "
            "when the footer was drawn after every turn)",
        )

    final = json.loads(await session.ev(CONVERSATION))
    check(
        "the whole conversation is still on the page",
        final["msgCount"] == 2 * TURNS,
        f"{final['msgCount']} messages for {TURNS} questions "
        f"(expected {2 * TURNS})",
    )
    check(
        "turns are compact: no padding baked into each message",
        all(t["padTop"] == "0px" and t["padBottom"] == "0px"
            for t in final["turns"]),
        f"paddings {[t['padTop'] for t in final['turns']]} -- Streamlit's own "
        "16px top and bottom per turn is what made the chat feel airy",
    )
    gaps = [t["gapToNext"] for t in final["turns"] if t["gapToNext"] is not None]
    check(
        "gaps between turns are deliberate, not accidental",
        bool(gaps) and all(g <= 80 for g in gaps),
        f"gaps {gaps}",
    )
    check(
        "disclaimer is not drawn between the conversation and the input",
        not final["hasFooter"],
        "a 123px footer after every turn is what created the apparent dead zone",
    )
    check(
        "disclaimer still reachable during a conversation",
        final["hasStrip"],
        "the sidebar strip carries the short form; the header carries the pill",
    )

    delta = abs((final["sbCentre"] or 0) - (final["taCentre"] or 0))
    check(
        "send button is vertically centred on the text box",
        delta <= SEND_CENTRE_TOLERANCE,
        f"centres differ by {delta}px (was 3px, caused by the parent flex "
        "box being align-items: flex-end)",
    )
    check(
        "send button is visibly the primary action",
        final["sbColour"] in ("rgb(15, 157, 88)", "rgb(0, 128, 0)"),
        f"background {final['sbColour']!r}",
    )

    # ---- responsive -------------------------------------------------------
    print()
    print(f"Responsive ({len(RESPONSIVE)} widths, same conversation):")
    for width, height in RESPONSIVE:
        await resize(session, width, height, width < 500)
        try:
            data = json.loads(await session.ev(CONVERSATION))
        except Exception as exc:
            check(f"{width}x{height}: measurable", False,
                  f"probe failed ({type(exc).__name__}: {exc})")
            continue
        label = f"{width}x{height}"
        check(
            f"{label}: newest answer clears the question bar",
            data["clearance"] is not None and CLEARANCE_MIN <= data["clearance"]
            <= CLEARANCE_MAX,
            f"clearance {data['clearance']}px (need {CLEARANCE_MIN}"
            f"-{CLEARANCE_MAX}px)",
        )
        check(
            f"{label}: still scrolled to the newest answer",
            data["atBottom"],
            f"scrollTop {data['scrollTop']} of {data['scrollMax']}",
        )
        check(
            f"{label}: no horizontal overflow",
            data["overflowX"] == 0,
            f"{data['overflowX']}px",
        )

    # ---- reader control ----------------------------------------------------
    # The control for the check above. Every assertion above holds trivially if
    # the scroll code simply forced the view to the bottom unconditionally, so
    # prove the opposite case: a reader who scrolls back up to re-read something
    # must stay where they put it, and a resize must not yank them down again.
    # If this section is removed the gate still goes green while the behaviour is
    # broken, which is the failure mode this whole file exists to avoid.
    print()
    print("Reader control (the opposite case):")
    await resize(session, 1440, 900, False)
    await session.ev(
        "(() => { const b = document.querySelector("
        "'[data-testid=\"ScrollToBottomContainer\"]');"
        " if (!b) return false; b.scrollTop = b.scrollHeight; return true; })()"
    )
    await asyncio.sleep(1.5)
    pinned = json.loads(await session.ev(CONVERSATION))
    check(
        "reader can be at the bottom to begin with",
        pinned["scrollMax"] - pinned["scrollTop"] <= 6,
        f"{pinned['scrollTop']} of {pinned['scrollMax']}",
    )

    # Wheel up, the way a person reads back through a long answer.
    for _ in range(8):
        await session.send(
            "Input.dispatchMouseEvent", type="mouseWheel", x=760, y=420,
            deltaX=0, deltaY=-200,
        )
        await asyncio.sleep(0.2)
    await asyncio.sleep(2.0)
    backed = json.loads(await session.ev(CONVERSATION))
    check(
        "a reader can scroll back up through the conversation",
        backed["scrollMax"] - backed["scrollTop"] > 40,
        f"{backed['scrollTop']} of {backed['scrollMax']} "
        f"(gap {backed['scrollMax'] - backed['scrollTop']}px)",
    )

    await resize(session, 1280, 800, False)
    await asyncio.sleep(1.5)
    after = json.loads(await session.ev(CONVERSATION))
    check(
        "resizing does not yank a reader who scrolled back up",
        after["scrollMax"] - after["scrollTop"] > 40,
        f"{after['scrollTop']} of {after['scrollMax']} -- still "
        f"{after['scrollMax'] - after['scrollTop']}px short of the bottom, "
        "so following is tracked from reader intent, not forced",
    )


async def main():
    chrome = start_chrome()
    ws = None
    try:
        ws, session = await connect()
        await run(session)
    finally:
        if ws is not None:
            await ws.close()
        chrome.kill()
        if _PROFILE_DIR:
            shutil.rmtree(_PROFILE_DIR, ignore_errors=True)

    print()
    if failures:
        print(f"RESULT: {len(failures)} conversation check(s) failed")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: the chat scrolls to the newest answer, keeps it clear of "
          "the question bar at every width, and leaves no dead zone")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))