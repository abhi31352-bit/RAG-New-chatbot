"""Gate: the question input must sit on the same vertical lines as the answers.

Why this exists
---------------
The chat input is the one element a visitor aims at before they have read
anything, so any horizontal offset reads as "the page is broken" rather than as
a 20px cosmetic slip. Three separate bugs shipped here, and none of them was
visible to AppTest, which never sees CSS or pixels:

  1. `[data-testid="stChatInput"]` was pinned with `position: fixed; left: 0;
     right: 0`. A fixed box is positioned against the viewport, not the content
     column, so its centred inner container was centred on the window. With the
     336px sidebar open the input sat ~398px to the LEFT of the answers.
  2. With that fixed, Streamlit's own nesting still left a ~20px gap: the input
     sits inside `[data-testid="stBottomBlockContainer"]` (which adds the same
     gutter `.block-container` gets) and then inside a
     `[data-testid="stVerticalBlockBorderWrapper"]` (which adds another ~19.4px
     of padding and a 1px border). `.block-container` gets neither. The offset
     is invisible at 1440 -- both boxes are centred there and the two insets
     cancel -- and plainly wrong at 1280 and below.
  3. The gutter is not a constant. Streamlit steps it 80px -> 16px at 768px ->
     12px at 480px. A hardcoded value is 4px out on a phone.

Measuring the textarea is not measuring the input: the textarea sits inside a
bordered box, and the box's outer edge is the line the eye compares against the
answer cards. This measures the box.

Usage
-----
    streamlit run src/app.py --server.port 8501 &
    python scripts/check_ui_align.py

Needs Chrome and `pip install websockets`. `websockets` is deliberately NOT in
requirements.txt: this is a developer check, and a dev-only dependency in the app
manifest would change the Render build. Set CHROME_PATH / UI_URL to override the
defaults, e.g. to run it against a hosted instance. Exits non-zero on any
failure, so it works as a gate.
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

# The input's own 1px border accounts for a 1px difference on each side, so the
# tolerance is 2px rather than 0. Anything larger is a real offset.
TOLERANCE_PX = 2

# The boundaries matter as much as the round numbers: 769/768 and 481/480 are
# exactly where the gutter steps, which is where a hardcoded padding breaks.
WIDTHS = [
    (1920, 1080), (1600, 900), (1500, 900), (1440, 900), (1400, 900),
    (1320, 900), (1280, 800), (1200, 900), (1100, 900), (1024, 800),
    (960, 900), (900, 900), (820, 1180),
    (769, 900), (768, 1024), (700, 900), (600, 900),
    (481, 900), (480, 900), (430, 900), (414, 896), (390, 844), (375, 812),
    (360, 780), (320, 700),
]

# A sidebar-collapsed pass is deliberately absent. Streamlit 1.38 renders no
# desktop sidebar collapse control at all: [data-testid="stSidebarCollapseButton"]
# computes to display:none and no expand control replaces it. Verified against a
# minimal Streamlit app with zero custom CSS, so this is upstream behaviour, not
# something the redesign caused. A first attempt simulated the collapsed column
# by forcing the sidebar's width to 0; the sidebar did not actually shrink
# (Streamlit sizes it on an inner element), so the "collapsed" measurements were
# really the open ones wearing a label. Shipping that would be a check that
# passes without checking. The widths below 768px already run with no sidebar,
# which is the same wide-main-column geometry.

_PROFILE_DIR = None


def start_chrome() -> subprocess.Popen:
    global _PROFILE_DIR
    if not os.path.exists(CHROME):
        raise RuntimeError(
            f"Chrome not found at {CHROME}. Set CHROME_PATH to your binary."
        )
    _PROFILE_DIR = tempfile.mkdtemp(prefix="hdfc-ui-align-")
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


def _cleanup() -> None:
    if _PROFILE_DIR:
        shutil.rmtree(_PROFILE_DIR, ignore_errors=True)


class Session:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    async def send(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps(
            {"id": mid, "method": method, "params": params}
        ))
        while True:
            raw = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=60))
            if raw.get("id") == mid:
                if "error" in raw:
                    raise RuntimeError(f"{method}: {raw['error']}")
                return raw.get("result", {})

    async def evaluate(self, expression):
        result = await self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"].get("text"))
        return result["result"].get("value")


# The painted box is the textarea's parent: Streamlit draws the rounded border
# and background there, so that -- not the textarea -- is the edge a viewer sees.
#
# The reference is the page's own text, NOT .block-container's padding edge.
# That distinction is the whole bug: .block-container's padding edge sits at
# 478px while every heading and chat message starts at 497px, one
# stVerticalBlockBorderWrapper inset further in. An earlier version of this check
# compared against the padding edge and reported a clean pass while the input sat
# 18px wide of the text at every single width.
ALIGNMENT_JS = """
(() => {
  const ta = document.querySelector('[data-testid="stChatInputTextArea"]');
  if (!ta) return {error: 'chat input not found'};
  const box = ta.parentElement.getBoundingClientRect();
  const bb = document.querySelector('.block-container');
  if (!bb) return {error: '.block-container not found'};
  const b = bb.getBoundingClientRect();
  const cs = getComputedStyle(bb);
  const de = document.documentElement;
  const side = document.querySelector('[data-testid="stSidebar"]');

  const edge = sel => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {left: Math.round(r.left), right: Math.round(r.right)};
  };

  // Prefer a real chat message once one exists (its column is what the user
  // reads against); fall back to the page heading on the landing page.
  const ref = edge('[data-testid="stChatMessage"]')
              || edge('.page-h1')
              || edge('.app-header');

  return {
    boxLeft: Math.round(box.left),
    boxRight: Math.round(box.right),
    blockPaddingEdge: Math.round(b.left + parseFloat(cs.paddingLeft)),
    refLeft: ref ? ref.left : null,
    refRight: ref ? ref.right : null,
    deltaLeft: ref ? Math.round(box.left - ref.left) : null,
    deltaRight: ref ? Math.round(box.right - ref.right) : null,
    blockPadding: Math.round(parseFloat(cs.paddingLeft)),
    sidebarWidth: side ? Math.round(side.getBoundingClientRect().width) : 0,
    horizontalScroll: de.scrollWidth > de.clientWidth + 1,
  };
})()
"""

async def _measure(s: Session, width: int, height: int) -> dict:
    await s.send(
        "Emulation.setDeviceMetricsOverride",
        width=width, height=height, deviceScaleFactor=1, mobile=(width < 500),
    )
    await asyncio.sleep(1.0)
    return await s.evaluate(ALIGNMENT_JS)



def _report(data: dict, label: str, note: str = "") -> str | None:
    """Return a failure string, or None when the measurement is acceptable."""
    if data.get("error"):
        return f"{label}: {data['error']}"
    dl, dr = data["deltaLeft"], data["deltaRight"]
    problems = []
    if dl is None or dr is None:
        problems.append("no reference element to align against")
    else:
        if abs(dl) > TOLERANCE_PX:
            problems.append(f"left edge off by {dl:+d}px")
        if abs(dr) > TOLERANCE_PX:
            problems.append(f"right edge off by {dr:+d}px")
    if data["horizontalScroll"]:
        problems.append("page scrolls horizontally")
    if not problems:
        print(f"  PASS: {label:<12} aligned within {TOLERANCE_PX}px "
              f"(dL={dl:+d} dR={dr:+d}, text column {data['refLeft']}.."
              f"{data['refRight']}, gutter {data['blockPadding']}px{note})")
        return None
    return f"{label}: " + "; ".join(problems)


async def _open_sweep(failures: list) -> bool:
    """Sweep every width with the sidebar open. Returns False if never ready."""
    raw = urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/json/list", timeout=10
    ).read()
    page = next(t for t in json.loads(raw) if t["type"] == "page")

    async with websockets.connect(
        page["webSocketDebuggerUrl"], max_size=40 * 1024 * 1024
    ) as ws:
        s = Session(ws)
        await s.send("Runtime.enable")
        await s.send("Page.enable")
        await s.send("Page.navigate", url=URL)

        ready = False
        for _ in range(90):
            await asyncio.sleep(1)
            try:
                ok = await s.evaluate(
                    "!!document.querySelector("
                    "'[data-testid=\"stChatInputTextArea\"]')"
                )
            except Exception:
                continue
            if ok:
                ready = True
                break
        if not ready:
            print("FAIL: the chat input never rendered")
            return False
        print(f"PASS: app hydrated at {URL}, chat input present")
        await asyncio.sleep(2)

        print()
        print(f"Sidebar open, {len(WIDTHS)} widths "
              f"({WIDTHS[-1][0]}-{WIDTHS[0][0]}px):")
        for width, height in WIDTHS:
            problem = _report(await _measure(s, width, height),
                             f"{width}x{height}")
            if problem:
                failures.append(problem)
                print(f"  FAIL: {problem}")

        await s.send("Emulation.clearDeviceMetricsOverride")
    return True


async def main() -> int:
    chrome = start_chrome()
    failures = []
    try:
        await _open_sweep(failures)
    finally:
        chrome.kill()
        _cleanup()

    print()
    if failures:
        print(f"RESULT: {len(failures)} alignment check(s) failed")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: the question input lines up with the text column at every "
          "width from 320px to 1920px")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
