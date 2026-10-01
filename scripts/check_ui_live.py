"""Drive headless Chrome over CDP to verify the Streamlit UI as painted.

Why this exists
---------------
Nothing else in the suite can tell whether the CSS actually applies. The UI
redesign shipped two bugs that every existing check passed straight over:

  * the answer card was a `<div>` opened in one st.markdown() call and closed in
    another. Streamlit renders each markdown block independently, so the opening
    tag became an empty div and the closing tag was discarded. The class was in
    the DOM and nothing was on screen.
  * the avatar-hiding rule targeted `stChatMessageContentAvatar`. Streamlit
    emits `stChatMessageAvatarUser` / `stChatMessageAvatarAssistant`. The rule
    parsed, matched nothing, and the avatars stayed visible.

AppTest inspects the element tree the Python code produces. It never sees CSS or
pixels. This drives a real browser and reads computed styles.

Usage
-----
    streamlit run src/app.py --server.port 8501 &
    python scripts/check_ui_live.py

Needs Chrome and `pip install websockets`. `websockets` is deliberately NOT in
requirements.txt: these are developer checks, and adding a dev-only dependency
to the app manifest would change the Render build. Set CHROME_PATH / UI_URL to
override the defaults. Exits non-zero on any failure, so it works as a gate.
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
PORT = int(os.environ.get("UI_CDP_PORT", "9333"))

WIDTHS = [(1440, 900), (1280, 800), (820, 1180), (390, 844)]

# Classes the redesign relies on. If Streamlit's own DOM shifts under a version
# bump, these go missing and the styling silently stops applying -- which is
# exactly how both bugs above survived.
EXPECTED_CLASSES = [
    # ours
    "app-header", "app-brand", "facts-pill", "page-h1", "page-sub",
    "page-intro", "suggest-label", "scheme-nav", "scheme-item",
    "disclaimer-strip", "footer-note", "footer-legal", "user-bubble",
    "source-row", "source-link", "loading-copy",
    # Streamlit's own, so a version bump that changes them is caught here
    "stVerticalBlockBorderWrapper", "stChatMessageAvatarUser",
    "stChatMessageAvatarAssistant", "stChatInput",
]


_PROFILE_DIR = None


def start_chrome() -> subprocess.Popen:
    global _PROFILE_DIR
    if not os.path.exists(CHROME):
        raise RuntimeError(
            f"Chrome not found at {CHROME}. Set CHROME_PATH to your binary."
        )
    _PROFILE_DIR = tempfile.mkdtemp(prefix="hdfc-ui-check-")
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
    # Wait for the debugging endpoint to answer before driving it.
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
        self.console_errors = []

    async def send(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps(
            {"id": mid, "method": method, "params": params}
        ))
        while True:
            raw = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=60))
            if raw.get("method") == "Runtime.consoleAPICalled":
                args = raw["params"].get("args", [])
                text = " ".join(str(a.get("value", a.get("description", "")))
                                for a in args)
                if raw["params"].get("type") == "error":
                    self.console_errors.append(text)
            if raw.get("method") == "Runtime.exceptionThrown":
                det = raw["params"]["exceptionDetails"]
                self.console_errors.append(
                    det.get("text", "") + " " +
                    str(det.get("exception", {}).get("description", ""))
                )
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


OVERFLOW_JS = """
(() => {
  const de = document.documentElement;
  const vw = de.clientWidth;
  const wide = [];
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && (r.right > vw + 1 || r.left < -1)) {
      wide.push({
        tag: el.tagName.toLowerCase(),
        cls: (el.className && el.className.baseVal !== undefined
              ? el.className.baseVal : String(el.className || '')).slice(0, 70),
        right: Math.round(r.right), left: Math.round(r.left),
      });
    }
  });
  return {
    viewport: vw,
    scrollWidth: de.scrollWidth,
    hasHorizontalScroll: de.scrollWidth > vw + 1,
    bodyScrollWidth: document.body.scrollWidth,
    offenders: wide.slice(0, 8),
    offenderCount: wide.length,
  };
})()
"""


async def main() -> int:
    chrome = start_chrome()
    failures = []
    try:
        raw = urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/json/list", timeout=10
        ).read()
        targets = json.loads(raw)
        page = next(
            t for t in targets if t["type"] == "page"
        )
        ws_url = page["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            s = Session(ws)
            await s.send("Runtime.enable")
            await s.send("Page.enable")
            await s.send("Page.navigate", url=URL)

            # Streamlit hydrates over a websocket. Poll for real content rather
            # than sleeping a fixed amount.
            ready = False
            for _ in range(60):
                await asyncio.sleep(1)
                try:
                    ok = await s.evaluate(
                        "!!document.querySelector('.app-header')"
                    )
                except Exception:
                    continue
                if ok:
                    ready = True
                    break
            if not ready:
                print("FAIL: the app never rendered .app-header")
                return 1
            print("PASS: app hydrated, .app-header present")

            # Let the welcome text and example cards settle.
            await asyncio.sleep(2)

            found = await s.evaluate(
                "(() => { const want = " + json.dumps(EXPECTED_CLASSES)
                + "; const html = document.documentElement.innerHTML;"
                " return want.filter(c => !html.includes(c)); })()"
            )
            if found:
                failures.append(f"missing classes: {found}")
                print(f"FAIL: classes absent from live DOM: {found}")
            else:
                print(f"PASS: all {len(EXPECTED_CLASSES)} redesign classes "
                      "present in the live DOM")

            for width, height in WIDTHS:
                await s.send(
                    "Emulation.setDeviceMetricsOverride",
                    width=width, height=height,
                    deviceScaleFactor=1, mobile=(width < 500),
                )
                await asyncio.sleep(1.2)
                data = await s.evaluate(OVERFLOW_JS)
                tag = f"{width}x{height}"
                if data["hasHorizontalScroll"]:
                    failures.append(f"{tag}: horizontal scroll "
                                    f"({data['scrollWidth']} > {data['viewport']})")
                    print(f"FAIL: {tag} scrolls horizontally "
                          f"({data['scrollWidth']}px content in "
                          f"{data['viewport']}px viewport)")
                    for off in data["offenders"]:
                        print(f"        offender: {off}")
                else:
                    print(f"PASS: {tag} no horizontal scroll "
                          f"(content {data['scrollWidth']}px)")

            await s.send("Emulation.clearDeviceMetricsOverride")
            await asyncio.sleep(1)

            if s.console_errors:
                uniq = sorted(set(s.console_errors))[:5]
                failures.append(f"console errors: {uniq}")
                print("FAIL: console errors:")
                for err in uniq:
                    print(f"        {err[:160]}")
            else:
                print("PASS: no uncaught console errors")

    finally:
        chrome.kill()
        _cleanup()

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) failed")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: all live-page checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))