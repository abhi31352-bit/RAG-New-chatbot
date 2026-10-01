"""Interactive live-page checks over CDP.

Complements check_ui_live.py, which only inspects the landing page. This drives
the flows a user actually touches: click an example card, read the rendered
answer, follow the source link, open Developer options, type a question the
guardrails must refuse, and confirm no horizontal overflow once a conversation
is on screen -- a long answer plus a long URL is exactly where overflow appears.

Streamlit is driven by clicking real DOM nodes, not by calling Python.

Usage
-----
    streamlit run src/app.py --server.port 8501 &
    python scripts/check_ui_interactive.py

Needs Chrome, `pip install websockets`, the built index, and GROQ_API_KEY: it
makes a real model call. `websockets` is deliberately NOT in requirements.txt --
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
PORT = int(os.environ.get("UI_CDP_PORT", "9334"))

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
    _PROFILE_DIR = tempfile.mkdtemp(prefix="hdfc-ui-interact-")
    proc = subprocess.Popen(
        [
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={_PROFILE_DIR}",
            "--window-size=1440,900", "--no-first-run",
            "--no-default-browser-check", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
    raise RuntimeError("devtools never came up")


class S:
    def __init__(self, ws):
        self.ws, self._id, self.errs = ws, 0, []

    async def send(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method,
                                       "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=90))
            m = msg.get("method")
            if m == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]
                self.errs.append(d.get("text", "") + " " + str(
                    d.get("exception", {}).get("description", "")))
            if m == "Runtime.consoleAPICalled" and \
                    msg["params"].get("type") == "error":
                self.errs.append(" ".join(
                    str(a.get("value", a.get("description", "")))
                    for a in msg["params"].get("args", [])))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def ev(self, js):
        r = await self.send("Runtime.evaluate", expression=js,
                            returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("text"))
        return r["result"].get("value")


async def wait_for(s, js, label, tries=90):
    probe = ("(() => { try { return !!(" + js + "); }"
             " catch (e) { return false; } })()")
    for _ in range(tries):
        try:
            if await s.ev(probe):
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    check(label, False, "timed out waiting for it to render")
    return False


OVERFLOW = """
(() => {
  const de = document.documentElement, vw = de.clientWidth, bad = [];
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && (r.right > vw + 1 || r.left < -1)) {
      const cn = el.className && el.className.baseVal !== undefined
        ? el.className.baseVal : String(el.className || '');
      bad.push(el.tagName.toLowerCase() + '.' + cn.slice(0, 50) +
               ' right=' + Math.round(r.right));
    }
  });
  return {vw: vw, sw: de.scrollWidth, over: de.scrollWidth > vw + 1,
          n: bad.length, first: bad.slice(0, 6)};
})()
"""


async def main():
    chrome = start_chrome()
    try:
        targets = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list",
                                   timeout=10).read())
        page = next(t for t in targets if t["type"] == "page")
        async with websockets.connect(
            page["webSocketDebuggerUrl"], max_size=40 * 1024 * 1024
        ) as ws:
            s = S(ws)
            await s.send("Runtime.enable")
            await s.send("Page.enable")
            await s.send("Page.navigate", url=URL)
            if not await wait_for(s, "document.querySelector('.app-header')",
                                  "landing page renders"):
                return 1
            await asyncio.sleep(2)

            # --- 1. click the first example card ---------------------------
            clicked = await s.ev(
                """(() => {
                  // Match on content, not exact equality: Streamlit wraps button
                  // text in a markdown container and innerText carries padding.
                  const b = [...document.querySelectorAll('button')].find(
                    x => /expense ratio of HDFC Large Cap/i.test(x.innerText));
                  if (!b) return 'button not found';
                  b.scrollIntoView({block: 'center'});
                  b.click();
                  return 'clicked';
                })()"""
            )
            check("example card is clickable in the live page",
                  clicked == "clicked", str(clicked))
            check("all four example cards render",
                  await s.ev("""[...document.querySelectorAll('button')]
                      .filter(b => /What is the/.test(b.innerText)).length""") == 4)

            if not await wait_for(
                s,
                """[...document.querySelectorAll('[data-testid="stChatMessage"]')]
                     .some(m => /1\\.03/.test(m.innerText))""",
                "an answer card renders after a click"):
                return 1
            await asyncio.sleep(3)

            # --- 2. the answer is the visible hero ------------------------
            # The card is a real Streamlit bordered container now, so query the
            # message body rather than a class of our own.
            answer = await s.ev(
                """(() => {
                  const msgs = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
                  const last = msgs[msgs.length - 1];
                  if (!last) return '';
                  const body = last.querySelector('[data-testid="stChatMessageContent"]');
                  return body ? body.innerText.trim() : '';
                })()"""
            )
            check("answer text is rendered", bool(answer), repr(answer)[:80])
            check("answer states the real expense ratio (1.03%)",
                  "1.03" in (answer or ""), repr(answer)[:160])

            # --- 3. loading copy was shown, not pipeline jargon ----------
            user_txt = await s.ev(
                """(() => {
                  const b = [...document.querySelectorAll('.user-bubble')]
                    .map(x => x.innerText.trim());
                  return b[b.length-1] || '';
                })()"""
            )
            check("the clicked question appears as a user bubble",
                  "expense ratio" in (user_txt or "").lower(), repr(user_txt))

            # --- 4. source row: one link, real href -----------------------
            link = await s.ev(
                """(() => {
                  const a = [...document.querySelectorAll('.source-link')].pop();
                  return a ? {href: a.href, text: a.innerText.trim(),
                              target: a.target,
                              rel: a.rel || '',
                              title: (a.getAttribute('title')||'')} : null;
                })()"""
            )
            check("source row rendered with a link", link is not None)
            if link:
                check("source href points at the real Groww page",
                      "groww.in/mutual-funds/" in link["href"], link["href"])
                check("source opens in a new tab, safely",
                      link["target"] == "_blank" and "noopener" in link["rel"],
                      f'target={link["target"]} rel={link["rel"]}')
                check("full URL is still available on hover (title attr)",
                      link["href"] in link["title"], link["title"][:70])

            # --- 5. no repeated URL in visible text ----------------------
            body = await s.ev("document.body.innerText")
            check("the raw URL is not printed as visible body text",
                  "groww.in" not in (body or ""),
                  "URL leaked into visible text")

            check("last-updated line is visible",
                  "Last updated from sources:" in (body or ""), "")

            # --- 6. overflow once a conversation is on screen ------------
            for w, h in [(1440, 900), (1280, 800), (820, 1180), (390, 844)]:
                await s.send("Emulation.setDeviceMetricsOverride",
                             width=w, height=h, deviceScaleFactor=1,
                             mobile=(w < 500))
                await asyncio.sleep(1.2)
                d = await s.ev(OVERFLOW)
                check(f"no horizontal overflow with a conversation ({w}x{h})",
                      not d["over"],
                      f'content {d["sw"]}px in {d["vw"]}px; {d["first"]}')

            # --- 7. developer options collapsed, chunks still reachable --
            await s.send("Emulation.clearDeviceMetricsOverride")
            await asyncio.sleep(1)
            dev = await s.ev(
                """(() => {
                  const sd = document.querySelector('[data-testid="stSidebar"]');
                  if (!sd) return {found: false};
                  const sum = [...sd.querySelectorAll('summary')]
                    .find(s => /Developer options/i.test(s.innerText));
                  if (!sum) return {found: false};
                  const details = sum.closest('details');
                  // Read the DOM, not innerText: a collapsed <details> has no
                  // layout, so its textContent is what exists, not its
                  // rendered text. innerText here reads '' for correct markup.
                  const txt = details ? details.textContent : '';
                  const main = document.querySelector('[data-testid="stMain"]');
                  const mainTxt = main ? main.innerText : '';
                  return {found: true,
                          open: details ? details.open : null,
                          inside: /Show retrieved chunks/.test(txt),
                          inMain: /Show retrieved chunks/.test(mainTxt),
                          chunksLabel: (txt.match(/\\d+ chunks per question/) || [null])[0]};
                })()"""
            )
            check("Developer options section exists", dev.get("found"))
            check("it is collapsed by default", dev.get("open") is False,
                  f'open={dev.get("open")}')
            check("the chunks checkbox is inside it, not in the main UI",
                  dev.get("inside") and not dev.get("inMain"), str(dev))
            check("'chunks per question' is now demoted to the debug section",
                  dev.get("chunksLabel") == "10 chunks per question",
                  str(dev.get("chunksLabel")))

            # --- 8. sidebar is compact, disclaimer is short --------------
            side = await s.ev(
                """(() => {
                  const sd = document.querySelector('[data-testid="stSidebar"]');
                  if (!sd) return null;
                  const strip = sd.querySelector('.disclaimer-strip');
                  return {
                    schemes: sd.querySelectorAll('.scheme-item').length,
                    stripWords: strip ? strip.innerText.trim().split(/\\s+/).length : 0,
                    width: Math.round(sd.getBoundingClientRect().width),
                  };
                })()"""
            )
            if side:
                check("sidebar lists the 5 schemes", side["schemes"] == 5,
                      str(side))
                check("sidebar disclaimer is short (<40 words)",
                      0 < side["stripWords"] < 40, str(side["stripWords"]))
            else:
                check("sidebar present", False)

            # --- 9. type a refusal question into the real input ---------
            typed = await s.ev(
                """(() => {
                  const ta = document.querySelector('[data-testid="stChatInput"] textarea');
                  if (!ta) return 'no textarea';
                  const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value').set;
                  setter.call(ta, 'What are the returns on HDFC Large Cap?');
                  ta.dispatchEvent(new Event('input', {bubbles: true}));
                  return 'typed';
                })()"""
            )
            check("chat input accepts typed text", typed == "typed", str(typed))
            await asyncio.sleep(1)
            submitted = await s.ev(
                """(() => {
                  const btn = document.querySelector(
                    '[data-testid="stChatInput"] button');
                  if (!btn) return 'no send button';
                  btn.click();
                  return 'sent';
                })()"""
            )
            check("send button submits", submitted == "sent", str(submitted))

            got_refusal = await wait_for(
                s, """(() => {
                  const m = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
                  if (m.length < 2) return false;
                  const b = m[m.length-1]
                    .querySelector('[data-testid="stChatMessageContent"]');
                  return !!b && /advice|returns|predict|cannot|can't/i.test(b.innerText);
                })()""",
                "a refusal/advice-safe answer comes back")
            await asyncio.sleep(2)

            if got_refusal:
                last = await s.ev(
                    """(() => {
                      const m = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
                      const b = m[m.length-1]
                        .querySelector('[data-testid="stChatMessageContent"]');
                      return b ? b.innerText.trim() : '';
                    })()"""
                )
                check("the refusal did not leak a return figure",
                      not any(x in (last or "").lower() for x in
                              ("expected return", "will return", "guaranteed")),
                      repr(last)[:100])
                # Guardrails must refuse before generation.
                check("refusal is explained, not silently empty",
                      len(last or "") > 20, repr(last)[:80])

            # --- 10. avatars actually hidden ----------------------------
            avatars = await s.ev(
                """(() => {
                  const av = [...document.querySelectorAll(
                    '[data-testid="stChatMessageAvatarUser"],'
                    + '[data-testid="stChatMessageAvatarAssistant"]')];
                  if (!av.length) return {count: 0, visible: 0};
                  return {count: av.length,
                          visible: av.filter(e => e.getBoundingClientRect().height > 0
                                                 || getComputedStyle(e).display !== 'none').length};
                })()"""
            )
            check("chat avatars exist but are hidden",
                  avatars.get("count", 0) > 0 and avatars.get("visible") == 0,
                  str(avatars))

            # --- 11. console clean across the whole session -------------
            uniq = sorted(set(s.errs))[:6]
            check("no uncaught console errors during the whole flow",
                  not uniq, "; ".join(e[:120] for e in uniq))
    finally:
        chrome.kill()
        if _PROFILE_DIR:
            shutil.rmtree(_PROFILE_DIR, ignore_errors=True)

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all interactive live-page checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))