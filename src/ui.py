"""Presentation layer for the Streamlit UI.

Everything here is CSS and HTML. It imports nothing from the RAG pipeline --
it is handed an `Answer` and a list of chunks and told how to draw them. That
separation is the point: if this module needed to reach into retrieval to render,
a styling change could quietly break the pipeline.

The visual reference
--------------------
`ui design/stitch-desgin.html` is the Google Stitch mockup of this product: a
light-surface fintech console. Its design tokens are transcribed below rather
than reinvented, so every value here can be traced to a line of that file:

  * emerald 700 `#047857` is the single action colour; 600 `#059669` the accent,
    800 `#065F46` the hover/emphasis, 50 `#ECFDF5` the tint, 200 `#A7F3D0` its
    border. It replaces the earlier ad-hoc green, which was never emerald.
  * slate carries all the text: 900 `#0F172A` headings, 600 `#475569` body,
    500 `#64748B` secondary, 400 `#94A3B8` faint.
  * `surface` is the neutral ramp: 50 `#F8FAFC` is the page, 100 `#F1F5F9` the
    inset fill, 200 `#E2E8F0` every border, 300 `#CBD5E1` the input edge.
  * Inter for text, JetBrains Mono for tags, timestamps and provenance. Both are
    requested from Google Fonts; the stacks below them are the system fonts, so
    a blocked or offline font request degrades to the platform face rather than
    to nothing.
  * three named shadows: `subtle`, `card` and `elevation`. Only the first two
    are used -- `elevation` is for the mockup's floating panels, and using it on
    a chat answer would read as a dev dashboard.
  * 6px pill scrollbar with a `#CBD5E1` thumb, carried over verbatim.

Layout follows the mockup's grid: a 56px top bar, a 256px sidebar, and a
centred 768px (`max-w-3xl`) message canvas and composer sharing one column. The
mockup's column is `max-w-3xl` inside a `px-6` canvas; here the shared `--gutter`
is that 24px inset, so the painted column lands at 720px rather than 768. That
28px is the price of keeping ONE gutter variable feeding both the content column
and the input column, which is what stops the question box drifting away from
the answers it is answering.

What is deliberately NOT ported
-------------------------------
The mockup is a static showcase, and three of its elements exist only to
demonstrate states that a real page does not have. Re-creating them here would
mean shipping controls that do nothing:

  * the "Preview: Landing / User Asked / Generating / ..." switcher -- the
    mockup's own scaffolding for reviewing six states in one screenshot.
  * Share / Bookmark / Feedback / GitHub in the top bar -- icons with no
    handler behind them.
  * the "Verified AMC Cache / 100% Match / Synced Oct 2026" sidebar pill -- a
    provenance claim this app cannot make about its own index.

Two more are out of reach without touching the pipeline, so they stay out:

  * the answer card's big primary stat and its mono spec grid (Benchmark Index,
    Minimum SIP, Exit Load, Riskometer Level). Reproducing those means reading
    fields out of the model's prose, which is answer-generation logic.
  * the compliance paragraph under the composer. In the mockup it sits in a
    fixed footer. Here the question bar is in-flow and sticky, so growing it by
    a paragraph's height pushes the gap under it past what the layout gates
    allow. The disclaimer is still on screen throughout a conversation, via the
    header pill and the sidebar strip.
"""
from __future__ import annotations

import html
import re
from typing import List, Optional, Sequence

import streamlit as st

# --- Design tokens, transcribed from the Stitch mockup -----------------------

# Emerald is the action and accent colour: one fill, reserved for the single
# primary action on the page (the send button) and for anything that is meant to
# read as a verified link or state. Nothing else wears it as a background.
GREEN = "#047857"          # emerald-700 -- primary action
GREEN_DARK = "#065F46"     # emerald-800 -- hover, emphasis inside prose
GREEN_MID = "#059669"      # emerald-600 -- status dots, glyphs, icon strokes
GREEN_TINT = "#ECFDF5"     # emerald-50  -- pills, active nav, link chips
GREEN_EDGE = "#A7F3D0"     # emerald-200 -- borders on tinted surfaces
GREEN_RING = "rgba(5, 150, 105, 0.18)"

INK = "#0F172A"            # slate-900 -- headings
BODY = "#334155"           # slate-700 -- answer prose
MUTED = "#475569"          # slate-600 -- secondary text
FAINT = "#94A3B8"          # slate-400 -- labels, provenance
BORDER = "#E2E8F0"         # surface-200 -- every hairline
EDGE = "#CBD5E1"           # surface-300 -- input border
SURFACE = "#F1F5F9"        # surface-100 -- inset fills
CARD = "#FFFFFF"
PAGE = "#F8FAFC"           # surface-50 -- the page itself
SLATE_DARK = "#0F172A"     # slate-900 -- the user bubble's fill

DANGER = "#E11D48"         # rose-600 -- scope refusals
DANGER_DARK = "#9F1239"    # rose-800
DANGER_TINT = "#FFF1F2"    # rose-50
DANGER_EDGE = "#FECDD3"    # rose-200
WARN_TINT = "#FFFAEB"
WARN_INK = "#B54708"
WARN_EDGE = "#FDE68A"

# Typefaces. The webfonts are requested by inject_css(); these stacks name the
# platform fallback so a blocked request changes the face, not the layout.
SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# The mockup's three shadows, by name.
SHADOW_SUBTLE = "0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.03)"
SHADOW_CARD = "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.04)"

FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2'
    '?family=Inter:wght@400;500;600;700'
    '&amp;family=JetBrains+Mono:wght@400;500;600&amp;display=swap">'
)

# Sidebar shows the five scheme names; each gets a short glyph so the list scans
# faster than five identical bullets. These are text, not icon assets -- no
# external files, no licensing questions, and they inherit font colour. The
# glyph shapes and the mono tag column both come from the mockup's sidebar.
SCHEME_GLYPHS = {
    "Large Cap": "◆",
    "Flexi Cap": "◇",
    "ELSS Tax Saver": "■",
    "Small Cap": "▲",
    "Balanced Advantage": "●",
}

# The mockup tags each row with a plan or category. Every scheme in this corpus
# is Direct-Growth, so three rows legitimately repeat -- that is information,
# not a copy-paste slip: it is the reader's confirmation that the facts on offer
# are all Direct-Growth NAVs.
SCHEME_TAGS = {
    "Large Cap": "Direct-G",
    "Flexi Cap": "Direct-G",
    "ELSS Tax Saver": "80C",
    "Small Cap": "Direct-G",
    "Balanced Advantage": "Hybrid",
}

SIDEBAR_SCHEMES = (
    "Large Cap",
    "Flexi Cap",
    "ELSS Tax Saver",
    "Small Cap",
    "Balanced Advantage",
)


def inject_css() -> None:
    """Install the stylesheet. Called once per script run, before any widgets."""
    st.markdown(FONT_LINK, unsafe_allow_html=True)
    st.markdown(f"""<style>
:root {{
  --green: {GREEN};
  --green-dark: {GREEN_DARK};
  --green-mid: {GREEN_MID};
  --green-tint: {GREEN_TINT};
  --green-edge: {GREEN_EDGE};
  --green-ring: {GREEN_RING};
  --ink: {INK};
  --body: {BODY};
  --muted: {MUTED};
  --faint: {FAINT};
  --border: {BORDER};
  --edge: {EDGE};
  --surface: {SURFACE};
  --card: {CARD};
  --page: {PAGE};
  --danger: {DANGER};
  --danger-dark: {DANGER_DARK};
  --danger-tint: {DANGER_TINT};
  --danger-edge: {DANGER_EDGE};
  --warn-tint: {WARN_TINT};
  --warn-ink: {WARN_INK};
  --shadow-subtle: {SHADOW_SUBTLE};
  --shadow-card: {SHADOW_CARD};
  --sans: {SANS};
  --mono: {MONO};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--page);
  font-family: var(--sans);
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

/* The mockup's 6px pill scrollbar, carried over verbatim. Streamlit's default
   is wide and square and it shows up along the transcript on every long answer. */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {EDGE}; border-radius: 9999px; }}
::-webkit-scrollbar-thumb:hover {{ background: #94A3B8; }}

/* --- Typography scale ----------------------------------------------------- */
/* The design's typeface, on everything.

   Streamlit 1.38 puts Source Sans Pro on [data-testid="stAppViewContainer"]
   through an emotion class, which wins on specificity and left almost the whole
   page in Streamlit's face: only the elements we style by hand -- the
   suggestion cards and the textarea -- came out in Inter, so the page mixed two
   typefaces with no rule saying so.

   :where() gives this declaration ZERO specificity, and !important is what
   carries it past Streamlit's own rules. That combination is deliberate: it is
   the only way to say "the typeface is the design's, everywhere, and no author
   rule of ours or Streamlit's outranks it" without a !important on every
   element. The mono faces below are !important for the same reason and do win,
   because among two important declarations the more specific one applies --
   and :where() is the least specific thing in CSS. */
:where(html, body, .stApp, .stApp *, [data-testid="stAppViewContainer"],
       [data-testid="stAppViewContainer"] *) {{
  font-family: var(--sans) !important;
}}
/* Code stays mono. Folding it into the sans sweep would have been a regression
   in the other direction. */
:where(code, pre, pre *, .stApp code, .stApp pre, .stApp pre *) {{
  font-family: var(--mono) !important;
}}
.stApp h1 {{ font-size: 1.875rem !important; line-height: 1.2 !important;
  letter-spacing: -0.025em; font-weight: 700 !important; color: var(--ink); }}
.stApp h2, .stApp h3 {{ letter-spacing: -0.02em; color: var(--ink); }}
/* Everything Streamlit marks as a caption is provenance in this design: scheme
   subtitles, chunk counts, the disclosure date. Mono, faint, small. */
.stApp .stCaptionContainer p, .stApp [data-testid="stCaptionContainer"] p,
.stApp small {{ color: var(--faint) !important; font-size: 0.6875rem;
  font-family: var(--mono) !important; }}
.stApp a {{ color: var(--green); text-decoration: none;
  overflow-wrap: anywhere; word-break: break-word; }}
.stApp a:hover {{ text-decoration: underline; }}
/* Prose is slate-700, not slate-900: the answer is a reading surface, and the
   headings already carry the ink. */
.stApp p, .stApp li {{ color: var(--body); line-height: 1.62;
  overflow-wrap: anywhere; word-break: break-word; }}

hr {{ border-color: var(--border) !important; margin: 1.25rem 0 !important; }}

/* --- Layout: centred column, never wider than needed --------------------- */
/* --gutter is the shared horizontal inset. .block-container and the chat input
   both read it, which is the only reason those two stay on the same vertical
   lines. It is the mockup's `px-6` at desktop; the two narrow steps below are
   Streamlit's own responsive insets, which the mockup does not specify. */
:root {{
  --gutter: 24px;
}}
@media (max-width: 768px) {{ :root {{ --gutter: 16px; }} }}
@media (max-width: 480px) {{ :root {{ --gutter: 12px; }} }}

/* The mockup's canvas is `px-6` around a `max-w-3xl` column. The cap is the
   same 768px, applied to .block-container and to the input's column alike.

   Bottom clearance for the sticky input bar.
   [data-testid="stBottom"] is in-flow and sticky, so it occupies its own height
   at the end of the scroll content rather than floating over it. Scrolled fully
   down, the block's content bottom therefore lands exactly on the top of the
   input. That in-flow reservation is the real reason answers are never hidden:
   it is what Streamlit's scroll-to-bottom measures against, and it is why
   replacing it with a viewport-pinned input (as an earlier version did) broke
   both the clearance and the auto-scroll at once.
   This padding is the small breathing gap ON TOP of that. */
.block-container {{
  max-width: 768px !important;
  padding-top: 1.5rem !important;
  padding-bottom: 1.75rem !important;
  padding-left: var(--gutter) !important;
  padding-right: var(--gutter) !important;
  margin: 0 auto !important;
}}

/* The suggestion row. Streamlit lays columns out in a fixed-width flex row
   that scrolls sideways on a phone; wrapping to full-width blocks is what keeps
   the "no horizontal scrolling" promise true at 390px.

   The column's testid in 1.38 is `column`, not `stColumn`: counted in the live
   DOM, `[data-testid="stColumn"]` matches 0 elements while
   `[data-testid="column"]` matches 4. Rules written against the older name match
   nothing at all, which had quietly killed this rule and the two grid rules
   below. `stColumn` is kept alongside `column` so the layout survives a rename
   in either direction. */
[data-testid="stHorizontalBlock"] {{ gap: 0.75rem; }}
[data-testid="stHorizontalBlock"] > [data-testid="column"],
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ min-width: 0; }}

/* --- Header --------------------------------------------------------------- */
/* The mockup's `h-14` top bar: 56px, translucent white over a blur, one
   hairline underneath. In Streamlit this sits inside the content column rather
   than spanning the viewport, so it reads as the same bar scoped to the
   conversation instead of to the window. */
.app-header {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; min-height: 3.5rem;
  padding: 0 0.25rem 0 0; margin-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.95);
  -webkit-backdrop-filter: blur(12px);
  backdrop-filter: blur(12px);
}}
.app-header-left {{
  display: flex; align-items: center; gap: 0.875rem; min-width: 0;
}}
.app-brand {{
  font-size: 0.875rem; font-weight: 600; color: var(--ink);
  letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 0.5rem; white-space: nowrap;
}}
/* The mockup's ping: a solid emerald-600 dot inside an expanding emerald-400
   ring. It is the only motion in the header, and it is what makes the bar read
   as "live" rather than as a static wordmark.

   The ring is a box-shadow rather than a second, overlapped element on purpose.
   Two elements would need one taken out of the flow to sit on top of the other,
   and this module has a standing rule that nothing may be positioned that way
   -- a box positioned against its nearest positioned ancestor stops tracking
   the column it lives in, which is exactly how the question box came to sit
   398px away from the answers once already. A shadow pulse needs no second box
   at all, and it animates without touching layout. */
.brand-dot {{
  display: inline-block; flex-shrink: 0;
  width: 10px; height: 10px; border-radius: 999px; background: var(--green-mid);
  animation: brand-ping 1.8s cubic-bezier(0, 0, 0.2, 1) infinite;
}}
@keyframes brand-ping {{
  0% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.75); }}
  70% {{ box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
}}
.app-sep {{ font-size: 0.75rem; color: var(--faint); }}
.app-tagline {{
  font-size: 0.75rem; color: var(--muted); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}}
.facts-pill {{
  background: var(--green-tint); color: var(--green);
  border: 1px solid rgba(5, 150, 105, 0.28);
  border-radius: 999px; padding: 0.125rem 0.5rem;
  font-size: 0.6875rem; font-weight: 500; white-space: nowrap; flex-shrink: 0;
}}

/* --- Sidebar -------------------------------------------------------------- */
/* The mockup's `w-64` on a `slate-50/80` field. 256px, not the 288px it uses at
   `lg`: 288 would take a quarter of a 1440px screen, and every measured budget
   for this column tops out at 280px. The four navigation rows and the two-line
   disclaimer fit in 256 comfortably. The main column reflows on its own when
   this changes, so nothing else has to be pinned. */
[data-testid="stSidebar"] {{
  background: var(--page);
  border-right: 1px solid var(--border);
  width: 256px !important;
  min-width: 256px !important;
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stSidebar"] .stMarkdown p {{ font-size: 0.75rem; }}
.sidebar-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.5rem; margin-bottom: 0.75rem; padding: 0 0.5rem;
}}
.sidebar-title {{
  font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.075em;
  color: var(--faint); font-weight: 700; margin: 0;
}}
.sidebar-count {{
  font-size: 0.625rem; font-weight: 600; color: var(--muted);
  background: rgba(226, 232, 240, 0.7);
  border-radius: 999px; padding: 0.125rem 0.375rem; white-space: nowrap;
}}
.scheme-nav {{ display: flex; flex-direction: column; gap: 0.25rem; }}
.scheme-item {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.625rem; padding: 0.5rem 0.75rem; border-radius: 8px;
  font-size: 0.75rem; color: var(--ink); font-weight: 500;
  border: 1px solid transparent;
  transition: background-color .14s ease, border-color .14s ease;
}}
.scheme-item:hover {{
  background: rgba(226, 232, 240, 0.5); color: var(--ink);
}}
.scheme-glyph {{ color: var(--green-mid); font-size: 0.625rem; width: 0.625rem; }}
.scheme-tag {{
  font-family: var(--mono) !important; font-size: 0.625rem; color: var(--faint);
  white-space: nowrap;
}}
/* The sidebar's pinned disclaimer, in the mockup's two-line form. Short by
   design -- the long form is the compliance banner on the landing page, and
   repeating a 60-word paragraph here made the sidebar the most visually
   dominant column on the page. */
.disclaimer-strip {{
  font-size: 0.6875rem; color: var(--faint); line-height: 1.5;
  border-top: 1px solid var(--border); padding: 1rem; margin: 0.5rem 0 0 0;
  background: rgba(255, 255, 255, 0.7);
}}
.disclaimer-strip strong {{
  display: block; color: var(--ink); font-weight: 600; margin-bottom: 0.125rem;
}}

/* Developer options, drawn as the mockup's accordion: hairline box, a slate
   header strip, mono values inside. Streamlit's own summary element carries
   the chevron, so the arrow needs no markup of ours. */
[data-testid="stExpander"] {{
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--card); margin-top: 0.75rem;
}}
details summary {{
  font-size: 0.6875rem; color: var(--muted); background: var(--surface);
  padding: 0.5rem 0.75rem;
}}

/* --- Cards --------------------------------------------------------------- */
/* Generic surface: the mockup's 1px border + `card` shadow. Deliberately NOT
   `elevation` -- that reads as a dev dashboard. */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.15rem;
  box-shadow: var(--shadow-subtle);
}}

/* --- Chat messages -------------------------------------------------------- */
/* Turn rhythm. Streamlit's own [data-testid="stChatMessage"] carries 16px of
   padding top AND bottom, plus 8px of margin on top: 40px of chrome per turn,
   24px of it between a question and its own answer. That is where the
   conversation's looseness came from, and it is dropped here.

   What replaces it is asymmetric on purpose, because chat reads in pairs: a
   question sits close to the answer it produced, and a clear break falls after
   the answer, before the next question. */
[data-testid="stChatMessage"] {{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  margin-bottom: 0 !important;
  display: flex; align-items: flex-start; gap: 0.75rem;
}}
/* Streamlit marks no role on the message, so the .user-bubble rendered inside
   it is what identifies a user turn, via :has(). */
[data-testid="stChatMessage"]:has(.user-bubble) {{
  flex-direction: row-reverse;
  margin-bottom: 0.5rem !important;
}}
[data-testid="stChatMessage"]:not(:has(.user-bubble)) {{
  margin-bottom: 1.25rem !important;
}}

/* Avatars. The mockup gives every turn a 32px circle -- emerald for the
   assistant, slate for the reader -- and they are the reason a turn reads as a
   conversation rather than as a log. These were hidden outright; hiding them
   bought alignment that the row-reverse above already provides for free, so they
   are back, wearing the mockup's circles.

   Streamlit renders a Material glyph inside an emotion-classed div. Verified in
   the live DOM: the avatar wrappers are stChatMessageAvatarUser /
   stChatMessageAvatarAssistant and they are SIBLINGS of stChatMessageContent,
   both children of stChatMessage. That is why row-reverse moves the reader's
   avatar to the right without touching either element's order. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{
  display: flex !important;
  align-items: center; justify-content: center;
  flex-shrink: 0; width: 2rem; height: 2rem; min-width: 2rem;
  border-radius: 999px; margin-top: 0.125rem;
}}
[data-testid="stChatMessageAvatarAssistant"] {{
  background: var(--green); color: #FFFFFF;
  box-shadow: var(--shadow-subtle);
}}
[data-testid="stChatMessageAvatarAssistant"] svg {{
  width: 1.25rem; height: 1.25rem;
}}
[data-testid="stChatMessageAvatarUser"] {{ background: #E2E8F0; color: #334155; }}
[data-testid="stChatMessageAvatarUser"] svg {{ width: 1.125rem; height: 1.125rem; }}

/* The older name is kept as a no-op safety net against a rename. */
[data-testid="stChatMessageContentAvatar"] {{ display: flex !important; }}
[data-testid="stChatMessageContent"] {{ gap: 0; align-items: flex-start; }}
[data-testid="stChatMessageContent"] {{
  font-size: 0.875rem; line-height: 1.65; overflow-wrap: anywhere;
  word-break: break-word; min-width: 0; flex: 1;
}}
/* The citation row and the status notice are siblings of the answer card inside
   the message body. Streamlit stacks those blocks with its own margins; give
   them a deliberate, smaller gap instead. */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"]
  .source-row {{ margin-top: 0; }}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"]
  .source-date {{ margin-top: 0.15rem; }}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] .notice {{
  margin-top: 0.5rem;
}}

/* User message. The mockup's bubble is slate-900 on white with the tail corner
   squared off (`rounded-2xl rounded-tr-sm`), which is the whole reason the
   bubble looks like a bubble: the corner nearest the assistant's reply is cut
   away. It was previously a green tint, which made the question read as a
   positive signal rather than as something the reader said. */
.user-bubble {{
  background: {SLATE_DARK};
  color: #FFFFFF;
  border-radius: 16px;
  border-bottom-right-radius: 4px;
  padding: 0.75rem 1rem;
  max-width: 512px;
  display: inline-block;
  margin-left: auto;
  font-size: 0.875rem; font-weight: 500; line-height: 1.5;
  overflow-wrap: anywhere; word-break: break-word;
}}

/* Assistant answer: a real bordered container (st.container(border=True)), so
   the card exists in Streamlit's own DOM rather than in a hand-written div.
   This is the element the user came to read, so it gets the most space and the
   most generous padding: the mockup's `rounded-2xl p-5 shadow-card`. */
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--card);
  border-color: var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow-card);
  padding: 1.25rem;
  overflow-wrap: anywhere;
}}
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] p {{
  margin: 0;
  font-size: 0.875rem; line-height: 1.7; color: var(--body);
}}
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"]
  strong {{ color: var(--ink); font-weight: 600; }}
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"]
  ul, ol {{ color: var(--body); }}

/* --- Source row ----------------------------------------------------------- */
/* The mockup's citation box: a hairline above, an emerald-tinted link carrying
   the source's own name, then the verified pill and the mono disclosure date.
   The full URL is never printed inline -- a 70-character address is the widest
   element on the page and was the direct cause of horizontal overflow. It stays
   in href= and in title=, so it is one hover (or one screen reader) away. */
.source-row {{
  display: flex; flex-wrap: wrap; align-items: center;
  gap: 0.4rem 0.625rem; margin-top: 0.15rem; min-width: 0;
  padding-top: 0.75rem; border-top: 1px solid var(--surface);
}}
.source-link {{
  display: inline-flex; align-items: center; gap: 0.375rem;
  font-size: 0.6875rem; color: var(--green); font-weight: 500;
  background: rgba(236, 253, 245, 0.7);
  border-radius: 6px; padding: 0.25rem 0.625rem;
  transition: background-color .14s ease;
}}
.source-link:hover {{ background: rgba(167, 243, 208, 0.6); }}
.source-label {{ color: var(--green); }}
/* The mockup's "Verified Fact" badge. Emitted only when the answer carries a
   source address, which is the condition the pipeline's validator sets -- so
   the claim it makes is the app's own grounding claim, not a new one. */
.verified-pill {{
  display: inline-flex; align-items: center; gap: 0.25rem;
  padding: 0.125rem 0.5rem; border-radius: 4px;
  background: var(--green-tint); color: var(--green);
  border: 1px solid var(--green-edge);
  font-size: 0.6875rem; font-weight: 500; white-space: nowrap;
}}
.verified-pill::before {{
  content: ""; width: 8px; height: 5px; flex-shrink: 0;
  border-left: 1.6px solid var(--green-mid);
  border-bottom: 1.6px solid var(--green-mid);
  transform: rotate(-45deg) translate(1px, -1px);
}}
.source-meta, .source-date {{
  font-family: var(--mono) !important; font-size: 0.625rem; color: var(--faint);
  white-space: nowrap;
}}
.source-date {{ margin-top: 0; }}

/* --- Buttons -------------------------------------------------------------- */
/* Suggestion cards. The mockup's card is `p-4 rounded-xl` over a `shadow-xs`,
   lifting to `shadow-card` and taking an emerald-60 border and a faint emerald
   wash on hover.

   Green fill stays reserved for the one primary action (the send button) so it
   keeps its meaning.

   These were laid out four across by Streamlit: 188px per column for a 40-44
   character question, so every one wrapped onto 4-5 lines. Forcing a 2x2 grid
   above 768px gives each one ~354px, which is two lines. Below 768px the
   existing media query stacks them full width. */
@media (min-width: 769px) {{
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  [data-testid="stHorizontalBlock"] > [data-testid="column"],
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    flex: 1 1 calc(50% - 0.375rem) !important;
    max-width: calc(50% - 0.375rem) !important;
  }}
}}
.stButton > button {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  color: var(--ink);
  font-size: 0.875rem; font-weight: 500;
  padding: 1rem;
  text-align: left; line-height: 1.4;
  height: 6rem; min-height: 0;
  transition: border-color .15s ease, background-color .15s ease,
              box-shadow .15s ease, color .15s ease;
  box-shadow: var(--shadow-subtle);
}}
.stButton > button:hover {{
  border-color: rgba(5, 150, 105, 0.45);
  background: rgba(236, 253, 245, 0.5);
  box-shadow: var(--shadow-card);
  color: var(--green-dark);
}}
.stButton > button:focus-visible {{
  outline: 2px solid var(--green); outline-offset: 2px;
}}
.stButton > button:active {{ background: rgba(236, 253, 245, 0.8); }}

/* --- Suggestion cards ----------------------------------------------------- */
/* The four suggestion widgets are built on EVERY rerun. They used to be built
   only while the transcript was empty, and that was a functional bug, not a
   layout one: Streamlit discards the state of any widget it stops constructing,
   so from the second question onward the cards were gone from the page and a
   click on one bound to nothing -- `clicked` stayed None and no question was
   ever submitted. One working card, three dead ones.

   Gating them on visibility instead of on existence fixes that. The marker
   below is what scopes these rules to this row: st.columns emits a bare
   stHorizontalBlock with no class of ours, and the row is the only four-column
   block on the page, so :has() on the marker is the whole identification story.

   Once there is a conversation to read, the cards drop from the mockup's 96px
   card to a compact 2x2 strip of chips -- 72px measured against 96px on the
   landing page, question text never clipped. They stay on the page and stay
   clickable; they simply stop being a block that sits between the newest
   answer and the input. Removing them outright was the alternative, but a
   display:none card cannot be clicked, and "all four cards work after the first
   question" is the actual requirement. */
[data-testid="stHorizontalBlock"] [data-testid="element-container"]:has(.example-row-marker) {{
  display: none;
}}
[data-testid="stHorizontalBlock"]:has(.example-row-marker.is-compact)
  .stButton > button {{
  height: auto;
  font-size: 0.6875rem;
  line-height: 1.25;
  padding: 0.3rem 0.55rem;
  border-radius: 8px;
  color: var(--muted);
  background: transparent;
  box-shadow: none;
}}
[data-testid="stHorizontalBlock"]:has(.example-row-marker.is-compact)
  .stButton > button:hover {{
  background: var(--green-tint); color: var(--green-dark);
  border-color: rgba(5, 150, 105, 0.4); box-shadow: none;
}}
/* Keeping each chip's text on one line is a desktop-only instruction. Below
   769px Streamlit's own responsive stacking is left in charge: forcing nowrap
   and zero-basis columns there put four full questions on one line and
   overflowed a 390px viewport sideways, which is a worse failure than a tall
   suggestion block. */
@media (min-width: 769px) {{
  [data-testid="stHorizontalBlock"]:has(.example-row-marker.is-compact)
    > [data-testid="column"],
  [data-testid="stHorizontalBlock"]:has(.example-row-marker.is-compact)
    > [data-testid="stColumn"] {{
    flex: 0 0 auto !important;
    max-width: none !important;
  }}
  [data-testid="stHorizontalBlock"]:has(.example-row-marker.is-compact)
    .stButton > button {{
    white-space: nowrap;
  }}
}}

/* --- Chat input ----------------------------------------------------------- */
/* Streamlit ALREADY pins the chat input: [data-testid="stBottom"] is sticky and
   lives inside the main column, so natively the input tracks the content column
   and the sidebar correctly.

   An earlier version of this file overrode that with a viewport-relative pin,
   which looks equivalent and is not: such a box is positioned against the
   window, so its centred inner container was centred on the window instead of
   the content. With the sidebar open the input sat ~398px left of the answers it
   was answering. So: no positioning overrides here. Only appearance. */
[data-testid="stBottom"] {{
  background: rgba(255, 255, 255, 0.95);
  -webkit-backdrop-filter: blur(12px);
  backdrop-filter: blur(12px);
  border-top: 1px solid var(--border);
  z-index: 20;
}}
/* Alignment. The input's painted box has to land on the same two vertical lines
   as the text above it, at every width, or the question looks like it was typed
   somewhere else.

   The way to get there is to stop fighting Streamlit's box model and instead
   REPLICATE it. The main column is:

       .block-container            max-width 768, margin auto, padding 0 --gutter
       > stVerticalBlockBorderWrapper   padding ~19.4px, border 1px
       > content                    <- headings, chat messages, answers

   and the chat input's column was:

       stBottomBlockContainer
       > stVerticalBlockBorderWrapper   padding ~19.4px, border 1px
       > stChatInput                  max-width 768, margin auto

   Two bugs lived in that difference:

   1. The input's wrapper inset was stripped (an attempt to make the input line
      up with .block-container's PADDING EDGE). But no text is ever drawn there
      -- every heading and every chat message starts one wrapper inset further
      in. Stripping the wrapper therefore aligned the input with a line nothing
      else uses, leaving it 18px too wide and 18px too far left. Measured as a
      flat -18px/+18px at every width from 320 to 1920.
   2. Earlier still, the input was pinned against the viewport, so with the
      sidebar open it sat ~398px left of the answers.

   So: give stBottomBlockContainer the same box .block-container has, leave both
   stVerticalBlockBorderWrapper elements exactly as Streamlit renders them, and
   let the input simply fill what is left. No offsets are hardcoded -- the
   ~19.4px comes from Streamlit's own stylesheet and will follow a version bump.
   If Streamlit ever drops that wrapper, both columns lose the inset together
   and still agree. */
[data-testid="stBottomBlockContainer"] {{
  max-width: 768px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: var(--gutter) !important;
  padding-right: var(--gutter) !important;
}}
[data-testid="stChatInput"] {{
  max-width: none !important;
  margin: 0.75rem 0 0.75rem 0;
  padding-left: 0;
  padding-right: 0;
}}
[data-testid="stChatInput"] > div {{
  max-width: none;
  padding-left: 0;
  padding-right: 0;
}}
/* The border and the fill go on the box, not on the textarea. That box is what
   the reader sees AND what the alignment gate measures -- it is the textarea's
   parent -- so styling the textarea would have put a second, inset border
   inside the painted one. The mockup's field: `rounded-xl`, `surface-300`
   edge, `shadow-card`, and an emerald focus ring rather than a colour swap. */
[data-testid="stChatInput"] [data-baseweb="base-input"] {{
  background: var(--card);
  border: 1px solid var(--edge);
  border-radius: 12px;
  box-shadow: var(--shadow-card);
  transition: border-color .15s ease, box-shadow .15s ease;
}}
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {{
  border-color: var(--green);
  box-shadow: 0 0 0 3px var(--green-ring);
}}
[data-testid="stChatInput"] textarea {{
  background: transparent;
  border: none;
  box-shadow: none;
  font-family: var(--sans);
  font-size: 0.875rem;
  color: var(--ink);
  padding: 0.75rem 1rem;
  min-height: 3rem;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
[data-testid="stChatInput"] textarea:focus {{
  border: none; box-shadow: none; outline: none;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--faint); }}
/* Disabled while a question is in flight. */
[data-testid="stChatInput"] textarea:disabled {{ opacity: 0.6; }}

/* Send button. This is the one primary action on the page, so it is the one
   thing that wears the green fill; the suggestion cards above deliberately stay
   neutral for exactly that reason. The mockup's control is `h-8 w-8 rounded-lg
   bg-emerald-700 hover:bg-emerald-800 shadow-xs active:scale-95`.

   Vertical alignment was genuinely wrong and the cause was one property:
   Streamlit wraps the submit button in a flex box with `align-items: flex-end`,
   so a 40px button sat flush to the bottom of a 46px row. `align-self: center`
   on the child overrides the parent's flex-end, which is the minimal correct
   fix; the explicit height then makes the painted button match the field's. */
[data-testid="stChatInputSubmitButton"] {{
  align-self: center !important;
  height: 2rem !important;
  min-height: 0 !important;
  width: 2rem !important;
  padding: 0 !important;
  border-radius: 8px !important;
  background: var(--green) !important;
  border: 1px solid var(--green) !important;
  box-shadow: var(--shadow-subtle) !important;
  transition: background-color .15s ease, border-color .15s ease,
              transform .1s ease;
}}
[data-testid="stChatInputSubmitButton"]:hover {{
  background: var(--green-dark) !important;
  border-color: var(--green-dark) !important;
}}
[data-testid="stChatInputSubmitButton"]:active {{ transform: scale(0.95); }}
[data-testid="stChatInputSubmitButton"]:focus-visible {{
  outline: 2px solid var(--green-dark); outline-offset: 2px;
}}
[data-testid="stChatInputSubmitButton"] svg,
[data-testid="stChatInputSubmitButton"] path {{ color: #FFFFFF; stroke: #FFFFFF; }}

/* --- Status notices ------------------------------------------------------- */
/* Muted surfaces rather than Streamlit's saturated yellow/red banners. The
   refusal gets the mockup's treatment -- a white card with a rose hairline and
   a rose title -- because a scope refusal is a real answer, not an error. */
[data-testid="stAlert"] {{
  border-radius: 10px; font-size: 0.8125rem;
  border: 1px solid var(--border);
}}
[data-testid="stAlert"][data-baseweb="warning"] {{
  background: var(--warn-tint); color: var(--warn-ink);
  border-color: var(--warn-edge);
}}
[data-testid="stAlert"][data-baseweb="error"] {{
  background: var(--danger-tint); color: var(--danger);
  border-color: var(--danger-edge);
}}
[data-testid="stAlert"][data-baseweb="success"],
[data-testid="stAlert"][data-baseweb="info"] {{
  background: var(--green-tint); color: var(--green-dark);
  border-color: var(--green-edge);
}}

/* --- Misc ----------------------------------------------------------------- */
/* Spinner: small and emerald instead of the default grey. */
[data-testid="stSpinner"] i {{ border-top-color: var(--green) !important; }}
details summary {{ font-size: 0.6875rem; color: var(--muted); }}
/* Belt and braces against horizontal scroll at any width. Nothing in this app
   should ever scroll sideways. */
html, body, .stApp {{ overflow-x: hidden; max-width: 100vw; }}
* {{ min-width: 0; }}
[data-testid="stExpander"] pre, .stApp pre {{
  white-space: pre-wrap !important; word-break: break-word !important;
}}

/* --- Responsive ----------------------------------------------------------- */
@media (max-width: 768px) {{
  /* horizontal padding comes from --gutter; do not re-declare it here or the
     two definitions drift apart (this rule used to set 1rem, which is 4px wider
     than Streamlit's own 12px at 480px and under).

     No padding-bottom override either. It used to set 1rem here, which silently
     cancelled the 1.75rem clearance above at exactly the widths where the
     newest answer is closest to the input bar. Narrower screens get the same
     clearance as wide ones. */
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  [data-testid="stHorizontalBlock"] > [data-testid="column"],
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    min-width: 100%; flex: 1 1 100%;
  }}
  .user-bubble {{ max-width: 100%; }}
  .footer-legal {{ max-width: none; }}
}}
/* The mockup hides the separator and tagline below `sm`, leaving just the
   wordmark and the Facts-only pill. It does not restack the bar into a column,
   so neither does this: at 320px the wordmark and the pill still fit on one
   line, and restacking would throw away the bar's 56px silhouette. */
@media (max-width: 640px) {{
  .app-sep, .app-tagline {{ display: none; }}
  .app-header {{ gap: 0.5rem; }}
}}
@media (max-width: 480px) {{
  /* --gutter is 12px here, which is what this rule used to set by hand. */
  .source-row {{ gap: 0.3rem 0.45rem; }}
}}
</style>""", unsafe_allow_html=True)


# --- Header ------------------------------------------------------------------


def render_header() -> None:
    """Top bar: wordmark, tagline, 'Facts-only' pill.

    The mockup separates the tagline from the wordmark with a literal pipe and
    hides both below `sm`. The pipe is markup here because it is a separator,
    not a character of either label.
    """
    st.markdown(
        f"""
<div class="app-header">
  <div class="app-header-left">
    <div class="app-brand">
      <span class="brand-dot"></span>HDFC Scheme Facts
    </div>
    <span class="app-sep">|</span>
    <span class="app-tagline">Mutual fund facts, simply explained</span>
  </div>
  <span class="facts-pill">Facts-only</span>
</div>
""",
        unsafe_allow_html=True,
    )


HEADER_CSS = f"""<style>
/* Rendered by render_header() and styled in inject_css(), so the two live
   together. Kept as a separate sheet because the header is the one element
   whose rules never need to know about Streamlit's DOM. */
.app-brand {{ white-space: nowrap; }}
</style>"""


def inject_header_css() -> None:
    st.markdown(HEADER_CSS, unsafe_allow_html=True)


# --- Sidebar -----------------------------------------------------------------


def render_sidebar(show_sources: bool, memory_len: int,
                   memory_window: int, chunks_label: str) -> "tuple":
    """Compact navigation column. Returns (show_sources, clear_requested).

    The debug controls live in a collapsed expander: they are genuinely useful
    for a class demo, but "10 chunks per question" was previously the most
    prominent text in the sidebar, ahead of the disclaimer itself.

    This function draws and reads widget state. It never touches memory or the
    transcript; clearing them is the caller's decision.
    """
    st.markdown(
        f'<div class="sidebar-head">'
        f'<span class="sidebar-title">HDFC Schemes</span>'
        f'<span class="sidebar-count">{len(SIDEBAR_SCHEMES)} Tracked</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    items = "".join(
        f'<div class="scheme-item"><span class="scheme-glyph">'
        f'{SCHEME_GLYPHS[name]}</span>'
        f'<span>{html.escape(name)}</span>'
        f'<span class="scheme-tag">{html.escape(SCHEME_TAGS[name])}</span>'
        f'</div>'
        for name in SIDEBAR_SCHEMES
    )
    st.markdown(f'<div class="scheme-nav">{items}</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="disclaimer-strip"><strong>Facts-only. No advice.</strong>'
        f'Facts come exclusively from {len(SIDEBAR_SCHEMES)} public scheme pages '
        f'&amp; fact sheets.</div>',
        unsafe_allow_html=True,
    )

    # Developer options, collapsed by default.
    with st.expander("Developer options"):
        next_sources = st.checkbox(
            "Show retrieved chunks",
            value=show_sources,
            help="Makes the RAG visible: which chunks the answer was built from.",
        )
        st.caption(chunks_label)
        st.caption(
            f"Conversation memory: {memory_len} of {memory_window} turns. "
            "Every turn is re-checked by the guardrails before it reaches the "
            "model, and history is never treated as a source of facts."
        )
        clear = st.button("Clear conversation memory",
                          use_container_width=True,
                          disabled=memory_len == 0)
    return next_sources, clear


# --- Chat pieces --------------------------------------------------------------


def _escape(text: str) -> str:
    """Escape model/URL text before it goes into raw HTML."""
    return html.escape(str(text), quote=True)


def render_suggestion_label(first_visit: bool) -> None:
    """Muted label above the example cards.

    The mockup's header is a two-sided row: an uppercase label and a hint that
    the cards are live. Both are inert text with no click to lose, so gating
    them on the empty transcript is free -- unlike the cards themselves.
    """
    if not first_visit:
        return
    st.markdown(
        '<div class="suggest-head">'
        '<span class="suggest-label">Try one of these</span>'
        '<span class="suggest-hint">Click to ask instantly</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_suggestion_marker(first_visit: bool) -> None:
    """Zero-size marker that tells CSS which row is the suggestion row.

    Rendered inside the first suggestion column on every run, so the compact
    rules in inject_css() can find this row through :has() and leave every other
    column block alone. `first_visit` only picks the class: the element itself
    always exists, because the cards beside it always exist.
    """
    cls = "example-row-marker"
    if not first_visit:
        cls += " is-compact"
    st.markdown(f'<div class="{cls}"></div>', unsafe_allow_html=True)


def render_resolution_note(resolved_question: str) -> None:
    """Show what we searched for when memory completed a pronoun."""
    st.markdown(
        f'<div class="resolve-note">Read as: {_escape(resolved_question)}</div>',
        unsafe_allow_html=True,
    )


def render_footer_note(note: str) -> None:
    st.markdown(f'<div class="footer-note">{_escape(note)}</div>',
                unsafe_allow_html=True)


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _trusted_html(text: str) -> str:
    """Escape, then re-enable just `**bold**`.

    The escape-then-reopen pattern only works because the input is a literal in
    this file. Never call it with model output, retrieval text, or anything a
    user typed -- those go through st.markdown() with unsafe_allow_html left
    off, or through _escape().
    """
    return _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", _escape(text))


def render_trusted_markdown(text: str, css_class: str = "footer-legal",
                            highlight: str = "") -> None:
    """Render markdown for a REPO CONSTANT only (e.g. DISCLAIMER, WELCOME).

    `highlight` wraps one exact phrase in an emerald span, which is how the
    welcome line gets the mockup's "Facts-only. No investment advice." accent
    without the copy itself having to carry markup.
    """
    body = _trusted_html(text)
    if highlight and highlight in body:
        body = body.replace(
            _escape(highlight), f'<span class="accent">{_escape(highlight)}</span>'
        )
    st.markdown(f'<div class="{css_class}">{body}</div>',
                unsafe_allow_html=True)


def render_user_bubble(text: str) -> None:
    """Right-aligned slate bubble with the assistant-facing corner squared off.

    The wrapper container is the chat message body; the bubble itself is the
    styled inline block.
    """
    st.markdown(
        f'<div class="user-bubble">{_escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _scheme_label(chunks: Optional[Sequence], answer) -> str:
    """A short human label for the source row: prefer the scheme name."""
    for chunk in chunks or ():
        name = getattr(chunk, "scheme_name", "") or ""
        if name:
            return name
    return ""


def render_source_row(source_url: str, last_updated: str,
                      scheme_label: str = "") -> None:
    """Compact, non-repeating citation, in the mockup's citation-box shape.

    The full URL is preserved verbatim -- it is the same string the pipeline
    produced, attached to the same anchor. Only the presentation changed: the
    long address is no longer printed inline, which is what removed the
    horizontal overflow. `title` keeps it available on hover.

    The Verified Fact badge is emitted only when there is an address to point
    at, so it asserts what the pipeline already established (the answer came
    back grounded and cited) rather than adding a new claim.
    """
    if not source_url:
        if last_updated:
            st.markdown(
                f'<div class="source-date">Last updated from sources: '
                f'{_escape(last_updated)}</div>',
                unsafe_allow_html=True,
            )
        return

    safe_url = _escape(source_url)
    subject = _escape(scheme_label) if scheme_label else "the indexed page"
    provenance = (
        f'<span class="source-meta">Disclosed</span>'
        f'<span class="source-date">{_escape(last_updated)}</span>'
        if last_updated
        else ""
    )
    st.markdown(
        f'<div class="source-row">'
        f'<a class="source-link" href="{safe_url}" target="_blank" '
        f'rel="noopener noreferrer" title="{safe_url}">'
        f'<span class="source-label">Source:</span> {subject} ↗</a>'
        f'<span class="verified-pill">Verified Fact</span>'
        f'{provenance}'
        f'</div>',
        unsafe_allow_html=True,
    )


def split_citation(text: str) -> "tuple":
    """Separate the answer prose from the citation block the validator appends.

    `Answer.text` deliberately ends with "Source: <url>" and "Last updated from
    sources: <date>" -- that is the pipeline's contract and it is untouched. But
    the UI also draws a proper source row, so rendering the raw text put the same
    URL on screen twice: once as bare text inside the prose, once as the link.
    That duplication was one of the reported layout problems.

    Returns (prose, url, date) so the caller can render the citation once, as a
    row, instead of twice. Purely a display concern: what gets stored in the
    transcript and fed to memory is the original `Answer.text`, unchanged.

    The strip is tail-only. A "Source:" phrase quoted mid-answer is part of the
    answer and is left alone.
    """
    if not text:
        return "", "", ""

    lines = text.rstrip().splitlines()
    url = ""
    date = ""
    # Walk backwards: the validator always appends date last, then URL.
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.lower().startswith("last updated from sources:") and not date:
            date = last.split(":", 1)[1].strip()
            lines.pop()
            continue
        source_match = re.match(r"^Source:\s*(\S+)\s*$", last, re.IGNORECASE)
        if source_match and not url:
            url = source_match.group(1)
            lines.pop()
            continue
        break

    return "\n".join(lines).strip(), url, date


def render_answer_card(text: str) -> None:
    """The assistant answer on a white card, with the prose as the hero.

    The card is a `st.container(border=True)`, not a <div> in raw HTML. The
    obvious version -- open the div with one st.markdown call and close it with
    another -- does not work: Streamlit sanitises and renders each markdown
    block on its own, so the opening tag became an empty div and the closing tag
    was discarded. The card then styled nothing while still appearing in the DOM,
    which is the worst kind of bug: present in markup, absent in pixels.

    A real container also keeps the model's own markdown rendering, which raw
    HTML would have flattened into literal asterisks.
    """
    prose, _url, _date = split_citation(text)
    with st.container(border=True):
        # Plain st.markdown, so a model's own bold/lists still render. The card's
        # typography comes from CSS on the container, not from wrapping the prose
        # in HTML (which would flatten the markdown to literal asterisks).
        if prose:
            st.markdown(prose)


def render_loading() -> None:
    """The mockup's generating state: a spinner line, then shimmering bars.

    No pipeline jargon in the copy (embedding, vector search, ...), and the
    shimmer is three fixed-width bars rather than a real skeleton of the answer:
    the widths are decorative and nothing about them is load-bearing.
    """
    with st.container(border=True):
        st.markdown(
            '<div class="loading-copy">'
            '<span class="loading-spin"></span>'
            'Searching official HDFC scheme disclosure &amp; factsheets…'
            '</div>'
            '<div class="shimmer">'
            '<div class="shimmer-line" style="width:83%"></div>'
            '<div class="shimmer-line" style="width:75%"></div>'
            '<div class="shimmer-line" style="width:50%"></div>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_status(kind: str, message: str) -> None:
    """Status strip under an answer: refused / not-found / provider error."""
    entries = {
        "refused": (
            "notice-refused", "Query outside scope",
            "This asks for advice or a prediction rather than a scheme fact, so "
            "it was refused before the model was called. Nothing left your "
            "machine.",
        ),
        "not_found": (
            "notice-notfound", "Not found in the indexed pages",
            "Try naming the scheme and the fact, e.g. “What is the exit load "
            "on HDFC Small Cap?”",
        ),
        "error": (
            "notice-error", "The model provider did not respond in time",
            "This is not a problem with your question or with the indexed data "
            "— try again.",
        ),
    }
    css, title, body = entries.get(kind, ("notice-generic", "", message))
    st.markdown(
        f'<div class="notice {css}">'
        f'<span class="notice-title">{_escape(title)}</span>'
        f'<span class="notice-body">{_escape(body)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


STATUS_CSS = f"""<style>
/* A status strip is an answer, not an alarm: a title line and a quiet body,
   on the mockup's card shape. The refusal wears a rose hairline because a
   scope refusal is the one status a reader must not have to notice. */
.notice {{
  margin-top: 0.6rem; padding: 0.875rem 1rem;
  border-radius: 12px; border: 1px solid var(--border);
  background: var(--surface); color: var(--muted);
  font-size: 0.75rem; line-height: 1.6;
  box-shadow: var(--shadow-subtle);
}}
.notice-title {{ display: block; font-weight: 600; margin-bottom: 0.25rem; }}
.notice-body {{ display: block; color: var(--muted); }}
.notice-refused {{
  background: var(--card); border-color: var(--danger-edge);
}}
.notice-refused .notice-title {{ color: var(--danger); }}
.notice-notfound {{
  background: var(--warn-tint); border-color: var(--warn-edge);
}}
.notice-notfound .notice-title {{ color: var(--warn-ink); }}
.notice-error {{
  background: var(--danger-tint); border-color: var(--danger-edge);
}}
.notice-error .notice-title {{ color: var(--danger); }}

/* The generating state. A CSS ring rather than an <svg>, because markdown
   sanitising drops inline SVG and a dropped spinner is a dropped spinner. */
.loading-copy {{
  display: flex; align-items: center; gap: 0.5rem;
  font-size: 0.75rem; font-weight: 500; color: var(--green-dark);
  animation: loading-breathe 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}}
.loading-spin {{
  flex-shrink: 0; width: 0.875rem; height: 0.875rem;
  border: 2px solid rgba(5, 150, 105, 0.25);
  border-top-color: var(--green-mid); border-radius: 999px;
  animation: loading-spin 0.8s linear infinite;
}}
@keyframes loading-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes loading-breathe {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.45; }}
}}
.shimmer {{ display: flex; flex-direction: column; gap: 0.5rem;
  padding-top: 0.75rem; }}
.shimmer-line {{
  height: 0.75rem; border-radius: 999px; background: var(--surface);
  animation: shimmer-pulse 1.6s ease-in-out infinite;
}}
.shimmer-line:nth-child(2) {{ animation-delay: 0.15s; }}
.shimmer-line:nth-child(3) {{ animation-delay: 0.3s; }}
@keyframes shimmer-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
</style>"""


def inject_status_css() -> None:
    st.markdown(STATUS_CSS, unsafe_allow_html=True)


def render_chunks(chunks: List) -> None:
    """Debug view of retrieved chunks. Kept, collapsed under Developer options."""
    with st.expander(f"Retrieved chunks ({len(chunks)})"):
        for rank, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**{rank}.** `{chunk.chunk_id}` — score {chunk.score:.3f} — "
                f"{chunk.scheme_id} · {chunk.section}"
            )
            st.caption(
                chunk.text[:400] + ("..." if len(chunk.text) > 400 else "")
            )


def render_page_heading(intro: str = "") -> None:
    """Hero block shown only while the conversation is empty.

    The mockup's landing hero is four pieces in a fixed order: an eyebrow pill
    with a status dot, the 30/36px wordmark, a one-line summary of what you can
    ask, and then an inset panel that introduces the assistant in its own voice.
    Merging any two of them was what made an earlier version read as a paragraph
    of developer copy.
    """
    st.markdown(
        '<div class="page-eyebrow">'
        '<span class="page-eyebrow-dot"></span>'
        'Official Fact-Grounding Engine'
        '</div>'
        '<div class="page-h1">HDFC Scheme Facts</div>'
        '<div class="page-sub">Ask about expense ratio, exit load, SIP minimum, '
        'benchmark, riskometer and more.</div>',
        unsafe_allow_html=True,
    )
    if intro:
        render_trusted_markdown(intro, css_class="page-intro",
                                highlight="Facts-only. No investment advice.")


PAGE_CSS = f"""<style>
/* Hero rhythm, from the mockup's `space-y-8` on the landing block. */
.page-eyebrow {{
  display: inline-flex; align-items: center; gap: 0.5rem;
  padding: 0.25rem 0.625rem; border-radius: 999px;
  background: var(--surface); color: var(--muted);
  font-size: 0.75rem; font-weight: 500; margin-bottom: 0.75rem;
}}
.page-eyebrow-dot {{
  width: 6px; height: 6px; border-radius: 999px; background: var(--green-mid);
}}
.page-h1 {{
  font-size: 1.875rem; font-weight: 700; color: var(--ink);
  letter-spacing: -0.025em; line-height: 1.2; margin-bottom: 0.5rem;
}}
@media (min-width: 640px) {{ .page-h1 {{ font-size: 2.25rem; }} }}
.page-sub {{
  font-size: 1rem; color: var(--muted); margin-bottom: 1.25rem;
  line-height: 1.55; max-width: 46ch;
}}
.page-intro {{
  font-size: 0.75rem; color: var(--muted); line-height: 1.65;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 0.875rem; max-width: none; margin-bottom: 0;
}}
.page-intro strong {{ color: var(--ink); font-weight: 600; }}
.accent {{ color: var(--green-dark); font-weight: 500; }}
.suggest-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.75rem; margin: 1.5rem 0 0.75rem 0;
}}
.suggest-label {{
  font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.075em;
  color: var(--faint); font-weight: 600; margin: 0;
}}
.suggest-hint {{ font-size: 0.6875rem; color: var(--faint); }}
@media (max-width: 768px) {{
  .page-sub {{ font-size: 0.9375rem; }}
  .suggest-hint {{ display: none; }}
}}
</style>"""


def inject_page_css() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def render_compliance_banner(title: str, body: str) -> None:
    """The mockup's mandatory compliance banner, as ONE element.

    The mockup's banner is a single rounded box holding a bold lead line above
    a paragraph. In Streamlit those two are separate widgets -- two markdown
    calls produce two sibling containers -- so drawing them as one box means
    emitting one element. Both are in one st.markdown() call on purpose: an
    element opened in one call and closed in another does not nest, because each
    markdown block is sanitised and rendered independently.

    `body` is passed verbatim. The leading bold restatement of `title` is
    dropped here rather than at the call site, so the PRD copy stays verbatim
    and there is exactly one place deciding not to print the same sentence
    twice.
    """
    lead = f"**{title}**"
    text = body[len(lead):].lstrip() if body.startswith(lead) else body
    st.markdown(
        f'<div class="compliance-banner">'
        f'<div class="footer-note compliance-title">'
        f'<span class="compliance-mark">i</span>{_escape(title)}</div>'
        f'<p class="footer-legal compliance-body">{_trusted_html(text)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


COMPLIANCE_CSS = f"""<style>
/* The mockup's banner: a hairline box on a 60%-white slate field, with the
   lead line set in ink and the paragraph left at the muted tone. */
.compliance-banner {{
  border: 1px solid rgba(226, 232, 240, 0.8);
  background: rgba(241, 245, 249, 0.6);
  border-radius: 12px; padding: 1rem; margin-top: 2rem;
  max-width: 78ch;
}}
.compliance-title {{
  display: flex; align-items: center; gap: 0.375rem;
  font-size: 0.75rem; font-weight: 600; color: var(--ink); margin-bottom: 0.35rem;
}}
.compliance-mark {{
  flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center;
  width: 1rem; height: 1rem; border-radius: 999px;
  background: var(--green-dark); color: #FFFFFF;
  font-family: var(--sans); font-size: 0.6875rem; font-weight: 700; line-height: 1;
}}
.compliance-body {{
  font-size: 0.6875rem; color: var(--muted); line-height: 1.65; margin: 0;
  max-width: none;
}}
.compliance-body strong {{ color: var(--ink); font-weight: 600; }}
@media (max-width: 768px) {{ .compliance-banner {{ padding: 0.875rem; }} }}
</style>"""


def inject_compliance_css() -> None:
    st.markdown(COMPLIANCE_CSS, unsafe_allow_html=True)


def render_persistent_note(text: str) -> None:
    st.markdown(f'<div class="source-date" style="margin-top:1.5rem">'
                f'{_escape(text)}</div>', unsafe_allow_html=True)


# --- Auto-scroll -------------------------------------------------------------

# Why this exists, and why it is JS in an iframe rather than CSS.
#
# Streamlit 1.38 simply does not follow the conversation. Measured on this app,
# three ways, on a fresh load each time:
#
#   * type a question and press Enter      -> scrollTop stays 78, max goes 78->203
#   * click an example card                -> scrollTop stays 78, max goes 78->203
#   * scroll to the bottom by hand first,
#     then ask a second question           -> scrollTop stays 203, max goes 203->256
#
# So the newest answer always renders about 125px below the fold and the reader
# has to go looking for it. No amount of CSS fixes that: the sticky question bar
# already reserves its space correctly (see .block-container), the content really
# is there, the browser simply is not asked to move. It needs a script.
#
# `st.components.v1.html` puts this in an iframe that carries both
# `allow-same-origin` and `allow-scripts`, so the iframe is same-origin with the
# page and can reach `window.parent.document`. `st.markdown` will not do: it
# strips <script> as part of sanitising markdown.
#
# Height 0 so the iframe contributes no layout of its own. Purely a scroll
# instruction: it reads the scroll container and writes scrollTop. It never
# touches the DOM, the transcript or the pipeline.
_SCROLL_LATEST = """
<script>
(function () {
  var SEL = '[data-testid="ScrollToBottomContainer"]';
  var doc = window.parent.document;
  var win = window.parent;

  function box() { return doc.querySelector(SEL); }
  function pin() {
    var b = box();
    if (b) { b.scrollTop = b.scrollHeight; }
    return b;
  }

  // Following is a question of INTENTION, not of position, and the distinction
  // is the whole ballgame here.
  //
  // The obvious implementation -- keep a running "is the reader at the bottom?"
  // flag from the container's scroll event, and only re-pin while it is true --
  // was tried and does not work. Scroll events fire for reasons that have
  // nothing to do with the reader: the browser re-anchors scrollTop when the
  // viewport reflows, and the container grows under an in-flight render. Measured
  // after shrinking the window to 390px, the flag had been knocked to false by
  // one of those, and from then on the pin refused to run at all -- the answer
  // ended up 425px under the question bar and stayed there. A position flag is
  // also latched: once wrong it never recovers.
  //
  // So the flag only ever moves in response to something a person did: a wheel,
  // a touch drag, a key, or a pointer press. Those are unambiguous.
  if (win.__hdfcFollowing === undefined) { win.__hdfcFollowing = true; }

  // Reader-intent listeners, on the scroll container itself so a wheel or a
  // scrollbar drag is caught wherever it lands.
  //
  // Re-attached on every execution for the same reason as the viewport hooks:
  // the flag a node carries outlives the iframe that attached the listener to
  // it, so a plain "already wired" guard would silently leave later turns with
  // no way to notice a reader scrolling back up. Storing the handler on the node
  // lets the previous one be removed, so the listeners never accumulate either.
  function wire() {
    var b = box();
    if (!b) { return; }
    var mark = function () {
      // Read on the next frame: for a wheel the browser has not applied the
      // delta yet, and reading now would always report the old position.
      win.requestAnimationFrame(function () {
        var c = box();
        if (!c) { return; }
        win.__hdfcFollowing =
          (c.scrollHeight - c.clientHeight - c.scrollTop) < 24;
      });
    };
    if (b.__hdfcMark) {
      b.removeEventListener('wheel', b.__hdfcMark);
      b.removeEventListener('touchmove', b.__hdfcMark);
      b.removeEventListener('pointerdown', b.__hdfcMark);
      b.removeEventListener('keydown', b.__hdfcMark, true);
    }
    b.__hdfcMark = mark;
    b.addEventListener('wheel', mark, {passive: true});
    b.addEventListener('touchmove', mark, {passive: true});
    b.addEventListener('pointerdown', mark, {passive: true});
    // Capture, so PageUp / Home / a scrollbar drag still count. Typing lands
    // here too and is harmless: the question box sits at the bottom, so the
    // reading is "at the bottom" and following stays true.
    b.addEventListener('keydown', mark, true);
  }

  // A viewport change and the reflow it triggers do not land in the same frame,
  // and something re-anchors scrollTop *after* any one-shot attempt: measured
  // 1440->1280 left the view 13px short, 1024 left it 122px short, and the two
  // narrow layouts dumped it back to 74px of a 602px scroll. So a single pin
  // loses the race whatever it is timed at.
  //
  // Poll for a short window instead of firing once. Cheap -- it only writes
  // scrollTop, which the browser clamps, so it cannot overshoot -- and it wins
  // because it is still running when the reflow that displaced the view has
  // finished. Cancelled and replaced on every trigger so the loops never stack.
  //
  // The reader still has the last word: __hdfcFollowing is cleared by a wheel,
  // touch, key or pointer press, and every pin checks it first.
  function settle(ms) {
    if (win.__hdfcSettle) { win.__hdfcSettle.cancelled = true; }
    var job = {cancelled: false, until: win.Date.now() + ms};
    win.__hdfcSettle = job;
    function again() {
      if (job.cancelled) { return; }
      if (win.__hdfcFollowing) { pin(); }
      if (win.Date.now() < job.until) { win.setTimeout(again, 90); }
    }
    again();
  }

  // Shrinking or rotating the window moves the answer without a Streamlit rerun,
  // so nothing else would re-pin it. Two triggers, because they cover different
  // ground: the window event catches a viewport change, and a ResizeObserver on
  // the scroll container also catches a sidebar toggle or the container being
  // replaced by a fresh element.
  //
  // Registered on EVERY execution, replacing whatever was there before, and
  // that ordering is the whole fix. Registering once behind a flag on the parent
  // window looks tidier and is wrong: a rerun replaces the component, which
  // destroys the iframe that did the registering, and a listener the parent
  // window holds on behalf of a discarded child dies with it. The flag
  // survives, the listener does not, so from the second question onwards nothing
  // was watching the viewport at all -- measured, the view stopped following
  // resizes and sat 425px under the question bar at 390px wide. Each execution
  // therefore re-attaches, and disconnects the previous observer so the
  // registrations never accumulate.
  function watch() {
    var b = box();
    if (b && b !== win.__hdfcObserved) {
      win.__hdfcObserved = b;
      win.__hdfcRO.observe(b);
    }
  }

  if (win.__hdfcOnResize) {
    win.removeEventListener('resize', win.__hdfcOnResize);
  }
  win.__hdfcOnResize = function () { settle(1600); };
  win.addEventListener('resize', win.__hdfcOnResize);

  if (win.ResizeObserver) {
    if (win.__hdfcRO) { try { win.__hdfcRO.disconnect(); } catch (e) {} }
    win.__hdfcRO = new win.ResizeObserver(function () { settle(1200); });
    watch();
  }

  wire();
  watch();

  // A rerun delivers the answer in pieces, so let the pin keep running for a
  // while rather than firing once.
  settle(2600);
})();
</script>
"""


def scroll_to_latest() -> None:
    """Bring the newest answer into view. No-op outside a browser."""
    try:
        st.components.v1.html(_SCROLL_LATEST, height=0)
    except Exception:  # noqa: BLE001 - a scroll hint must never break the app
        # AppTest and any headless render have no iframe to reach. Losing the
        # convenience is acceptable; taking down a demo over it is not.
        pass


__all__ = [
    "GREEN", "GREEN_DARK", "GREEN_MID", "GREEN_TINT", "GREEN_EDGE", "INK",
    "BODY", "MUTED", "FAINT", "BORDER", "EDGE", "SURFACE", "CARD", "PAGE",
    "DANGER", "DANGER_TINT", "DANGER_EDGE", "WARN_TINT", "WARN_INK",
    "SIDEBAR_SCHEMES", "SCHEME_GLYPHS", "SCHEME_TAGS",
    "inject_css", "inject_header_css", "inject_status_css", "inject_page_css",
    "inject_compliance_css",
    "render_header", "render_sidebar", "render_user_bubble", "render_source_row",
    "render_answer_card", "render_loading", "render_status", "render_chunks",
    "render_page_heading", "render_suggestion_label", "render_suggestion_marker",
    "render_resolution_note", "render_compliance_banner",
    "render_footer_note", "render_trusted_markdown", "scroll_to_latest",
]