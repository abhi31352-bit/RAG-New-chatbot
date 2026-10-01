"""Presentation layer for the Streamlit UI.

Everything here is CSS and HTML. It imports nothing from the RAG pipeline --
it is handed an `Answer` and a list of chunks and told how to draw them. That
separation is the point: if this module needed to reach into retrieval to render,
a styling change could quietly break the pipeline.

What the CSS is actually fighting
---------------------------------
Streamlit's defaults are built for data tools, not for consumer fintech:

  * `st.chat_message` gives every message a large tinted block with an avatar
    bubble, so user and assistant text compete for attention. Here the user
    bubble shrinks to a light green pill and the assistant answer sits on plain
    white, because the answer is the thing being read.
  * Source citations render as a long raw URL in a markdown link. A 70-character
    URL is the widest element on the page and is what pushes the layout into
    horizontal overflow. Here the URL is a compact "View source" link and the
    full address is kept in `title=` so it is still available on hover.
  * The sidebar repeats the full disclaimer paragraph, which is the most
    visually dominant thing in a column that should be navigation.

Colour is declared once, as custom properties, so the palette is greppable and
the green cannot drift between the header, the buttons and the links.
"""
from __future__ import annotations

import html
import re
from typing import List, Optional, Sequence

import streamlit as st

# Groww-inspired palette, NOT Groww branding. Green is the action/accent colour
# because that is the convention for Indian investing products and it reads as
# "gain/positive" in this category; the specific values below are chosen for
# contrast on a near-white background, not copied from any brand.
GREEN = "#0F9D58"
GREEN_DARK = "#0B7A45"
GREEN_TINT = "#EAF7F0"
GREEN_RING = "rgba(15, 157, 88, 0.18)"

INK = "#1A1D21"
MUTED = "#6B7280"
FAINT = "#9AA1A9"
BORDER = "#E8EAED"
CARD = "#FFFFFF"
PAGE = "#FAFAFA"
DANGER = "#B42318"
DANGER_TINT = "#FEF3F2"
WARN_TINT = "#FFFAEB"
WARN_INK = "#B54708"

# Sidebar shows the five scheme names; each gets a short glyph so the list scans
# faster than five identical bullets. These are text, not icon assets -- no
# external files, no licensing questions, and they inherit font colour.
SCHEME_GLYPHS = {
    "Large Cap": "◆",
    "Flexi Cap": "◇",
    "ELSS Tax Saver": "■",
    "Small Cap": "▲",
    "Balanced Advantage": "●",
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
    st.markdown(f"""<style>
:root {{
  --green: {GREEN};
  --green-dark: {GREEN_DARK};
  --green-tint: {GREEN_TINT};
  --ink: {INK};
  --muted: {MUTED};
  --faint: {FAINT};
  --border: {BORDER};
  --card: {CARD};
  --page: {PAGE};
  --danger: {DANGER};
  --danger-tint: {DANGER_TINT};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--page);
}}

/* --- Typography scale ----------------------------------------------------- */
.stApp h1 {{ font-size: 2rem !important; line-height: 1.2 !important;
  letter-spacing: -0.02em; font-weight: 650 !important; color: var(--ink); }}
.stApp h2, .stApp h3 {{ letter-spacing: -0.01em; color: var(--ink); }}
/* Muted everything Streamlit marks as caption. These are the secondary lines:
   dates, hints, scheme subtitles. They were competing with body text. */
.stApp .stCaptionContainer p, .stApp [data-testid="stCaptionContainer"] p,
.stApp small {{ color: var(--muted) !important; font-size: 0.8125rem; }}
.stApp p, .stApp li {{ color: var(--ink); line-height: 1.62;
  overflow-wrap: anywhere; word-break: normal; }}
.stApp a {{ color: var(--green); text-decoration: none;
  overflow-wrap: anywhere; word-break: break-word; }}
.stApp a:hover {{ text-decoration: underline; }}

hr {{ border-color: var(--border) !important; margin: 1.25rem 0 !important; }}

/* --- Layout: centred column, never wider than needed --------------------- */
/* The single biggest visual problem: default Streamlit content spans the full
   browser width, so on a 27" monitor the answer text sits in a thin ribbon in
   the middle of an ocean of white. Capping block width and centring fixes it. */
.block-container {{
  max-width: 980px !important;
  padding-top: 1.75rem !important;
  padding-bottom: 7rem !important;  /* clear the fixed chat input */
  margin: 0 auto !important;
}}

/* Example-card row. Streamlit lays columns out in a fixed-width flex row that
   scrolls sideways on a phone; wrapping to full-width blocks is what keeps the
   "no horizontal scrolling" promise true on a 375px screen. */
[data-testid="stHorizontalBlock"] {{ gap: 0.6rem; }}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ min-width: 0; }}

/* --- Header --------------------------------------------------------------- */
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"] {{ right: 8px; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ display: none; }}

/* --- Cards --------------------------------------------------------------- */
/* Generic surface. Restrained: 1px border, 10px radius, one very soft shadow.
   Deliberately NOT a heavy drop shadow -- that reads as a dev dashboard. */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.15rem;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}}

/* --- Chat messages -------------------------------------------------------- */
/* User: right-aligned, tinted pill. Assistant: left, white, no heavy block.
   Streamlit nests the avatar, so the alignment is done on the inner content. */
[data-testid="stChatMessage"] {{ margin-bottom: 0.5rem; }}
[data-testid="stChatMessageContent"] {{ gap: 0.7rem; align-items: flex-start; }}
/* Hide the default avatars. They are large, repeated on every turn, and the
   user/assistant distinction is already carried by alignment and tint.
   Testids verified against the live DOM in 1.38: the avatar wrappers are
   stChatMessageAvatarUser / stChatMessageAvatarAssistant. The older
   stChatMessageContentAvatar name is kept as a no-op safety net. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageContentAvatar"] {{ display: none; }}
[data-testid="stChatMessageContent"] {{ gap: 0; }}
[data-testid="stChatMessageContentBody"] {{
  font-size: 0.9375rem; line-height: 1.65; overflow-wrap: anywhere;
  word-break: normal; min-width: 0;
}}

/* User message = tinted, narrower, pushed right. Streamlit has no role
   attribute on the message, so the .user-bubble marker inside it is what
   identifies the role, via :has(). */
[data-testid="stChatMessage"]:has(.user-bubble) {{
  flex-direction: row-reverse;
}}
.user-bubble {{
  background: var(--green-tint);
  border: 1px solid rgba(15, 157, 88, 0.18);
  border-radius: 14px;
  padding: 0.55rem 0.9rem;
  color: var(--ink);
  max-width: 78%;
  display: inline-block;
  margin-left: auto;
  font-size: 0.9375rem; line-height: 1.55;
}}

/* Assistant answer: a real bordered container (st.container(border=True)), so
   the card exists in Streamlit's own DOM rather than in a hand-written div.
   This is the element the user came to read, so it gets the most space and the
   least decoration. */
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--card);
  border-color: var(--border);
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  overflow-wrap: anywhere;
}}
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] p {{
  margin: 0;
  font-size: 1rem; line-height: 1.68; color: var(--ink);
}}

/* --- Source row ----------------------------------------------------------- */
/* The old citation printed the whole groww.in URL inline. That is ~70 chars of
   unbroken text: the widest thing on the page, and the direct cause of
   horizontal overflow. It becomes a compact row, with the full URL kept in
   title= for hover and for screen readers. The address itself is unchanged --
   only its presentation. */
.source-row {{
  display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 0.4rem 0.6rem; margin-top: 0.15rem; min-width: 0;
}}
.source-label {{
  font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--faint); font-weight: 600;
}}
.source-link {{
  font-size: 0.8125rem; color: var(--green); font-weight: 550;
  white-space: nowrap;
}}
.source-meta {{ font-size: 0.8125rem; color: var(--muted); }}
.source-date {{ font-size: 0.75rem; color: var(--faint); margin-top: 0.3rem; }}

/* --- Buttons -------------------------------------------------------------- */
/* Example cards: full-width, left-aligned, border on hover. Green fill is
   reserved for the one primary action (the send button) so it keeps meaning. */
.stButton > button {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--ink);
  font-size: 0.8125rem; font-weight: 500;
  padding: 0.6rem 0.85rem;
  text-align: left; line-height: 1.4;
  height: auto; min-height: 0;
  transition: border-color .15s ease, background-color .15s ease,
              box-shadow .15s ease;
  box-shadow: none;
}}
.stButton > button:hover {{
  border-color: rgba(15, 157, 88, 0.45);
  background: var(--green-tint);
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
  color: var(--green-dark);
}}
.stButton > button:focus-visible {{
  outline: 2px solid var(--green); outline-offset: 2px;
}}
.stButton > button:active {{ background: rgba(15, 157, 88, 0.10); }}

/* --- Chat input ----------------------------------------------------------- */
/* Pinned so it stays reachable while scrolling a long conversation. The send
   button is the single primary action on the page, so it is the single filled
   green element. */
[data-testid="stChatInput"] {{
  position: fixed; bottom: 0; left: 0; right: 0;
  background: rgba(250, 250, 250, 0.94);
  backdrop-filter: blur(8px);
  border-top: 1px solid var(--border);
  padding: 0.7rem 0 0.85rem 0;
  z-index: 999;
  margin: 0 !important;
}}
[data-testid="stChatInput"] > div {{
  max-width: 980px; margin: 0 auto; padding: 0 1rem;
}}
[data-testid="stChatInput"] textarea {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  font-size: 0.9375rem;
  padding: 0.65rem 0.85rem;
  min-height: 2.75rem;
  overflow-wrap: anywhere;
}}
[data-testid="stChatInput"] textarea:focus {{
  border-color: var(--green);
  box-shadow: 0 0 0 3px var(--green-ring);
  outline: none;
}}
/* Disabled while a question is in flight. */
[data-testid="stChatInput"] textarea:disabled {{ opacity: 0.6; }}

/* --- Sidebar -------------------------------------------------------------- */
[data-testid="stSidebar"] {{
  background: var(--card);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.25rem; }}
[data-testid="stSidebar"] .stMarkdown p {{ font-size: 0.8125rem; }}
.sidebar-title {{
  font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--faint); font-weight: 650; margin: 0 0 0.6rem 0;
}}
.scheme-nav {{ display: flex; flex-direction: column; gap: 2px; }}
.scheme-item {{
  display: flex; align-items: center; gap: 0.55rem;
  padding: 0.45rem 0.6rem; border-radius: 8px;
  font-size: 0.8125rem; color: var(--ink); font-weight: 450;
  transition: background-color .14s ease;
}}
.scheme-item:hover {{ background: var(--green-tint); color: var(--green-dark); }}
.scheme-glyph {{ color: var(--green); font-size: 0.6875rem; width: 0.85rem; }}
/* The persisted disclaimer strip. Short by design -- the long form lives in the
   header, and repeating a 60-word paragraph twice made the sidebar the most
   visually dominant column on the page. */
.disclaimer-strip {{
  font-size: 0.75rem; color: var(--muted); line-height: 1.55;
  border-top: 1px solid var(--border); padding-top: 0.8rem; margin-top: 0.5rem;
}}
.disclaimer-strip strong {{ color: var(--ink); font-weight: 600; }}

/* --- Status notices ------------------------------------------------------- */
/* Muted surfaces rather than Streamlit's saturated yellow/red banners. */
[data-testid="stAlert"] {{ border-radius: 10px; font-size: 0.875rem;
  border: 1px solid var(--border); }}
[data-testid="stAlert"][data-baseweb="warning"] {{
  background: var(--warn-tint); color: var(--warn-ink);
  border-color: rgba(181, 71, 8, 0.18);
}}
[data-testid="stAlert"][data-baseweb="error"] {{
  background: var(--danger-tint); color: var(--danger);
  border-color: rgba(180, 35, 24, 0.18);
}}
[data-testid="stAlert"][data-baseweb="success"],
[data-testid="stAlert"][data-baseweb="info"] {{
  background: var(--green-tint); color: var(--green-dark);
  border-color: rgba(15, 157, 88, 0.18);
}}

/* --- Misc ----------------------------------------------------------------- */
/* Spinner: small and green instead of the default grey. */
[data-testid="stSpinner"] i {{ border-top-color: var(--green) !important; }}
.loading-copy {{ font-size: 0.9375rem; color: var(--ink); line-height: 1.6; }}
/* Expander (developer chunks) inherits card treatment. */
[data-testid="stExpander"] {{
  border: 1px solid var(--border); border-radius: 10px;
  background: var(--card);
}}
details summary {{ font-size: 0.8125rem; color: var(--muted); }}
/* Belt and braces against horizontal scroll at any width. Nothing in this app
   should ever scroll sideways. */
html, body, .stApp {{ overflow-x: hidden; max-width: 100vw; }}
* {{ min-width: 0; }}
[data-testid="stExpander"] pre, .stApp pre {{
  white-space: pre-wrap !important; word-break: break-word !important;
}}

/* --- Footer --------------------------------------------------------------- */
/* The full PRD disclaimer, once, at the bottom of the page. Not in the sidebar,
   which must stay navigation; not in the header, which must stay a wordmark.
   Here it cannot be cropped away and it cannot crowd the conversation. */
.page-footer {{
  margin-top: 2.25rem; padding-top: 1.1rem;
  border-top: 1px solid var(--border);
}}
.footer-note {{
  font-size: 0.8125rem; color: var(--ink); font-weight: 600;
  margin-bottom: 0.35rem;
}}
.footer-legal {{
  font-size: 0.75rem; color: var(--muted); line-height: 1.6;
  max-width: 78ch;
}}
.footer-legal strong {{ color: var(--ink); font-weight: 600; }}

/* "Read as: <resolved>" -- memory folded a pronoun into a scheme. Quiet, or it
   reads as an error. */
.resolve-note {{
  font-size: 0.75rem; color: var(--faint); font-style: italic;
  margin: -0.35rem 0 0.35rem 0;
}}

/* --- Responsive ----------------------------------------------------------- */
@media (max-width: 768px) {{
  .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important;
    padding-bottom: 7.5rem !important; }}
  .stApp h1 {{ font-size: 1.55rem !important; }}
  .user-bubble {{ max-width: 90%; }}
  .assistant-card {{ padding: 0.85rem 0.95rem; }}
  .answer-text {{ font-size: 0.9375rem; }}
  [data-testid="stChatInput"] > div {{ padding: 0 0.75rem; }}
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    min-width: 100%; flex: 1 1 100%;
  }}
  .footer-legal {{ max-width: none; }}
}}
@media (max-width: 480px) {{
  .block-container {{ padding-left: 0.75rem !important;
    padding-right: 0.75rem !important; }}
  .user-bubble {{ max-width: 100%; }}
  .source-row {{ gap: 0.25rem 0.45rem; }}
}}
</style>""", unsafe_allow_html=True)


# --- Header ------------------------------------------------------------------


def render_header() -> None:
    """Top bar: wordmark left, tagline under it, 'Facts-only' pill right."""
    st.markdown(
        f"""
<div class="app-header">
  <div class="app-header-left">
    <div class="app-brand">
      <span class="brand-dot"></span>HDFC Scheme Facts
    </div>
    <div class="app-tagline">Mutual fund facts, simply explained</div>
  </div>
  <span class="facts-pill">Facts-only</span>
</div>
""",
        unsafe_allow_html=True,
    )


HEADER_CSS = f"""<style>
.app-header {{
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 1rem; padding-bottom: 1.1rem; margin-bottom: 1.4rem;
  border-bottom: 1px solid var(--border);
}}
.app-brand {{
  font-size: 1.0625rem; font-weight: 650; color: var(--ink);
  letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.45rem;
}}
.brand-dot {{
  width: 8px; height: 8px; border-radius: 50%;
  background: {GREEN}; display: inline-block;
}}
.app-tagline {{ font-size: 0.8125rem; color: var(--muted); margin-top: 0.2rem; }}
.facts-pill {{
  background: {GREEN_TINT}; color: {GREEN_DARK};
  border: 1px solid rgba(15, 157, 88, 0.22);
  border-radius: 999px; padding: 0.28rem 0.7rem;
  font-size: 0.75rem; font-weight: 600; white-space: nowrap;
}}
@media (max-width: 480px) {{
  .app-header {{ flex-direction: column; gap: 0.6rem; }}
  .facts-pill {{ align-self: flex-start; }}
}}
</style>"""


def inject_header_css() -> None:
    st.markdown(HEADER_CSS, unsafe_allow_html=True)


# --- Sidebar -----------------------------------------------------------------


def render_sidebar(show_sources: bool, memory_len: int,
                   memory_window: int,
                   chunks_label: str) -> "tuple":
    """Compact navigation column. Returns (show_sources, clear_requested).

    The debug controls live in a collapsed expander: they are genuinely useful
    for a class demo, but "10 chunks per question" was previously the most
    prominent text in the sidebar, ahead of the disclaimer itself.

    This function draws and reads widget state. It never touches memory or the
    transcript; clearing them is the caller's decision.
    """
    st.markdown('<div class="sidebar-title">HDFC Schemes</div>',
                unsafe_allow_html=True)

    items = "".join(
        f'<div class="scheme-item"><span class="scheme-glyph">'
        f'{SCHEME_GLYPHS[name]}</span><span>{html.escape(name)}</span></div>'
        for name in SIDEBAR_SCHEMES
    )
    st.markdown(f'<div class="scheme-nav">{items}</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="disclaimer-strip"><strong>Facts-only.</strong> '
        f'No investment advice. Facts come only from 5 public scheme pages and '
        f'every answer links its source.</div>',
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

    Short on the landing page, omitted once a conversation exists so the cards
    do not compete with the transcript above them.
    """
    if not first_visit:
        return
    st.markdown('<div class="suggest-label">Try one of these</div>',
                unsafe_allow_html=True)


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


def render_trusted_markdown(text: str, css_class: str = "footer-legal") -> None:
    """Render markdown for a REPO CONSTANT only (e.g. DISCLAIMER, WELCOME).

    Escapes first, then re-enables just `**bold**`. Accepting this as an
    escape-then-reopen pattern only works because the input is a literal in this
    file. Never call it with model output, retrieval text, or anything a user
    typed -- those go through st.markdown() with unsafe_allow_html left off, or
    through _escape().
    """
    body = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", _escape(text))
    st.markdown(f'<div class="{css_class}">{body}</div>',
                unsafe_allow_html=True)


def render_user_bubble(text: str) -> None:
    """Right-aligned tinted pill. The wrapper container is the chat message body;
    the bubble itself is the styled inline block."""
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
    """Compact, non-repeating citation.

    The full URL is preserved verbatim -- it is the same string the pipeline
    produced, attached to the same anchor. Only the presentation changed: the
    long address is no longer printed inline, which is what removed the
    horizontal overflow. `title` keeps it available on hover.
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
    meta = _escape(scheme_label) if scheme_label else "Groww"
    st.markdown(
        f'<div class="source-row">'
        f'<span class="source-label">Source</span>'
        f'<a class="source-link" href="{safe_url}" target="_blank" '
        f'rel="noopener noreferrer" title="{safe_url}">View source ↗</a>'
        f'<span class="source-meta">· {meta}</span>'
        f'</div>'
        + (
            f'<div class="source-date">Last updated from sources: '
            f'{_escape(last_updated)}</div>'
            if last_updated
            else ""
        ),
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
    """Polite loading copy. No pipeline jargon (embedding, vector search, ...)."""
    with st.container(border=True):
        st.markdown(
            '<div class="loading-copy">Finding the relevant scheme facts…</div>',
            unsafe_allow_html=True,
        )


def render_status(kind: str, message: str) -> None:
    """Status strip under an answer: refused / not-found / provider error."""
    classes = {
        "refused": ("notice-refused", "Refused before the model was called. "
                    "Nothing left your machine."),
        "not_found": ("notice-notfound", "Not found in the indexed pages."),
        "error": ("notice-error", "The model provider did not respond in time. "
                  "This is not a problem with your question or with the indexed "
                  "data — try again."),
    }
    css, text = classes.get(kind, ("notice-generic", message))
    st.markdown(
        f'<div class="notice {css}">{_escape(text)}</div>',
        unsafe_allow_html=True,
    )


STATUS_CSS = f"""<style>
.notice {{
  margin-top: 0.6rem; padding: 0.55rem 0.8rem;
  border-radius: 8px; font-size: 0.8125rem; line-height: 1.5;
  border: 1px solid var(--border); color: var(--muted); background: #FBFBFC;
}}
.notice-refused {{ background: var(--green-tint);
  border-color: rgba(15,157,88,0.2); color: var(--green-dark); }}
.notice-notfound {{ background: var(--warn-tint);
  border-color: rgba(181,71,8,0.18); color: var(--warn-ink); }}
.notice-error {{ background: var(--danger-tint);
  border-color: rgba(180,35,24,0.18); color: var(--danger); }}
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

    Two lines with different jobs: the sub-line says what you can ask, the intro
    says what this is and where the facts come from. Merging them into one
    paragraph was what made the old landing block read as a paragraph of
    developer copy.
    """
    st.markdown(
        '<div class="page-h1">HDFC Scheme Facts</div>'
        '<div class="page-sub">Ask about expense ratio, exit load, SIP minimum, '
        'benchmark, riskometer and more.</div>',
        unsafe_allow_html=True,
    )
    if intro:
        render_trusted_markdown(intro, css_class="page-intro")


PAGE_CSS = f"""<style>
.page-h1 {{
  font-size: 2rem; font-weight: 650; color: var(--ink);
  letter-spacing: -0.02em; line-height: 1.2; margin-bottom: 0.35rem;
}}
.page-sub {{ font-size: 0.9375rem; color: var(--muted); margin-bottom: 1.35rem;
  line-height: 1.55; }}
.page-intro {{
  font-size: 0.875rem; color: var(--muted); line-height: 1.62;
  max-width: 68ch; margin-bottom: 1.6rem;
}}
.page-intro strong {{ color: var(--ink); font-weight: 600; }}
.suggest-label {{
  font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--faint); font-weight: 650; margin: 0 0 0.6rem 0;
}}
@media (max-width: 768px) {{
  .page-h1 {{ font-size: 1.55rem; }}
  .page-sub {{ font-size: 0.875rem; }}
}}
</style>"""


def inject_page_css() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def render_persistent_note(text: str) -> None:
    st.markdown(f'<div class="source-date" style="margin-top:1.5rem">'
                f'{_escape(text)}</div>', unsafe_allow_html=True)


__all__ = [
    "GREEN", "GREEN_DARK", "GREEN_TINT", "INK", "MUTED", "BORDER", "CARD",
    "PAGE", "SIDEBAR_SCHEMES", "SCHEME_GLYPHS",
    "inject_css", "inject_header_css", "inject_status_css", "inject_page_css",
    "render_header", "render_sidebar", "render_user_bubble", "render_source_row",
    "render_answer_card", "render_loading", "render_status", "render_chunks",
    "render_page_heading", "render_suggestion_label", "render_resolution_note",
    "render_footer_note", "render_trusted_markdown",
]
