"""Post the headline changes found by poller.py to Bluesky (run it after poller.py).

Only full-title changes are posted (`detail.title` in changes.jsonl). The scripts share files in --data-dir:
  changes.jsonl  written by poller.py. A change is pending until its key "<content_id>:<version>" (the
                 CMS revision, independent of when we polled) is recorded in posted.json.
  state.json     written by poller.py, read here for the article's description and first-seen time.
  posted.json    written by this script: posted change keys and each article's thread.

The first change to an article starts a thread, in the style of nytdiff.bsky.social:
  root   the headline as it was before the edit. It quotes ABC's own post for the article (found in the
         public feed of @news.abc.net.au by the article id in its link) or, if there is none, is a link
         card with a thumbnail rendered from the headline. Replies are switched off (an empty threadgate,
         as on ABC's own posts); quoting is left on.
  reply  "Change in Headline" and an image of the word diff, one per change, each replying to the last
         post in the thread. The alt text is "Before: ...\\nAfter: ...".
If posted.json doesn't know an article's thread (say the cache was lost), the account's own recent posts
are searched for it, and the thread is continued rather than started again.
Every change is posted, oldest first, however long ago it was detected. An article's edits always go out
in order: if one fails, the article's later edits wait for the next run.

--dry-run never logs in or writes posted.json. It prints what it would post as markdown and writes, into
--preview-dir, the images and would-post.json: every request exactly as it would be sent (with a placeholder
DID and placeholder references to its own earlier posts), for checking the threadgates, embeds, links and
attachments. (It can only look for existing threads if BSKY_USERNAME is set.)

Environment: BSKY_USERNAME and BSKY_PASSWORD (an app password).
Usage: uv run poster.py [--data-dir data] [--dry-run] [--preview-dir preview]
"""

import argparse
import base64
import difflib
import hashlib
import io
import itertools
import json
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple, TypedDict
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import font_source_serif_pro
from atproto import Client, models
from atproto.exceptions import AtProtocolError
from PIL import Image, ImageDraw, ImageFont

from poller import user_agent, write_text_atomic

BlobRef = models.blob_ref.BlobRef
IpldLink = models.blob_ref.IpldLink

TIMEZONE = ZoneInfo("Australia/Sydney")  # for the times shown on images

TITLE_FIELD = "detail.title"
REPLY_TEXT = "Change in Headline"
POST_LANGUAGES = ["en-AU"]
MAX_POST_LENGTH = 290  # Bluesky allows 300 characters

PUBLIC_API = "https://public.api.bsky.app"  # reads that need no login
OFFICIAL_HANDLE = "news.abc.net.au"
FEED_PAGE_SIZE = 100
OFFICIAL_FEED_PAGES = 5  # about 5 days of the official account, more than the 3.5 days the poller covers
OWN_FEED_PAGES = 5
ARTICLE_ID_AT_END = re.compile(r"/(\d{6,})$")

POST_LOG_SCHEMA = 1


# Images: Source Serif Pro, with an inline word diff like NYTdiff's.

FONT_DIR = Path(font_source_serif_pro.font_directory)
REGULAR_FONT = str(FONT_DIR / "SourceSerifPro-Regular.ttf")
BOLD_FONT = str(FONT_DIR / "SourceSerifPro-Bold.ttf")

UNCHANGED, REMOVED, ADDED = "=", "-", "+"  # the kinds of token in a word diff (the symbols are difflib-style)
TOKEN_PATTERN = re.compile(r"\s+|\w+(?:[’'\-]\w+)*|[^\w\s]")  # spaces, words (don't, witch-hunt), punctuation
MOSTLY_DIFFERENT = 0.5  # below this word similarity, show the old headline struck out, then the new one

PAPER, INK = (247, 244, 236), (28, 28, 30)
HIGHLIGHT = {REMOVED: (248, 187, 198), ADDED: (160, 244, 202)}
TEXT_COLOUR = {UNCHANGED: INK, REMOVED: (120, 20, 38), ADDED: (10, 70, 40)}

DIFF_WIDTH, DIFF_PADDING = 800, 36
DIFF_FONT_SIZES = (36, 32, 28, 24)  # tried in turn: the first at which the text fits in MAX_DIFF_LINES is used
MAX_DIFF_LINES = 8
# Proportions of the font size, which came from a 36 px font
LINE_HEIGHT_RATIO, TEXT_OFFSET_RATIO, STRIKE_WIDTH_RATIO = 14 / 9, 1 / 18, 1 / 12
HIGHLIGHT_MARGIN_RATIO, HIGHLIGHT_BOTTOM_RATIO, HIGHLIGHT_RADIUS_RATIO = 1 / 9, 2 / 9, 1 / 7

THUMB_WIDTH, THUMB_HEIGHT, THUMB_PADDING = 1200, 630, 80
THUMB_FONT_SIZES = range(76, 35, -4)  # 76, 72 ... 36: the largest at which the headline fits is used
THUMB_LINE_SPACING = 1.2
THUMB_CAPTION_SPACE = 70  # room kept clear for the caption under the headline
THUMB_CAPTION_SIZE = 30


class Token(NamedTuple):
    kind: str  # UNCHANGED, REMOVED or ADDED
    text: str


class PlacedToken(NamedTuple):
    kind: str
    text: str
    x: float
    width: float


class DiffImage(NamedTuple):
    png: bytes
    width: int
    height: int
    font_size: int


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def diff_tokens(old: str, new: str) -> list[Token]:
    old_tokens, new_tokens = TOKEN_PATTERN.findall(old.strip()), TOKEN_PATTERN.findall(new.strip())
    old_words = [t for t in old_tokens if t.strip()]
    new_words = [t for t in new_tokens if t.strip()]
    if difflib.SequenceMatcher(None, old_words, new_words, autojunk=False).ratio() < MOSTLY_DIFFERENT:
        removed = [Token(REMOVED, t) for t in old_tokens]
        added = [Token(ADDED, t) for t in new_tokens]
        return [*removed, Token(UNCHANGED, " "), *added]

    tokens: list[Token] = []
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    for op, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if op == "equal":
            tokens += [Token(UNCHANGED, t) for t in old_tokens[old_start:old_end]]
            continue
        for kind, changed in ((REMOVED, old_tokens[old_start:old_end]), (ADDED, new_tokens[new_start:new_end])):
            text = "".join(changed)
            if not text.strip():
                continue
            if tokens and tokens[-1].text.strip():
                tokens.append(Token(UNCHANGED, " "))  # a plain space before the changed words
            tokens += [Token(kind, t) for t in TOKEN_PATTERN.findall(text.strip())]
            if not changed[-1].strip():
                tokens.append(Token(UNCHANGED, " "))  # the changed words had a trailing space; keep it unhighlighted
    return tokens


def split_to_fit(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    """The text as pieces no wider than max_width. Only a word too long for a whole line needs more than one."""
    if font.getlength(text) <= max_width:
        return [text]
    pieces, current = [], ""
    for character in text:
        if current and font.getlength(current + character) > max_width:
            pieces.append(current)
            current = character
        else:
            current += character
    return [*pieces, current]


def wrap_tokens(tokens: list[Token], fonts: dict[str, ImageFont.FreeTypeFont],
                max_width: float) -> list[list[PlacedToken]]:
    lines: list[list[PlacedToken]] = []
    line: list[PlacedToken] = []
    x = 0.0
    for token in tokens:
        font = fonts[token.kind]
        for text in split_to_fit(token.text, font, max_width):  # a word wider than the image is broken up
            width = font.getlength(text)
            is_word = bool(text.strip())
            if line and is_word and x + width > max_width:
                lines.append(line)
                line, x = [], 0.0
            if not line and not is_word:
                continue  # no leading space on a wrapped line
            line.append(PlacedToken(token.kind, text, x, width))
            x += width
    lines.append(line)
    return lines


def render_diff_image(old: str, new: str) -> DiffImage:
    """The word diff on paper-coloured ground. The font shrinks, down to 24 px, when the text is long."""
    tokens = diff_tokens(old, new)
    for size in DIFF_FONT_SIZES:
        regular, bold = ImageFont.truetype(REGULAR_FONT, size), ImageFont.truetype(BOLD_FONT, size)
        fonts = {UNCHANGED: regular, REMOVED: regular, ADDED: bold}
        lines = wrap_tokens(tokens, fonts, DIFF_WIDTH - 2 * DIFF_PADDING)
        if len(lines) <= MAX_DIFF_LINES:
            break  # otherwise the smallest size is kept and the image simply grows taller

    line_height = round(size * LINE_HEIGHT_RATIO)
    margin = round(size * HIGHLIGHT_MARGIN_RATIO)
    bottom = round(size * HIGHLIGHT_BOTTOM_RATIO)
    height = 2 * DIFF_PADDING + len(lines) * line_height - (line_height - size) // 2
    image = Image.new("RGB", (DIFF_WIDTH, height), PAPER)
    draw = ImageDraw.Draw(image)

    for row, line in enumerate(lines):
        top = DIFF_PADDING + row * line_height
        while line and not line[-1].text.strip():
            line = line[:-1]
        for kind, run in itertools.groupby(line, key=lambda placed: placed.kind):  # one box per run
            run = list(run)
            if kind != UNCHANGED:
                left = DIFF_PADDING + run[0].x - margin
                right = DIFF_PADDING + run[-1].x + run[-1].width + margin
                draw.rounded_rectangle([left, top + margin, right, top + line_height - bottom],
                                       radius=round(size * HIGHLIGHT_RADIUS_RATIO), fill=HIGHLIGHT[kind])
        for placed in line:
            left = DIFF_PADDING + placed.x
            draw.text((left, top + round(size * TEXT_OFFSET_RATIO)), placed.text, font=fonts[placed.kind],
                      fill=TEXT_COLOUR[placed.kind])
            if placed.kind == REMOVED and placed.text.strip():
                strike_y = top + line_height // 2 - 1
                draw.line([left, strike_y, left + placed.width, strike_y], fill=TEXT_COLOUR[REMOVED],
                          width=max(2, round(size * STRIKE_WIDTH_RATIO)))
    return DiffImage(encode_png(image), image.width, image.height, size)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        for piece in split_to_fit(word, font, max_width):  # a word wider than the image is broken up
            candidate = f"{current} {piece}".strip()
            if current and font.getlength(candidate) > max_width:
                lines.append(current)
                current = piece
            else:
                current = candidate
    return [*lines, current]


def fit_thumbnail_text(headline: str) -> tuple[list[str], int]:
    """The wrapped headline and the largest font size (down to 36 px) at which it fits above the caption.
    A headline too long even then is cut off with an ellipsis."""
    max_width = THUMB_WIDTH - 2 * THUMB_PADDING
    max_height = THUMB_HEIGHT - 2 * THUMB_PADDING - THUMB_CAPTION_SPACE
    for size in THUMB_FONT_SIZES:
        font = ImageFont.truetype(BOLD_FONT, size)
        lines = wrap_text(headline, font, max_width)
        if len(lines) * size * THUMB_LINE_SPACING <= max_height:
            return lines, size
    kept = lines[: int(max_height // (size * THUMB_LINE_SPACING))]
    last = kept[-1]
    while last and font.getlength(f"{last}…") > max_width:
        last = last[:-1]
    kept[-1] = f"{last.rstrip()}…"
    return kept, size


def render_thumbnail(headline: str, seen_at: str) -> bytes:
    """1200x630 (the shape link cards are cropped to): the headline on white, and when we saw it."""
    lines, size = fit_thumbnail_text(headline)
    image = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(BOLD_FONT, size)
    for row, line in enumerate(lines):
        draw.text((THUMB_PADDING, THUMB_PADDING + int(row * size * THUMB_LINE_SPACING)), line, font=font,
                  fill=(24, 24, 27))
    rule_y = THUMB_HEIGHT - THUMB_PADDING - 52
    draw.rectangle([THUMB_PADDING, rule_y, THUMB_PADDING + 90, rule_y + 4], fill=(24, 24, 27))
    caption_font = ImageFont.truetype(REGULAR_FONT, THUMB_CAPTION_SIZE)
    draw.text((THUMB_PADDING, THUMB_HEIGHT - THUMB_PADDING - THUMB_CAPTION_SIZE), f"Headline as seen {seen_at}",
              font=caption_font, fill=(110, 110, 118))
    return encode_png(image)


def format_time(timestamp: float) -> str:
    moment = datetime.fromtimestamp(timestamp, TIMEZONE)
    clock = f"{moment:%-I:%M}{moment:%p}".replace("AM", "am").replace("PM", "pm")
    return f"{moment:%-d %b %Y}, {clock} {moment:%Z}"


# Bluesky

class PostRef(TypedDict):
    uri: str
    cid: str  # the post's content hash (not an ABC article id)


class ExistingThread(TypedDict):
    root: PostRef
    last: PostRef
    replies: dict[str, str]  # uri of each diff reply, by its alt text "Before: ...\nAfter: ..."


_TID_ALPHABET = "234567abcdefghijklmnopqrstuvwxyz"
_last_tid_micros = 0


def new_record_key() -> str:
    """A TID, the record key Bluesky clients use for posts: 13 base32 characters made of a microsecond
    timestamp (kept increasing within this process) and 10 random bits."""
    global _last_tid_micros
    _last_tid_micros = max(int(time.time() * 1_000_000), _last_tid_micros + 1)
    value = (_last_tid_micros << 10) | random.getrandbits(10)
    return "".join(_TID_ALPHABET[(value >> shift) & 31] for shift in range(60, -1, -5))


def new_client(base_url: str | None = None) -> Client:
    client = Client(base_url)
    client.request.add_additional_header("User-Agent", user_agent())
    return client


def strong_ref(post: PostRef) -> models.ComAtprotoRepoStrongRef.Main:
    return models.ComAtprotoRepoStrongRef.Main(uri=post["uri"], cid=post["cid"])


def diff_alt_text(change: dict) -> str:
    return f"Before: {change['old']}\nAfter: {change['new']}"


def article_id_from_url(url: str) -> str | None:
    parts = urlsplit(url)
    match = ARTICLE_ID_AT_END.search(parts.path)
    return match.group(1) if match and (parts.hostname or "").endswith("abc.net.au") else None


def feed_posts(get_page: Callable[..., models.AppBskyFeedGetAuthorFeed.Response], max_pages: int) -> Iterator:
    """The posts of an author feed, newest first, following the cursor for at most max_pages pages."""
    cursor = None
    for _ in range(max_pages):
        page = get_page(cursor=cursor)
        for item in page.feed:
            if not item.reason:  # skip reposts
                yield item.post
        cursor = page.cursor
        if not cursor:
            return


def fetch_official_posts(client: Client) -> dict[str, PostRef]:
    """ABC's own Bluesky post for each article, by article id, from the account's newest posts."""
    get_page = partial(client.get_author_feed, OFFICIAL_HANDLE, filter="posts_no_replies", limit=FEED_PAGE_SIZE)
    posts: dict[str, PostRef] = {}
    for post in feed_posts(get_page, OFFICIAL_FEED_PAGES):
        card = getattr(post.embed, "external", None)  # the link card of the post
        article_id = article_id_from_url(card.uri) if card else None
        if article_id:
            posts.setdefault(article_id, PostRef(uri=post.uri, cid=post.cid))
    return posts


def fetch_own_threads(client: Client, actor: str,
                      article_ids_by_official_uri: dict[str, str]) -> dict[str, ExistingThread]:
    """Rebuild each article's thread from the account's newest posts. A thread's root is a quote of the
    official post or a link card, which says which article it is about."""
    get_page = partial(client.get_author_feed, actor, filter="posts_with_replies", limit=FEED_PAGE_SIZE)
    threads: dict[str, ExistingThread] = {}  # by the uri of the thread's root
    article_ids: dict[str, str] = {}  # root uri to article id
    for post in feed_posts(get_page, OWN_FEED_PAGES):
        reply = post.record.reply
        root = PostRef(uri=reply.root.uri, cid=reply.root.cid) if reply else PostRef(uri=post.uri, cid=post.cid)
        thread = threads.setdefault(root["uri"], ExistingThread(root=root, last=PostRef(uri=post.uri, cid=post.cid),
                                                                replies={}))
        for image in getattr(post.embed, "images", None) or []:
            thread["replies"].setdefault(image.alt, post.uri)  # newest post first, so the newest wins
        if not reply:
            card = getattr(post.embed, "external", None)
            quoted = getattr(getattr(post.embed, "record", None), "uri", None)
            article_id = article_id_from_url(card.uri) if card else article_ids_by_official_uri.get(quoted)
            if article_id:
                article_ids[root["uri"]] = article_id
    by_article: dict[str, ExistingThread] = {}
    for root_uri, article_id in article_ids.items():  # newest root first
        by_article.setdefault(article_id, threads[root_uri])
    return by_article


def web_url(post_uri: str) -> str:
    did, _, rkey = post_uri.removeprefix("at://").split("/")
    return f"https://bsky.app/profile/{did}/post/{rkey}"


def shorten(text: str) -> str:
    return text if len(text) <= MAX_POST_LENGTH else text[: MAX_POST_LENGTH - 1] + "…"


# Records: one set of builders serves a real post and the dry run's record of it

def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def blob_cid(data: bytes) -> str:
    """The CID Bluesky gives an uploaded blob: CIDv1, raw codec, SHA-256, base32."""
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + hashlib.sha256(data).digest()
    return "b" + base64.b32encode(raw).decode().lower().rstrip("=")


def quote_embed(post: PostRef) -> models.AppBskyEmbedRecord.Main:
    return models.AppBskyEmbedRecord.Main(record=strong_ref(post))


def link_card_embed(url: str, title: str, description: str, thumbnail: BlobRef) -> models.AppBskyEmbedExternal.Main:
    card = models.AppBskyEmbedExternal.External(uri=url, title=title, description=description, thumb=thumbnail)
    return models.AppBskyEmbedExternal.Main(external=card)


def diff_image_embed(alt: str, image: BlobRef, width: int, height: int) -> models.AppBskyEmbedImages.Main:
    aspect_ratio = models.AppBskyEmbedDefs.AspectRatio(width=width, height=height)
    return models.AppBskyEmbedImages.Main(images=[models.AppBskyEmbedImages.Image(
        alt=alt, image=image, aspect_ratio=aspect_ratio)])


def root_commit(did: str, rkey: str, text: str, embed) -> models.ComAtprotoRepoApplyWrites.Data:
    """The root post and its threadgate in one commit, so the thread is never open to replies, even
    briefly. allow=[] means nobody can reply; quoting stays allowed."""
    created_at = now_iso()
    post = models.AppBskyFeedPost.Record(created_at=created_at, text=text, langs=POST_LANGUAGES, embed=embed)
    threadgate = models.AppBskyFeedThreadgate.Record(
        created_at=created_at, post=f"at://{did}/app.bsky.feed.post/{rkey}", allow=[])
    writes = [
        models.ComAtprotoRepoApplyWrites.Create(collection="app.bsky.feed.post", rkey=rkey, value=post),
        models.ComAtprotoRepoApplyWrites.Create(collection="app.bsky.feed.threadgate", rkey=rkey, value=threadgate),
    ]
    return models.ComAtprotoRepoApplyWrites.Data(repo=did, writes=writes)


def reply_commit(did: str, text: str, embed, thread: dict) -> models.ComAtprotoRepoCreateRecord.Data:
    reply_to = models.AppBskyFeedPost.ReplyRef(root=strong_ref(thread["root"]), parent=strong_ref(thread["last"]))
    post = models.AppBskyFeedPost.Record(
        created_at=now_iso(), text=text, langs=POST_LANGUAGES, embed=embed, reply=reply_to)
    return models.ComAtprotoRepoCreateRecord.Data(repo=did, collection="app.bsky.feed.post", record=post)


def card_description(article: dict) -> str:
    return (article.get("detail") or {}).get("description") or ""


def thumbnail_png(change: dict, article: dict) -> bytes:
    return render_thumbnail(change["old"], format_time(article.get("first_seen") or time.time()))


# Publishing: for real, or as a dry run that only reports

class Publisher(ABC):
    """Decides what a thread's root and replies contain. Subclasses supply the primitives that differ:
    LivePublisher sends them to Bluesky, DryRunPublisher records them."""

    def __init__(self):
        self.public = new_client(PUBLIC_API)
        self._official_posts: dict[str, PostRef] | None = None
        self._own_threads: dict[str, ExistingThread] | None = None

    def official_posts(self) -> dict[str, PostRef]:
        if self._official_posts is None:
            try:
                self._official_posts = fetch_official_posts(self.public)
            except AtProtocolError as exc:
                print(f"warning: could not read the official feed ({exc}); using a link card", file=sys.stderr)
                self._official_posts = {}
        return self._official_posts

    def official_post(self, article_id: str) -> PostRef | None:
        return self.official_posts().get(article_id)

    def find_existing_thread(self, article_id: str) -> ExistingThread | None:
        """The article's thread among the account's own posts, for when posted.json doesn't know it.
        A failure to read the feed propagates, so the article waits for the next run instead of getting
        a second thread."""
        if self._own_threads is None:
            actor = self.own_actor()
            official_ids = {post["uri"]: article for article, post in self.official_posts().items()}
            self._own_threads = fetch_own_threads(self.public, actor, official_ids) if actor else {}
        return self._own_threads.get(article_id)

    def post_root(self, change: dict, article: dict) -> PostRef:
        """Start the article's thread with the headline as it was before this change."""
        headline = shorten(change["old"])
        official = self.official_post(change["content_id"])
        if official and self.can_quote(official["uri"]):
            print(f"**Root post**, quote of the official post {web_url(official['uri'])}; "
                  f"replies disabled, quotes allowed\n> {headline}\n")
            try:
                return self.create_root(change, headline, quote_embed(official))
            except AtProtocolError as exc:
                print(f"warning: quote failed ({exc}); falling back to a link card", file=sys.stderr)
        print(f"**Root post**, link card (no usable official post); replies disabled, quotes allowed\n> {headline}\n")
        thumbnail = self.upload(thumbnail_png(change, article), f"{change['content_id']}-thumb.png")
        card = link_card_embed(change["url"], headline, card_description(article), thumbnail)
        return self.create_root(change, headline, card)

    def post_diff_reply(self, change: dict, thread: dict, number: int) -> PostRef:
        """Reply to the thread's last post with an image of this change."""
        diff = render_diff_image(change["old"], change["new"])
        print(f"**Reply**, `{REPLY_TEXT}` with an image of the diff ({diff.width}x{diff.height}); alt text:\n"
              f"> Before: {change['old']}  \n> After: {change['new']}\n")
        image = self.upload(diff.png, f"{change['content_id']}-{number}-diff.png")
        embed = diff_image_embed(diff_alt_text(change), image, diff.width, diff.height)
        return self.create_reply(change, REPLY_TEXT, embed, thread)

    @abstractmethod
    def own_actor(self) -> str | None:
        """The account's DID or handle, or None if unknown."""

    @abstractmethod
    def can_quote(self, post_uri: str) -> bool:
        """Whether the post can be quoted."""

    @abstractmethod
    def upload(self, png: bytes, name: str) -> BlobRef:
        """Upload an image (named for the dry run's files) and return the reference to embed."""

    @abstractmethod
    def create_root(self, change: dict, text: str, embed) -> PostRef:
        """Create the root post (and its threadgate)."""

    @abstractmethod
    def create_reply(self, change: dict, text: str, embed, thread: dict) -> PostRef:
        """Create a reply to the thread's last post."""

    @abstractmethod
    def finish(self) -> None:
        """Called once after all articles are handled."""


class LivePublisher(Publisher):
    def __init__(self):
        super().__init__()
        self._client: Client | None = None
        self._login_error: Exception | None = None

    @property
    def client(self) -> Client:
        """Logs in on first use, so a run with nothing to post never logs in. A failed login isn't retried."""
        if self._login_error:
            raise self._login_error
        if self._client is None:
            try:
                if "BSKY_USERNAME" not in os.environ or "BSKY_PASSWORD" not in os.environ:
                    raise RuntimeError("BSKY_USERNAME and BSKY_PASSWORD must be set")
                client = new_client()
                client.login(os.environ["BSKY_USERNAME"], os.environ["BSKY_PASSWORD"])
                self._client = client
            except Exception as exc:
                self._login_error = exc
                raise
        return self._client

    def own_actor(self) -> str:
        return self.client.me.did

    def can_quote(self, post_uri: str) -> bool:
        """False if the post is gone, has quoting disabled, or its author has blocked us."""
        try:
            posts = self.client.get_posts([post_uri]).posts
        except AtProtocolError as exc:  # the check failed: try the quote anyway
            print(f"warning: could not check {post_uri}: {exc}", file=sys.stderr)
            return True
        if not posts:
            return False
        post = posts[0]
        author_viewer = post.author.viewer
        quoting_disabled = post.viewer and post.viewer.embedding_disabled
        blocked = author_viewer and (author_viewer.blocked_by or author_viewer.blocking)
        return not (quoting_disabled or blocked)

    def upload(self, png: bytes, name: str) -> BlobRef:
        return self.client.upload_blob(png).blob

    def create_root(self, change: dict, text: str, embed) -> PostRef:
        commit = root_commit(self.client.me.did, new_record_key(), text, embed)
        result = self.client.com.atproto.repo.apply_writes(commit)
        root = PostRef(uri=result.results[0].uri, cid=result.results[0].cid)
        print(f"Posted: {web_url(root['uri'])}\n")
        return root

    def create_reply(self, change: dict, text: str, embed, thread: dict) -> PostRef:
        created = self.client.com.atproto.repo.create_record(reply_commit(self.client.me.did, text, embed, thread))
        print(f"Posted: {web_url(created.uri)}\n")
        return PostRef(uri=created.uri, cid=created.cid)

    def finish(self) -> None:
        """Nothing to flush: every post was sent as it was created."""


DRY_RUN_DID = "did:plc:" + "a" * 24  # stands in for the account's DID; nothing about the real account is written
DRY_RUN_CID = "bafyreidryrunplaceholder"  # stands in for the content hash of a post that is never created


class DryRunPublisher(Publisher):
    """Posts nothing and never logs in. It builds the same records as LivePublisher and writes them, with the
    images and a check-friendly summary, to <preview_dir>/would-post.json."""

    def __init__(self, preview_dir: Path):
        super().__init__()
        self.preview_dir = preview_dir
        self.operations: list[dict] = []
        self._attachments: list[dict] = []

    def own_actor(self) -> str | None:
        return os.environ.get("BSKY_USERNAME")

    def can_quote(self, post_uri: str) -> bool:
        return True  # checking needs a login

    def upload(self, png: bytes, name: str) -> BlobRef:
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        (self.preview_dir / name).write_bytes(png)
        cid = blob_cid(png)
        self._attachments.append({"file": name, "mimeType": "image/png", "size": len(png), "cid": cid})
        print(f"Image: `{self.preview_dir / name}`\n")
        return BlobRef(mime_type="image/png", size=len(png), ref=IpldLink(link=cid))

    def record_operation(self, change: dict, step: str, method: str, commit, created_uri: str) -> PostRef:
        self.operations.append({
            "article_id": change["content_id"],
            "article_url": change["url"],
            "step": step,
            "xrpc": {"method": method, "input": models.get_model_as_dict(commit)},
            "attachments": self._attachments,
            "returns": {"uri": created_uri, "cid": DRY_RUN_CID},  # what the thread's next post will point at
        })
        self._attachments = []
        return PostRef(uri=created_uri, cid=DRY_RUN_CID)

    def create_root(self, change: dict, text: str, embed) -> PostRef:
        rkey = new_record_key()
        commit = root_commit(DRY_RUN_DID, rkey, text, embed)
        return self.record_operation(change, "root", "com.atproto.repo.applyWrites", commit,
                                     f"at://{DRY_RUN_DID}/app.bsky.feed.post/{rkey}")

    def create_reply(self, change: dict, text: str, embed, thread: dict) -> PostRef:
        commit = reply_commit(DRY_RUN_DID, text, embed, thread)
        return self.record_operation(change, "reply", "com.atproto.repo.createRecord", commit,
                                     f"at://{DRY_RUN_DID}/app.bsky.feed.post/{new_record_key()}")

    def finish(self) -> None:
        report = {
            "dry_run": True,
            "generated_at": now_iso(),
            "placeholders": {"did": DRY_RUN_DID, "cid": DRY_RUN_CID,
                             "note": "Stand-ins: no login happens and nothing is posted. Official post "
                                     "references and article links are real."},
            "operations": self.operations,
        }
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        path = self.preview_dir / "would-post.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"Every post as it would be sent, as JSON: `{path}`\n")


# What has been posted

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return default


def change_key(change: dict) -> str:
    return f"{change['content_id']}:{change['version']}"


class PostLog:
    """posted.json: the changes already posted and each article's thread, so replies can continue it. A
    read-only log (dry run) still updates in memory. Nothing is ever forgotten: changes.jsonl never forgets
    a change either, and both grow by only a few lines a day."""

    def __init__(self, path: Path, writable: bool):
        data = read_json(path, {})
        if data.get("schema") != POST_LOG_SCHEMA:
            data = {"schema": POST_LOG_SCHEMA, "posted": {}, "threads": {}}
        self.path, self.writable, self.data = path, writable, data
        self.modified = False
        self._recovered_replies: dict[str, dict[str, str]] = {}  # threads found among the account's own posts

    def is_recorded(self, change: dict) -> bool:
        return change_key(change) in self.data["posted"]

    def record_posted(self, change: dict, uri: str, now: float) -> None:
        self.data["posted"][change_key(change)] = {"uri": uri, "at": now}
        self.modified = True

    def thread(self, article_id: str) -> dict | None:
        return self.data["threads"].get(article_id)

    def set_thread(self, article_id: str, root: PostRef, last: PostRef, now: float) -> dict:
        thread = {"root": root, "last": last, "at": now}
        self.data["threads"][article_id] = thread
        self.modified = True
        return thread

    def adopt_existing(self, article_id: str, existing: ExistingThread, now: float) -> None:
        """Continue a thread that was found among the account's own posts."""
        self.set_thread(article_id, existing["root"], existing["last"], now)
        self._recovered_replies[article_id] = existing["replies"]

    def find_recovered_reply(self, change: dict) -> str | None:
        """The uri of the reply that already posted this change, if the thread was recovered and has it."""
        return self._recovered_replies.get(change["content_id"], {}).get(diff_alt_text(change))

    def record_reply(self, thread: dict, change: dict, reply: PostRef, now: float) -> None:
        thread.update(last=reply, at=now)
        self.record_posted(change, reply["uri"], now)

    def save(self) -> None:
        if self.writable:
            write_text_atomic(self.path, json.dumps(self.data, ensure_ascii=False, separators=(",", ":")))


def read_changes(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    except OSError:
        return []


def pending_changes(changes: list[dict], log: PostLog) -> list[dict]:
    """The title changes not yet recorded, oldest first."""
    titles = (c for c in changes if c["field"] == TITLE_FIELD and c.get("version"))
    return [c for c in sorted(titles, key=lambda c: c["detected_at"]) if not log.is_recorded(c)]


def publish_change(change: dict, article: dict, number: int, log: PostLog, publisher: Publisher, now: float) -> None:
    """Post one change as a reply in its article's thread. The thread comes from posted.json, else from the
    account's own posts, else it is started with this change's "before" headline."""
    article_id = change["content_id"]
    print(f"### {change['url']}\n")
    if log.thread(article_id) is None:
        existing = publisher.find_existing_thread(article_id)
        if existing:
            print("_Continuing the thread found among the account's own posts._\n")
            log.adopt_existing(article_id, existing, now)
        else:
            root = publisher.post_root(change, article)
            log.set_thread(article_id, root, root, now)
        log.save()
    if uri := log.find_recovered_reply(change):  # the recovered thread already has this change
        log.record_posted(change, uri, now)
        log.save()
        return
    thread = log.thread(article_id)
    reply = publisher.post_diff_reply(change, thread, number)
    log.record_reply(thread, change, reply, now)
    log.save()


def run(args: argparse.Namespace) -> int:
    now = time.time()
    log = PostLog(args.data_dir / "posted.json", writable=not args.dry_run)
    articles = read_json(args.data_dir / "state.json", {}).get("articles", {})
    publisher = DryRunPublisher(args.preview_dir) if args.dry_run else LivePublisher()

    pending = pending_changes(read_changes(args.data_dir / "changes.jsonl"), log)
    print(f"## Bluesky poster{' (dry run: nothing is posted, no login)' if args.dry_run else ''}\n")
    print(f"{len(pending)} title change(s) to post.\n")

    failures = 0
    failed_articles: set[str] = set()
    numbers: dict[str, int] = {}  # how many of an article's changes were handled in this run
    for change in pending:  # oldest first
        article_id = change["content_id"]
        if article_id in failed_articles:
            continue  # keep the article's edits in order: they wait for the next run
        numbers[article_id] = numbers.get(article_id, 0) + 1
        try:
            publish_change(change, articles.get(article_id) or {}, numbers[article_id], log, publisher, now)
        except Exception as exc:  # one failing article must not stop the others
            failures += 1
            failed_articles.add(article_id)
            print(f"error posting {article_id}: {exc}", file=sys.stderr)
    log.save()
    publisher.finish()

    if output_path := os.environ.get("GITHUB_OUTPUT"):  # tells the workflow whether to save the cache
        with open(output_path, "a", encoding="utf-8") as file:
            file.write(f"dirty={str(log.writable and log.modified).lower()}\n")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true",
                        help="no login, no posts, no posted.json; print what would be posted and render previews")
    parser.add_argument("--preview-dir", type=Path, default=Path("preview"), help="dry run: where the images go")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
