"""Poll ABC News (Just In) and log changes to each article's own headline.

State and the change log live in --data-dir:

  state.json     {"schema": 3, "articles": {<content_id>: {
                     "url", "first_seen", "last_seen",
                     "list":   {title, synopsis, published, updated,             # from the Just In listing
                                image_id, image_url, image_alt},
                     "detail": {title, teaser, short,                            # from articledetail
                                description, published, updated, version, type,
                                production_unit, authors, byline, topics, keywords}}}}
  changes.jsonl  one line per detected headline change (append-only): detected_at, content_id, url,
                 field, old, new, list_updated, version (null for list.* fields)
  snapshots/     with --snapshot only: a timestamped copy of state.json from every run (debugging)

Each run lists the --limit newest articles (default and maximum 250), fetches every listed article's
detail JSON (rate-limited, see DETAIL_CONCURRENCY) and compares the tracked headline fields with the
stored ones. Every listed article is polled each run because an article's own dates don't reliably
change when its headline does; use --limit to poll fewer. An article is forgotten once it is missing from
a full-size (250 item) listing, because it has aged out of the window; a --limit run, or a listing that
comes back short, forgets nothing. Headlines and metadata only: article body text is never stored.

Usage: uv run poller.py [--data-dir data] [--limit 250] [--dry-run] [--snapshot]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

BASE_URL = "https://www.abc.net.au"

# Two internal ABC front-end JSON endpoints. They are undocumented and can change without notice, so
# a response that isn't the expected shape makes the script fail loudly.
#   LISTING_URL ("collectionstories"): what the "load more" button on https://www.abc.net.au/news/justin
#     calls (documentId 10719976 is the Just In collection; at most 250 items). The same articles and
#     titles as the public RSS feed, plus a lastUpdated date, in about a third of the bytes.
#   DETAIL_URL ("articledetail", one article per request): the JSON an article page embeds in
#     __NEXT_DATA__, without the page. It has the full title, teaser, short title and the CMS version.
LISTING_URL = f"{BASE_URL}/news-web/api/loader/collectionstories"
LISTING_PARAMS = {"name": "PaginationArticles", "documentId": "10719976"}
DETAIL_URL = f"{BASE_URL}/news-web/api/loader/articledetail"
DEFAULT_LIMIT = 250  # the listing's maximum (about 3.5 days of Just In)

DEFAULT_REPO = "newsdiffbot/abcnewsheadlines"  # used in the User-Agent when run outside Actions
SCHEMA = 3

# At most DETAIL_CONCURRENCY requests are in flight, and each slot waits DETAIL_DELAY_S after its
# request: 2 requests per second at most, so 250 articles take about 2 minutes.
DETAIL_CONCURRENCY = 2
DETAIL_DELAY_S = 1.0

# (section, field) pairs whose changes are logged. Other stored fields are refreshed silently.
TRACKED_FIELDS = (("list", "title"), ("detail", "title"), ("detail", "teaser"), ("detail", "short"))

QUOTE_AND_DASH_TABLE = str.maketrans("‘’“”–— ", "''\"\"-- ")
BREAKING_PREFIX = re.compile(r"^(?:breaking|live)\s*[:\-]\s*", re.IGNORECASE)


def normalise(title: str) -> str:
    """Comparison key: ignores case, whitespace, quote style and a Breaking:/Live: prefix."""
    title = " ".join(title.translate(QUOTE_AND_DASH_TABLE).split())
    return BREAKING_PREFIX.sub("", title).casefold()


def clean_text(text: str | None) -> str | None:
    """NBSP to space, trim indentation, collapse spaces; keep line breaks."""
    if not text:
        return None
    lines = (" ".join(line.split()) for line in text.splitlines())  # split() also treats NBSP as space
    return "\n".join(line for line in lines if line) or None


def image_id_and_url(src: str | None) -> tuple[str | None, str | None]:
    """Images come in many crops (?impolicy=...&cropH=...). Drop the query; the last path
    segment (a 32-char hash) is the unique id."""
    if not src:
        return None, None
    parts = urlsplit(src)
    return (parts.path.rstrip("/").rsplit("/", 1)[-1] or None), f"{parts.scheme}://{parts.netloc}{parts.path}"


def descriptor_text(node) -> str:
    """Concatenate the text leaves of a rendered 'descriptor' tree (listing synopsis, article byline)."""
    if isinstance(node, dict):
        own = node.get("content", "") if node.get("type") == "text" else ""
        return own + "".join(descriptor_text(v) for v in node.values() if isinstance(v, (dict, list)))
    if isinstance(node, list):
        return "".join(descriptor_text(v) for v in node)
    return ""


def get_in(obj, *path):
    """obj[path[0]][path[1]]... with None for anything missing or not a dict."""
    for key in path:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def user_agent() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)
    return f"abcdiff/0.1 (unofficial headline-change monitor; non-commercial; +https://github.com/{repo})"


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temp file so a crash can't leave a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, "utf-8")
    tmp.replace(path)


def parse_listing(data) -> list[tuple[str, str, dict]]:
    """[(article_id, url, list_fields)] from the listing JSON, newest first.
    Raises ValueError if the shape isn't what we expect."""
    items = data.get("collection") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("listing response has no 'collection' list")
    parsed = []
    for item in items:
        article_id, link, title = item.get("id"), item.get("link"), clean_text(item.get("title"))
        if not (article_id and link and title):
            raise ValueError(f"listing item missing id/link/title: {str(item)[:120]}")
        dates = item.get("dates") or {}
        image = item.get("image") or {}
        image_id, image_url = image_id_and_url(image.get("imgSrc"))
        parsed.append((str(article_id), urljoin(BASE_URL, link), {
            "title": " ".join(title.split()),
            "synopsis": clean_text(descriptor_text(item.get("synopsis"))),
            "published": dates.get("firstPublished"),
            "updated": dates.get("lastUpdated"),
            "image_id": image_id,
            "image_url": image_url,
            "image_alt": clean_text(image.get("altText")),
        }))
    return parsed


def parse_detail(data, article_id: str) -> dict:
    """The `detail` fields from an articledetail response. Raises ValueError if it isn't the
    article we asked for or has no headline."""
    if not isinstance(data, dict) or str(data.get("id")) != article_id:
        raise ValueError(f"articledetail response is not for article {article_id}")
    titles = get_in(data, "dataLayerDocumentPrepared", "document", "title") or {}
    alt_titles = data.get("titleAlt") or {}
    dateline = data.get("datelinePrepared") or {}
    version = get_in(data, "headTagsCMSPrepared", "version")
    byline = get_in(data, "newsBylinePrepared", "byline")  # null, or a rendered descriptor tree
    published = dateline.get("publishedDate") or get_in(data, "headTagsArticlePrepared", "published")
    detail = {
        "title": clean_text(titles.get("title") or alt_titles.get("lg") or data.get("title")),
        "teaser": clean_text(titles.get("teaser") or alt_titles.get("md")),
        "short": clean_text(titles.get("short") or alt_titles.get("sm")),
        "description": clean_text(data.get("synopsis")),
        "published": published,
        # updatedDate is absent when the article was never edited; headTagsSocialPrepared.updated isn't
        "updated": get_in(data, "headTagsSocialPrepared", "updated") or dateline.get("updatedDate") or published,
        "version": str(version) if version is not None else None,
        "type": data.get("articleType"),
        "production_unit": data.get("productionUnit"),
        "authors": [a["name"] for a in get_in(data, "newsBylinePrepared", "authors") or [] if a.get("name")],
        "byline": clean_text(byline if isinstance(byline, str) else descriptor_text(byline)),
        "topics": [t["topic"] for t in get_in(data, "relatedTopicsPrepared", "topics") or [] if t.get("topic")],
        "keywords": get_in(data, "headTagsArticlePrepared", "keywords") or [],
    }
    if not (detail["title"] or detail["teaser"]):
        raise ValueError(f"no headline in articledetail response for {article_id}")
    return detail


async def fetch_listing(client: httpx.AsyncClient, limit: int) -> list[tuple[str, str, dict]]:
    params = {**LISTING_PARAMS, "offset": 0, "size": limit, "total": limit}
    response = await client.get(LISTING_URL, params=params)
    response.raise_for_status()
    return parse_listing(response.json())


async def fetch_details(client: httpx.AsyncClient, article_ids: list[str]) -> dict[str, dict | Exception]:
    """Fetch every article's detail within the rate limit. An article that fails (network, HTTP status,
    unexpected shape) maps to its exception instead of its detail."""
    slots = asyncio.Semaphore(DETAIL_CONCURRENCY)

    async def fetch_one(article_id: str) -> dict | Exception:
        async with slots:
            try:
                response = await client.get(DETAIL_URL, params={"id": article_id})
                response.raise_for_status()
                result = parse_detail(response.json(), article_id)
            except Exception as exc:  # skip this article only; the caller reports it
                result = exc
            await asyncio.sleep(DETAIL_DELAY_S)
            return result

    results = await asyncio.gather(*(fetch_one(article_id) for article_id in article_ids))
    return dict(zip(article_ids, results, strict=True))


def find_changes(article_id: str, article: dict, section: str, new_fields: dict, detected_at: str) -> list[dict]:
    """Tracked fields of `section` whose stored value differs from `new_fields`. A first sighting has no
    stored value, so it never counts as a change."""
    old_fields = article.get(section)
    if not old_fields:
        return []
    list_fields = new_fields if section == "list" else article["list"]
    changes = []
    for tracked_section, field in TRACKED_FIELDS:
        old, new = old_fields.get(field), new_fields.get(field)
        if tracked_section == section and old and new and normalise(old) != normalise(new):
            changes.append({
                "detected_at": detected_at,
                "content_id": article_id,
                "url": article["url"],
                "field": f"{section}.{field}",
                "old": old,
                "new": new,
                "list_updated": list_fields["updated"],
                "version": new_fields.get("version") if section == "detail" else None,
            })
    return changes


def load_articles(state_path: Path) -> dict[str, dict] | None:
    """The stored articles, or None (a cold start) if there is no usable state file."""
    try:
        state = json.loads(state_path.read_text("utf-8"))
        if state.get("schema") != SCHEMA:
            return None
        return state["articles"]
    except (OSError, ValueError, KeyError, AttributeError):
        return None


async def run(args: argparse.Namespace) -> int:
    state_path = args.data_dir / "state.json"
    changes_path = args.data_dir / "changes.jsonl"
    stored_articles = load_articles(state_path)
    cold_start = stored_articles is None
    articles: dict[str, dict] = stored_articles or {}

    async with httpx.AsyncClient(headers={"User-Agent": user_agent()}, timeout=30, follow_redirects=True) as client:
        try:
            listing = await fetch_listing(client, args.limit)
        except (httpx.HTTPError, ValueError) as exc:  # JSON decode errors are ValueErrors
            print(f"listing failed: {exc}", file=sys.stderr)
            return 1

        now = int(time.time())
        detected_at = datetime.fromtimestamp(now, UTC).isoformat()
        changes: list[dict] = []
        dirty = False
        listed_ids: list[str] = []

        for article_id, url, list_fields in listing:
            if article_id in listed_ids:
                continue
            listed_ids.append(article_id)
            article = articles.get(article_id)
            if article is None:
                articles[article_id] = {"url": url, "first_seen": now, "last_seen": now,
                                        "list": list_fields, "detail": None}
                dirty = True
                continue
            article["last_seen"] = now
            if article["list"] != list_fields:
                changes += find_changes(article_id, article, "list", list_fields, detected_at)
                article["list"] = list_fields
                dirty = True

        fetched = failed = 0
        for article_id, result in (await fetch_details(client, listed_ids)).items():
            if isinstance(result, Exception):
                failed += 1
                print(f"detail {article_id} failed: {type(result).__name__}: {result}", file=sys.stderr)
                continue
            fetched += 1
            article = articles[article_id]
            if article["detail"] != result:
                changes += find_changes(article_id, article, "detail", result, detected_at)
                article["detail"] = result
                dirty = True

    if len(listed_ids) >= DEFAULT_LIMIT:  # a full-size listing is the whole window: what is missing has aged out
        listed = set(listed_ids)
        for article_id in [i for i in articles if i not in listed]:
            del articles[article_id]
            dirty = True

    print(f"{'COLD START ' if cold_start else ''}articles={len(articles)} (listed {len(listed_ids)}) "
          f"changes={len(changes)} details: fetched={fetched} failed={failed}")
    for change in changes:
        print(f"CHANGED {change['content_id']} {change['field']}\n  - {change['old']}\n  + {change['new']}")

    if not args.dry_run:
        state_json = json.dumps({"schema": SCHEMA, "articles": articles}, ensure_ascii=False, separators=(",", ":"))
        write_text_atomic(state_path, state_json)
        if args.snapshot:
            snapshot_name = f"state-{datetime.fromtimestamp(now, UTC):%Y%m%dT%H%M%SZ}.json"
            write_text_atomic(args.data_dir / "snapshots" / snapshot_name, state_json)
        if changes:
            with changes_path.open("a", encoding="utf-8") as file:
                file.writelines(json.dumps(change, ensure_ascii=False) + "\n" for change in changes)

    if output_path := os.environ.get("GITHUB_OUTPUT"):  # lets the workflow decide whether to save the cache
        with open(output_path, "a", encoding="utf-8") as file:
            file.write(f"dirty={str(dirty and not args.dry_run).lower()}\n")
    if listed_ids and not fetched:  # the detail endpoint is down or changed shape: fail loudly (state is still saved)
        print("every articledetail request failed", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"how many of the newest articles to list and poll (default and maximum {DEFAULT_LIMIT})")
    parser.add_argument("--dry-run", action="store_true", help="don't write state or changes")
    parser.add_argument("--snapshot", action="store_true",
                        help="debugging: also keep a timestamped copy of state.json (data/snapshots/) each run")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
