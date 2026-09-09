# app/matcher.py
"""
Matching engine.

The job: given a list of live URLs (from a sitemap) and a list of dead URLs,
find the live URL that is most plausibly "the same page" for each dead one.

The signal that matters most is the distinctive noun in the path -- a company
name, a product name, a person. Those words are rare across the corpus, so
inverse document frequency does the heavy lifting: a shared "feedonomics"
counts for far more than a shared "marketing". Everything else (character
similarity, adjacent word pairs, path context) is there to break ties.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from urllib.parse import parse_qs, unquote, urlparse

WORD_RE = re.compile(r"[a-z0-9]+")

# How many leading candidates get the expensive character-level comparison.
SHORTLIST = 80

# Words too common in URL paths to say anything about which page is meant.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "s", "t", "the",
    "their", "this", "to", "up", "was", "what", "when", "which", "who", "why",
    "will", "with", "you", "your", "amp", "html", "index", "php", "page",
}

# Path segments that mark a URL as an archive or wrapper rather than an article.
ARCHIVE_SEGMENTS = {
    "tag", "tags", "topic", "topics", "category", "categories", "author",
    "authors", "page", "search", "archive", "archives", "label",
}

WRAPPER_SEGMENTS = {"amp"}


# --------------------------------------------------------------------------
# text handling
# --------------------------------------------------------------------------

def stem(word: str) -> str:
    """Very light suffix trim. Enough to join singular/plural, no more."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def words_of(text: str, keep_stopwords: bool = False) -> list[str]:
    out = []
    for raw in WORD_RE.findall(text.lower()):
        if not keep_stopwords and raw in STOPWORDS:
            continue
        out.append(stem(raw))
    return out


def bigrams(seq: list[str]) -> set[tuple[str, str]]:
    return set(zip(seq, seq[1:]))


# --------------------------------------------------------------------------
# URL handling
# --------------------------------------------------------------------------

@dataclass
class UrlParts:
    url: str            # the URL after unwrapping
    original: str       # exactly what came in
    segments: list[str]
    slug: str
    is_archive: bool
    notes: list[str] = field(default_factory=list)


def unwrap(raw: str) -> UrlParts:
    """
    Reduce a dead URL to the content path it was really pointing at.

      /wp-login.php?...&redirect_to=<url>   ->  <url>
      /amp/news/foo/                        ->  /news/foo/
      /news/foo/?utm_source=x               ->  /news/foo/
    """
    original = (raw or "").strip()
    url = original
    notes: list[str] = []

    parsed = urlparse(url)

    # A login or redirect wrapper carries the real destination in the query.
    query = parse_qs(parsed.query)
    for key in ("redirect_to", "redirect", "url", "u", "next", "return"):
        if key in query and query[key] and query[key][0].strip():
            target = unquote(query[key][0]).strip()
            if "/" in target:
                url = target
                parsed = urlparse(url)
                notes.append("unwrapped redirect target")
                break

    segments = [s for s in parsed.path.split("/") if s]

    while segments and segments[0].lower() in WRAPPER_SEGMENTS:
        segments.pop(0)
        notes.append("stripped /amp/")

    # Trailing paginators such as /page/2 tell us nothing about the topic.
    while len(segments) >= 2 and segments[-2].lower() == "page" and segments[-1].isdigit():
        segments = segments[:-2]
        notes.append("dropped pagination")

    if segments and segments[-1].lower().endswith((".php", ".html", ".htm")):
        segments[-1] = segments[-1].rsplit(".", 1)[0]

    is_archive = any(s.lower() in ARCHIVE_SEGMENTS for s in segments[:-1]) or (
        bool(segments) and segments[0].lower() in ARCHIVE_SEGMENTS
    )

    slug = segments[-1].lower() if segments else ""
    clean = f"{parsed.scheme or 'https'}://{parsed.netloc}/" + "/".join(segments)

    return UrlParts(
        url=clean,
        original=original,
        segments=[s.lower() for s in segments],
        slug=slug,
        is_archive=is_archive,
        notes=notes,
    )


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

@dataclass
class Doc:
    url: str
    segments: list[str]
    slug: str
    slug_words: list[str]
    all_words: set[str]
    slug_bigrams: set[tuple[str, str]]
    depth: int
    mass: float = 0.0


class Index:
    """Inverted index over the live URLs, weighted by inverse document frequency."""

    def __init__(self, live_urls: list[str]):
        self.docs: list[Doc] = []
        seen: set[str] = set()

        for url in live_urls:
            url = (url or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)

            parsed = urlparse(url)
            segments = [s.lower() for s in parsed.path.split("/") if s]
            slug = segments[-1] if segments else parsed.netloc.lower()
            slug_words = words_of(slug)
            all_words = set(slug_words)
            for seg in segments[:-1]:
                all_words.update(words_of(seg))

            self.docs.append(
                Doc(
                    url=url,
                    segments=segments,
                    slug=slug,
                    slug_words=slug_words,
                    all_words=all_words,
                    slug_bigrams=bigrams(slug_words),
                    depth=len(segments),
                )
            )

        self.n = max(len(self.docs), 1)
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(self.docs):
            for word in doc.all_words:
                self.postings[word].append(i)

        self.df = {word: len(ids) for word, ids in self.postings.items()}
        self.idf = {
            word: math.log(1.0 + self.n / count) for word, count in self.df.items()
        }
        self.default_idf = math.log(1.0 + self.n)

        for doc in self.docs:
            doc.mass = sum(self.idf.get(w, self.default_idf) for w in doc.all_words) or 1.0

        self.by_slug: dict[str, int] = {}
        for i, doc in enumerate(self.docs):
            self.by_slug.setdefault(doc.slug, i)

        # Shallow URLs are the section and hub pages; archives fall back to these.
        self.hubs = sorted(
            range(len(self.docs)), key=lambda i: (self.docs[i].depth, len(self.docs[i].url))
        )[:400]

        # Rare words are almost always proper nouns -- brands, products, people.
        self.rare_cutoff = max(2, int(self.n * 0.0008))
        self.common_cutoff = max(50, int(self.n * 0.12))

    def weight(self, word: str) -> float:
        return self.idf.get(word, self.default_idf)

    # -- candidate generation ---------------------------------------------

    def candidates(self, query_words: list[str], cap: int = 3000) -> set[int]:
        ranked = sorted(set(query_words), key=lambda w: -self.weight(w))
        picked: set[int] = set()
        for word in ranked:
            ids = self.postings.get(word)
            if not ids:
                continue
            if len(ids) > self.common_cutoff and picked:
                continue  # a word this common only adds noise once we have leads
            picked.update(ids)
            if len(picked) >= cap:
                break
        if not picked:
            for word in ranked:
                picked.update(self.postings.get(word, ())[:cap])
        return picked

    # -- scoring -----------------------------------------------------------

    def rough(self, q_set: set[str], q_mass: float, doc: Doc) -> float:
        """Cheap first pass: how much IDF mass do the two share? Used to shortlist."""
        shared = q_set & doc.all_words
        if not shared:
            return 0.0
        s_mass = sum(self.weight(w) for w in shared)
        coverage = s_mass / q_mass
        precision = s_mass / doc.mass
        return coverage + 0.5 * precision

    def score(self, q: UrlParts, q_words: list[str], q_bigrams, doc: Doc) -> tuple[float, list[str]]:
        q_set = set(q_words)
        shared = q_set & doc.all_words
        if not shared:
            return 0.0, []

        q_mass = sum(self.weight(w) for w in q_set) or 1.0
        s_mass = sum(self.weight(w) for w in shared)

        coverage = s_mass / q_mass          # how much of the dead URL is explained
        precision = s_mass / doc.mass       # guards against long pages swallowing short queries
        harmonic = (
            2 * coverage * precision / (coverage + precision)
            if (coverage + precision)
            else 0.0
        )

        seq = SequenceMatcher(None, q.slug, doc.slug).ratio()

        shared_bigrams = q_bigrams & doc.slug_bigrams
        bigram_ratio = len(shared_bigrams) / max(len(q_bigrams), 1) if q_bigrams else 0.0

        base = 0.40 * coverage + 0.14 * precision + 0.16 * harmonic
        base += 0.18 * seq + 0.12 * bigram_ratio

        # The named-entity rule: a shared rare word is the strongest evidence
        # there is, so reward it explicitly rather than letting it average out.
        rare = sorted(
            (w for w in shared if self.df.get(w, 0) <= self.rare_cutoff),
            key=lambda w: -self.weight(w),
        )
        if rare:
            base += min(0.10, 0.045 * len(rare))

        # Same section of the site is weak corroboration.
        if q.segments[:-1] and doc.segments[:-1]:
            if set(q.segments[:-1]) & set(doc.segments[:-1]):
                base += 0.02

        if q.is_archive:
            # A tag or topic page wants another listing page, not one article.
            doc_archive = [s for s in doc.segments[:-1] if s in ARCHIVE_SEGMENTS]
            q_archive = [s for s in q.segments[:-1] if s in ARCHIVE_SEGMENTS]
            if doc_archive:
                base += 0.16 if set(doc_archive) & set(q_archive) else 0.10
            elif doc.depth <= 2:
                base += 0.06

        score = min(100.0, round(base * 100, 1))
        return score, rare[:4] or sorted(shared, key=lambda w: -self.weight(w))[:4]

    # -- public API --------------------------------------------------------

    def match(self, dead_url: str, alternates: int = 2) -> dict:
        q = unwrap(dead_url)

        if not self.docs:
            return {
                "url": "",
                "score": 0.0,
                "method": "no live URLs indexed",
                "matched_on": "",
                "notes": q.notes,
                "alternates": [],
            }

        exact = self.by_slug.get(q.slug)
        if exact is not None:
            doc = self.docs[exact]
            doc_is_archive = any(s in ARCHIVE_SEGMENTS for s in doc.segments[:-1])
            # A tag page should land on a tag page, not on an article that
            # happens to share its one-word slug.
            if not q.is_archive or doc_is_archive or doc.depth <= 2:
                return {
                    "url": doc.url,
                    "score": 100.0,
                    "method": "exact slug",
                    "matched_on": "",
                    "notes": q.notes,
                    "alternates": [],
                }

        q_words = words_of(q.slug)
        for seg in q.segments[:-1]:
            if seg not in ARCHIVE_SEGMENTS:
                q_words.extend(words_of(seg))
        q_words = list(dict.fromkeys(q_words))
        q_bigrams = bigrams(words_of(q.slug))

        q_set = set(q_words)
        q_mass = sum(self.weight(w) for w in q_set) or 1.0

        shortlist = []
        for i in self.candidates(q_words):
            r = self.rough(q_set, q_mass, self.docs[i])
            if r > 0:
                shortlist.append((r, i))
        shortlist.sort(reverse=True)

        scored: list[tuple[float, int, list[str]]] = []
        for _, i in shortlist[:SHORTLIST]:
            s, why = self.score(q, q_words, q_bigrams, self.docs[i])
            if s > 0:
                scored.append((s, i, why))

        if not scored:
            # Nothing shares a word. Send it to the closest hub so the row is
            # never left without a destination.
            hub = self._fallback_hub(q)
            return {
                "url": hub,
                "score": 0.0,
                "method": "section fallback",
                "matched_on": "",
                "notes": q.notes + ["no shared words with any live URL"],
                "alternates": [],
            }

        scored.sort(key=lambda x: (-x[0], len(self.docs[x[1]].url)))
        best_score, best_i, best_why = scored[0]

        method = "distinctive word" if best_why and self.df.get(best_why[0], 99) <= self.rare_cutoff else "word overlap"
        if q.is_archive:
            method = "archive to hub"

        return {
            "url": self.docs[best_i].url,
            "score": best_score,
            "method": method,
            "matched_on": ", ".join(best_why),
            "notes": q.notes,
            "alternates": [
                {"url": self.docs[i].url, "score": s}
                for s, i, _ in scored[1 : 1 + alternates]
            ],
        }

    def _fallback_hub(self, q: UrlParts) -> str:
        first = q.segments[0] if q.segments else ""
        for i in self.hubs:
            if self.docs[i].segments and self.docs[i].segments[0] == first:
                return self.docs[i].url
        shallowest = self.docs[self.hubs[0]] if self.hubs else self.docs[0]
        parsed = urlparse(shallowest.url)
        return f"{parsed.scheme}://{parsed.netloc}/"
