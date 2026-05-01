from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import unescape
from math import log
from pathlib import Path
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")
FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)
LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]+|\([^()]*\))*\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?:[^()]+|\([^()]*\))*\)")
HTML_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
    "im",
    "ive",
    "cant",
    "not",
    "help",
    "need",
    "want",
    "get",
}


@dataclass(frozen=True)
class TextBlock:
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Article:
    path: Path
    relative_path: str
    company: str
    section: str
    title: str
    text: str
    tokens: tuple[str, ...]
    blocks: tuple[TextBlock, ...]


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def strip_frontmatter(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    return text[match.end() :] if match else text


def clean_inline_markdown(text: str) -> str:
    text = IMAGE_RE.sub(" ", text)
    text = LINK_RE.sub(r"\1", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("\u00a0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return unescape(text).strip()


def normalize_block(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = HEADING_RE.sub("", line)
        line = clean_inline_markdown(line)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def split_blocks(text: str) -> list[str]:
    blocks = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            if stripped:
                return clean_inline_markdown(stripped)
    return ""


def infer_title(text: str, file_path: Path) -> str:
    for line in text.splitlines()[:20]:
        if line.lower().startswith("title:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if value:
                return clean_inline_markdown(value)
    heading = first_heading(text)
    return heading or file_path.stem


def infer_section(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return parts[0].replace("-", " ").title() + " / " + parts[1].replace("-", " ").title()
    if parts:
        return parts[0].replace("-", " ").title()
    return ""


def infer_company(relative_path: str) -> str:
    root = Path(relative_path).parts[0].lower()
    if root == "hackerrank":
        return "HackerRank"
    if root == "claude":
        return "Claude"
    if root == "visa":
        return "Visa"
    return ""


class CorpusIndex:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.articles = self._load_articles()
        self.document_frequency = Counter()
        for article in self.articles:
            for token in set(article.tokens):
                self.document_frequency[token] += 1
        total_documents = max(len(self.articles), 1)
        self.idf = {
            token: log((1 + total_documents) / (1 + document_count)) + 1.0
            for token, document_count in self.document_frequency.items()
        }

    def _load_articles(self) -> list[Article]:
        articles: list[Article] = []
        for path in sorted(self.data_root.rglob("*.md")):
            raw_text = path.read_text(encoding="utf-8")
            body = strip_frontmatter(raw_text)
            relative_path = path.relative_to(self.data_root).as_posix()
            title = infer_title(raw_text, path)
            company = infer_company(relative_path)
            section = infer_section(relative_path)
            blocks: list[TextBlock] = []
            normalized_blocks: list[str] = []
            for block in split_blocks(body):
                cleaned = normalize_block(block)
                if not cleaned:
                    continue
                block_tokens = tuple(tokenize(cleaned))
                blocks.append(TextBlock(text=cleaned, tokens=block_tokens))
                normalized_blocks.append(cleaned)
            combined_text = "\n\n".join(normalized_blocks)
            tokens = tuple(tokenize(f"{title}\n{combined_text}\n{relative_path}"))
            articles.append(
                Article(
                    path=path,
                    relative_path=relative_path,
                    company=company,
                    section=section,
                    title=title,
                    text=combined_text,
                    tokens=tokens,
                    blocks=tuple(blocks),
                )
            )
        return articles

    def search(
        self,
        query: str,
        company: str | None = None,
        hints: Iterable[str] = (),
        limit: int = 5,
    ) -> list[tuple[Article, float]]:
        query_tokens = tokenize(query)
        query_token_counts = Counter(query_tokens)
        hint_tokens = set(tokenize(" ".join(hints)))
        scored: list[tuple[Article, float]] = []
        for article in self.articles:
            if company and article.company and article.company.lower() != company.lower():
                continue
            article_token_set = set(article.tokens)
            overlap = article_token_set.intersection(query_token_counts)
            score = 0.0
            for token in overlap:
                score += self.idf.get(token, 1.0)
            if hint_tokens:
                hint_overlap = article_token_set.intersection(hint_tokens)
                score += 1.5 * len(hint_overlap)
            title_tokens = set(tokenize(article.title))
            score += 1.2 * len(title_tokens.intersection(query_token_counts))
            path_tokens = set(tokenize(article.relative_path))
            score += 0.7 * len(path_tokens.intersection(query_token_counts))
            if "release-notes" in article.relative_path:
                score -= 1.2
            # Prefer overview / index pages for information-style queries
            title_lower = article.title.lower()
            if article.relative_path.endswith("index.md") or "overview" in title_lower:
                score += 0.9
            # Titles that look like explainers should be preferred
            if title_lower.startswith(("what", "how", "why", "about", "overview")):
                score += 0.8
            # If the user mentions interviews, prefer Hackerrank interview docs
            q_lower = query.lower()
            if "interview" in q_lower:
                if "interview" in article.relative_path or "interview" in title_lower or "hackerrank" in article.relative_path:
                    score += 1.5
            if company and article.company and article.company.lower() == company.lower():
                score += 0.75
            if score > 0.0:
                scored.append((article, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def best_article(self, query: str, company: str | None = None, hints: Iterable[str] = ()) -> tuple[Article | None, float]:
        # Try a company-scoped search first. If the best scoped match is weak,
        # allow a fallback to an unscoped search so that tickets with incorrect
        # `Company` fields can still retrieve relevant docs (e.g., interview
        # issues that belong to HackerRank but claim `Claude` as company).
        matches = self.search(query=query, company=company, hints=hints, limit=1)
        best_scoped = matches[0] if matches else (None, 0.0)
        if best_scoped[1] < 1.0:
            # perform an unscoped search and pick the stronger result
            unscoped = self.search(query=query, company=None, hints=hints, limit=1)
            best_unscoped = unscoped[0] if unscoped else (None, 0.0)
            # compare scores
            if best_unscoped[1] > best_scoped[1] + 0.2:
                return best_unscoped
        return best_scoped


def best_block(article: Article, query: str) -> str:
    if not article.blocks:
        return ""
    query_tokens = set(tokenize(query))

    def score_block(block: TextBlock) -> float:
        if not block.tokens:
            return 0.0
        overlap = query_tokens.intersection(block.tokens)
        score = len(overlap)
        if any(line.startswith("-") or line[:1].isdigit() for line in block.text.splitlines()):
            score += 0.5
        if len(block.text) < 80:
            score += 0.2
        return score

    ranked = sorted(article.blocks, key=score_block, reverse=True)
    return ranked[0].text if ranked else article.blocks[0].text


def response_excerpt(article: Article, query: str, max_sentences: int = 4) -> str:
    block_text = best_block(article, query)
    if not block_text:
        block_text = article.text
    block_text = block_text.replace("\n", " ")
    block_text = MULTISPACE_RE.sub(" ", block_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", block_text)
    if len(sentences) <= max_sentences:
        return block_text.strip()
    return " ".join(sentences[:max_sentences]).strip()