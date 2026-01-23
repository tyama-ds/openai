# coding: utf-8
"""Sentence-level fact checker with web search and LLM verification.

Workflow
1) Split text into sentences.
2) Extract claims per sentence.
3) Search the web and collect referenced URLs as evidence.
4) Verify claims using a separate LLM call.
5) Output labels, corrections, and evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s+")
URL_RE = re.compile(r"https?://[^\s)\]】>\"']+")


@dataclass
class Evidence:
    title: str
    url: str
    snippet: str
    source: str


@dataclass
class ClaimResult:
    claim: str
    label: str
    correction: str
    rationale: str
    evidence: List[Evidence]


@dataclass
class SentenceResult:
    sentence: str
    extraction_method: str
    references: List[str]
    claims: List[ClaimResult]


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture = False
        self._current_href = ""
        self._current_text: List[str] = []
        self.results: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        attrs_dict = {k: v for k, v in attrs if v is not None}
        class_attr = attrs_dict.get("class", "")
        href = attrs_dict.get("href")
        if href and "result__a" in class_attr:
            self._capture = True
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture:
            title = "".join(self._current_text).strip()
            self.results.append({"title": title, "href": self._current_href})
            self._capture = False
            self._current_href = ""
            self._current_text = []


class TextExtractionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k: v for k, v in attrs if v is not None}
        class_attr = attrs_dict.get("class", "")
        id_attr = attrs_dict.get("id", "")
        if tag in {"script", "style", "nav", "footer", "header", "aside"}:
            self._skip_depth += 1
            return
        if "ad" in class_attr.lower() or "ads" in class_attr.lower():
            self._skip_depth += 1
            return
        if "ad" in id_attr.lower() or "ads" in id_attr.lower():
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth > 0 and tag in {
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "div",
            "section",
            "article",
        }:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    parts = SENTENCE_SPLIT_RE.split(normalized)
    sentences: List[str] = []
    buffer = ""
    for part in parts:
        if buffer:
            buffer = f"{buffer} {part}".strip()
        else:
            buffer = part.strip()
        if buffer.endswith(("。", "！", "?", "!", ".")):
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)
    return [s for s in sentences if s]


def extract_references(sentence: str) -> List[str]:
    return URL_RE.findall(sentence)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _openai_available() -> bool:
    return importlib.util.find_spec("openai") is not None


def _selenium_available() -> bool:
    return importlib.util.find_spec("selenium") is not None


def extract_claims(sentence: str, method: str, api_key: Optional[str]) -> List[str]:
    if method == "rule":
        candidates = [c.strip() for c in re.split(r"[、,;；]", sentence) if c.strip()]
        return candidates if candidates else [sentence]

    if not api_key or not _openai_available():
        return [sentence]

    import openai

    openai.api_key = api_key
    prompt = (
        "次の文から検証可能な主張を箇条書きで抽出してください。"
        "推測や感想は除外し、検証可能な事実のみを短く書いてください。"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは主張抽出器です。"},
            {"role": "user", "content": f"{prompt}\n\n文: {sentence}"},
        ],
        temperature=0,
        max_tokens=200,
    )
    content = response.choices[0].message["content"].strip()
    lines = [line.strip("-・ ") for line in content.splitlines() if line.strip()]
    return lines if lines else [sentence]


def _fetch_url(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FactChecker/1.0)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(200000).decode(charset, errors="ignore")


def _extract_text(html: str) -> str:
    parser = TextExtractionParser()
    parser.feed(html)
    return re.sub(r"\s+", " ", parser.text()).strip()


def _is_ad_like(url: str, title: str) -> bool:
    lowered = f"{title} {url}".lower()
    ad_keywords = ("adservice", "doubleclick", "adsystem", "/ads", "adclick")
    return any(keyword in lowered for keyword in ad_keywords)


def _enrich_evidence(evidence: List[Evidence]) -> List[Evidence]:
    enriched: List[Evidence] = []
    for item in evidence:
        if not item.url or _is_ad_like(item.url, item.title):
            continue
        try:
            html = _fetch_url(item.url)
            snippet = _extract_text(html)[:400]
        except Exception as exc:
            snippet = f"取得失敗: {exc}"
        enriched.append(
            Evidence(
                title=item.title,
                url=item.url,
                snippet=snippet,
                source=item.source,
            )
        )
    return enriched


def _extract_ddg_results(html: str) -> List[Dict[str, str]]:
    parser = DuckDuckGoHTMLParser()
    parser.feed(html)
    results = []
    for item in parser.results:
        href = item["href"]
        if "duckduckgo.com/l/?" in href:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            href = qs.get("uddg", [href])[0]
        results.append({"title": item["title"], "url": href})
    return results


def search_web(query: str, max_results: int = 5) -> List[Evidence]:
    search_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    html = _fetch_url(search_url)
    results = _extract_ddg_results(html)[:max_results]
    evidence: List[Evidence] = []
    for item in results:
        evidence.append(
            Evidence(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet="",
                source="duckduckgo",
            )
        )
    return _enrich_evidence(evidence)


def _search_engine_url(engine: str, query: str) -> str:
    encoded = urllib.parse.urlencode({"q": query})
    if engine == "google":
        return f"https://www.google.com/search?{encoded}"
    if engine == "bing":
        return f"https://www.bing.com/search?{encoded}"
    return f"https://duckduckgo.com/?{encoded}"


def search_web_selenium(query: str, engine: str, max_results: int = 5) -> List[Evidence]:
    if not _selenium_available():
        return [
            Evidence(
                title="",
                url="",
                snippet="selenium がインストールされていません。",
                source="selenium",
            )
        ]

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By

    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    results: List[Evidence] = []
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(_search_engine_url(engine, query))
        if engine == "google":
            anchors = driver.find_elements(By.CSS_SELECTOR, "div#search a h3")
            for anchor in anchors[:max_results]:
                link = anchor.find_element(By.XPATH, "./ancestor::a")
                results.append(
                    Evidence(
                        title=anchor.text,
                        url=link.get_attribute("href"),
                        snippet="",
                        source=f"selenium:{engine}",
                    )
                )
        elif engine == "bing":
            anchors = driver.find_elements(By.CSS_SELECTOR, "li.b_algo h2 a")
            for anchor in anchors[:max_results]:
                results.append(
                    Evidence(
                        title=anchor.text,
                        url=anchor.get_attribute("href"),
                        snippet="",
                        source=f"selenium:{engine}",
                    )
                )
        else:
            anchors = driver.find_elements(By.CSS_SELECTOR, "a.result__a")
            for anchor in anchors[:max_results]:
                results.append(
                    Evidence(
                        title=anchor.text,
                        url=anchor.get_attribute("href"),
                        snippet="",
                        source=f"selenium:{engine}",
                    )
                )
    except Exception as exc:
        results = [
            Evidence(
                title="",
                url="",
                snippet=f"selenium 検索に失敗: {exc}",
                source=f"selenium:{engine}",
            )
        ]
    finally:
        driver.quit()

    return _enrich_evidence(results)


def build_reference_evidence(urls: Iterable[str]) -> List[Evidence]:
    evidence: List[Evidence] = []
    for url in urls:
        if _is_ad_like(url, ""):
            continue
        try:
            content = _fetch_url(url)
            snippet = _extract_text(content)[:400]
        except Exception as exc:
            snippet = f"取得失敗: {exc}"
        evidence.append(
            Evidence(
                title="",
                url=url,
                snippet=snippet,
                source="reference",
            )
        )
    return evidence


def verify_claim(
    claim: str,
    evidence: List[Evidence],
    api_key: Optional[str],
) -> ClaimResult:
    if not api_key or not _openai_available():
        return ClaimResult(
            claim=claim,
            label="not_verified",
            correction="",
            rationale="OpenAI API キーが未設定、または openai パッケージがありません。",
            evidence=evidence,
        )

    import openai

    openai.api_key = api_key
    evidence_text = "\n".join(
        f"- {item.title} ({item.url}) {item.snippet}" for item in evidence
    )
    prompt = (
        "以下の主張が証拠と整合するか判定してください。"
        "ラベルは supported / not_true / not_verified のいずれか。"
        "誤りの場合は訂正案を短く提示してください。"
        "JSON形式で返してください。"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたはファクトチェッカーです。"},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n主張: {claim}\n\n証拠:\n{evidence_text}"
                ),
            },
        ],
        temperature=0,
        max_tokens=300,
    )
    content = response.choices[0].message["content"].strip()
    data = {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "label": "not_verified",
            "correction": "",
            "rationale": "LLM 応答の JSON 解析に失敗しました。",
        }
    return ClaimResult(
        claim=claim,
        label=data.get("label", "not_verified"),
        correction=data.get("correction", ""),
        rationale=data.get("rationale", ""),
        evidence=evidence,
    )


def fact_check_text(
    text: str,
    claim_method: str,
    api_key: Optional[str],
    search_backend: str,
    search_engine: str,
    max_sentences: Optional[int],
) -> List[SentenceResult]:
    sentences = split_sentences(text)
    if max_sentences:
        sentences = sentences[:max_sentences]
    results: List[SentenceResult] = []
    for sentence in sentences:
        references = extract_references(sentence)
        evidence = build_reference_evidence(references)
        claims = extract_claims(sentence, claim_method, api_key)
        claim_results: List[ClaimResult] = []
        for claim in claims:
            if search_backend == "selenium":
                search_evidence = search_web_selenium(
                    claim, engine=search_engine, max_results=5
                )
            else:
                search_evidence = search_web(claim)
            combined = evidence + search_evidence
            claim_results.append(verify_claim(claim, combined, api_key))
        results.append(
            SentenceResult(
                sentence=sentence,
                extraction_method=claim_method,
                references=references,
                claims=claim_results,
            )
        )
    return results


def render_markdown(results: List[SentenceResult]) -> str:
    lines = []
    for idx, sentence_result in enumerate(results, 1):
        lines.append(f"## Sentence {idx}")
        lines.append(sentence_result.sentence)
        lines.append("")
        for claim_idx, claim in enumerate(sentence_result.claims, 1):
            lines.append(
                f"### Claim {claim_idx}: {claim.claim}\n"
                f"- Label: {claim.label}\n"
                f"- Rationale: {claim.rationale}\n"
                f"- Correction: {claim.correction}"
            )
            if claim.evidence:
                lines.append("- Evidence:")
                for item in claim.evidence:
                    lines.append(
                        f"  - {item.title} ({item.url}) {item.snippet}"
                    )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentence-level fact checker")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("--output", default="fact_check_results.json")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--claim-method", choices=["llm", "rule"], default="llm")
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (overrides OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--search-backend",
        choices=["http", "selenium"],
        default="http",
        help="Web search backend (http uses DuckDuckGo HTML)",
    )
    parser.add_argument(
        "--search-engine",
        choices=["duckduckgo", "google", "bing"],
        default="duckduckgo",
        help="Search engine for selenium backend",
    )
    parser.add_argument("--max-sentences", type=int)
    args = parser.parse_args()

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    text = load_text(args.input)
    results = fact_check_text(
        text=text,
        claim_method=args.claim_method,
        api_key=api_key,
        search_backend=args.search_backend,
        search_engine=args.search_engine,
        max_sentences=args.max_sentences,
    )

    if args.format == "markdown":
        output = render_markdown(results)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(output)
    else:
        payload = [asdict(result) for result in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
