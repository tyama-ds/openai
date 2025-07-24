# coding: utf-8
"""PDF文書を読み込み、要約と分割を行った後、
各セクションに対してファクトチェックを実行するサンプルスクリプト。
ファクトチェックや要約の部分は外部APIの利用を想定しているため、
ここではダミー実装を行っている。
"""

import csv
import os
import sys
from typing import List

# pdfminer.six の高レベルAPIを利用する想定
try:
    from pdfminer.high_level import extract_text
except ImportError:
    extract_text = None  # 依存ライブラリがない場合は None


def read_pdf(path: str) -> str:
    """PDFファイルまたはテキストファイルから文章を取得する

    Parameters
    ----------
    path: str
        ファイルパス (PDF or TXT)

    Returns
    -------
    str
        抽出したテキスト。取得できなければ空文字列。
    """
    if path.lower().endswith(".pdf") and extract_text is not None:
        try:
            return extract_text(path)
        except Exception as e:
            print(f"PDF読み込みでエラーが発生しました: {e}")

    # 上記で取得できなかった場合はテキストファイルとして読み込みを試みる
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"テキスト読み込みに失敗しました: {e}")
        return ""


def summarize_text(text: str) -> str:
    """文章の要約を生成する関数

    OPENAI_API_KEY が設定されている場合は OpenAI API を利用し、
    そうでなければ HuggingFace の pipeline による要約を試みる。
    いずれも利用できない場合は先頭100文字を返す。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            openai.api_key = api_key
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "以下の文章を短く要約してください。"},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            return res.choices[0].message["content"].strip()
        except Exception as e:
            print(f"OpenAI API でエラーが発生しました: {e}")

    try:
        from transformers import pipeline
        summarizer = pipeline("summarization")
        result = summarizer(text, max_length=120, min_length=30, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        print(f"HuggingFace による要約に失敗しました: {e}")
        return text[:100] + "..." if len(text) > 100 else text


def split_text(text: str, max_chars: int = 200) -> List[str]:
    """テキストをおよそ 100~200 文字単位で分割する

    文の途中で分断しないように句点や改行位置を優先して切り出し、
    前後の文脈が極力失われないように調整する。
    デフォルトの ``max_chars`` は 200 文字だが、実際の長さは
    区切り位置により前後する。
    """
    segments = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            # 句点または改行位置を探す
            break_pos = text.rfind('。', start, end)
            if break_pos == -1:
                break_pos = text.rfind('\n', start, end)
            if break_pos == -1:
                break_pos = end
            else:
                break_pos += 1
        else:
            break_pos = end
        segment = text[start:break_pos].strip()
        if segment:
            segments.append(segment)
        start = break_pos
    return segments


def fact_check_segment(summary: str, segment: str) -> List[List[str]]:
    """セグメントに対してファクトチェックを実行する関数"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # API キーがない場合は解析不能として返す
        return [["1", segment[:50] + "...", "高", "判定不能", "",
                 "OpenAI API キーが設定されていません"]]

    try:
        import openai
        openai.api_key = api_key
        prompt = (
            "### セクション要約\n" + summary + "\n\n" +
            "### 対象テキスト\n" + segment + "\n\n" +
            "次の方針でファクトチェックしてください。文章から主要な主張と、" +
            "以下のハルシネーションしやすい項目に該当する記述を抽出して評価します:\n" +
            "年号・日付・時系列、統計・数値・割合、医学・健康・法律、地理・固有名詞、" +
            "出典のない引用、因果関係の断定、技術仕様、歴史的出来事、金融・経済指標。\n" +
            "リスクは『高・中・低』、判定は『正確・一部不正確・誤り・判定不能』から選び、" +
            "情報ソースのURLも必ず記載してください。\n" +
            "以下のMarkdown表で回答してください:\n" +
            "| # | ファクトチェック対象 | リスク | 判定 | 根拠 (簡潔) | 情報ソース / リンク | コメント / 修正案 |"
        )
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは厳格かつ中立なファクトチェッカーです。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = res.choices[0].message["content"]

        rows = []
        for line in content.splitlines():
            if line.startswith("|") and line.count("|") >= 7:
                parts = [p.strip() for p in line.strip().strip("|").split("|")]
                if len(parts) == 7 and parts[0].isdigit():
                    rows.append(parts)
        if rows:
            return rows
        return [["1", segment[:50] + "...", "高", "判定不能", "", "表解析失敗"]]
    except Exception as e:
        return [["1", segment[:50] + "...", "高", "判定不能", "", f"APIエラー: {e}"]]


def format_table(section_no: int, rows: List[List[str]], counts: dict, remain: int) -> str:
    header = f"### セクション {section_no} のファクトチェック"
    table = [
        "| # | ファクトチェック対象 | リスク | 判定 | 根拠 (簡潔) | 情報ソース / リンク | コメント / 修正案 |",
        "|---|------------------|------|------|-------------|---------------------|-------------------|",
    ]
    table.extend(["| " + " | ".join(row) + " |" for row in rows])
    summary_line = (
        f"**要約**  \n- 正確：{counts['correct']}\t"
        f"一部不正確：{counts['partial']}\t"
        f"誤り：{counts['wrong']}\t判定不能：{counts['unknown']}  "
    )
    footer = f"***\n(残り {remain} セクション / 処理済み {section_no})"
    return "\n".join([header] + table + [summary_line, footer])


def save_csv(all_rows: List[List[str]], path: str) -> None:
    """結果をCSVに保存する"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "#",
            "ファクトチェック対象",
            "リスク",
            "判定",
            "根拠",
            "情報ソース / リンク",
            "コメント",
        ])
        for i, row in enumerate(all_rows, 1):
            writer.writerow([i] + row[1:])


def main(pdf_path: str) -> None:
    text = read_pdf(pdf_path)
    if not text:
        print("PDFからテキストを取得できませんでした")
        return

    summary = summarize_text(text)
    segments = split_text(text)
    all_rows = []
    total = len(segments)
    auto = False
    for idx, seg in enumerate(segments, 1):
        rows = fact_check_segment(summary, seg)
        all_rows.extend(rows)
        counts = dict(correct=0, partial=0, wrong=0, unknown=0)
        for r in rows:
            verdict = r[3]
            if verdict == "正確":
                counts["correct"] += 1
            elif "一部不正確" in verdict:
                counts["partial"] += 1
            elif verdict == "誤り":
                counts["wrong"] += 1
            else:
                counts["unknown"] += 1
        remain = total - idx
        print(format_table(idx, rows, counts, remain))
        if remain and not auto:
            command = input(f"ーセクション{idx}についてチェック完了ー 次に進む場合は『続行』と入力してください。")
            if command.strip() in ("自動で続行", "一括で続行"):
                auto = True
            else:
                while command.strip() != "続行":
                    command = input()

    save_csv(all_rows, 'result.csv')
    print('― 全セクションの検証が完了しました ―')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pdf_fact_checker.py <pdf file>")
    else:
        main(sys.argv[1])
