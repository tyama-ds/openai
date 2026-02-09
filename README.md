# openai
for_codex

## pdf_fact_checker.py
PDFファイルを読み込み、要約・分割後にファクトチェックを行うサンプルスクリプトです。
OpenAI API キーが設定されている場合は ChatGPT を利用して要約とファクトチェックを
実行し、設定されていない場合は HuggingFace Transformers による要約のみを行います。

### インストール
```
pip install openai pdfminer.six transformers
```

### 使い方
```
python pdf_fact_checker.py <PDFファイルパス>
```
結果は標準出力にMarkdown形式で表示され、`result.csv` にも保存されます。

OpenAI API を使用する場合は `OPENAI_API_KEY` 環境変数にキーを設定してください。

サンプル文書 `sample_document.txt` を同梱しているので、任意の方法で PDF に変換して試すことができます。

## Deep Research Tool
CLI 指向のリサーチオーケストレーターです。OpenAI/Anthropic の LLM、DuckDuckGo 検索、Evidence Locker（証跡出力）、Markdown レポート生成をサポートしています。

### インストールとセットアップ
```bash
pip install -r requirements.txt
# 生成系 LLM を使う場合は、必要に応じて以下もインストール
pip install openai anthropic
```

> DuckDuckGo 検索クライアント (`duckduckgo_search`) がインストールできないオフライン環境では、検索結果を模倣するモッククライアントに自動フォールバックします（タイトル・概要はテスト用のダミーになります）。

環境変数で API キーを渡せます。

- OpenAI: `OPENAI_API_KEY=<your_key>`
- Anthropic: `ANTHROPIC_API_KEY=<your_key>`

### 基本的な実行方法
```bash
python -m deep_research_tool.cli "調査したいテーマ" \
  --llm-provider openai \
  --search-provider duckduckgo \
  --iterations 3 \
  --max-results 5
```

主なオプション（`--help` でも確認できます）:

- `brief` (必須): 調査テーマや問い。
- `--requirements`: 追加要件やゴールを文字列で指定。
- `--iterations`: リサーチループ回数。各ループで検索・要約・不足補完を行います。
- `--toc-items`: 目次項目数。初回の LLM プランニングで生成される調査目次の長さを指定。
- `--max-results`: 1 クエリあたりの検索件数。
- `--llm-provider`: `openai` / `anthropic` / `mock` から選択。
- `--search-provider`: 現在は `duckduckgo` をサポート（将来的に Selenium 等に拡張可能）。
- `--additional-pdf`: 参考資料となる PDF ファイルパス（複数指定可）。`additional_pdf=True` が有効になるイメージで、既存レポート等を下敷きにします。
- `--output`: 出力レポートの保存先（デフォルト `reports/report.md`）。

### 実行結果
- 調査レポート: デフォルトでは Markdown (`reports/report.md`) に保存。必要なら `--output` で別パスに変更。
- Evidence Locker: アクセスした URL や引用を JSON/CSV (`reports/report.evidence.json`, `reports/report.evidence.csv`) に保存し、他レポートとの統合に利用できます。
- 検証ダッシュボード: レポート内に不確実性の高いポイントを可視化するセクションを自動追加します。

### 例
最低限の動作確認（モック LLM／短縮ループ）:
```bash
python -m deep_research_tool.cli "量子コンピュータの動向" --llm-provider mock --iterations 1 --max-results 2
```

OpenAI を使用し、既存 PDF レポートも参照に入れる例:
```bash
export OPENAI_API_KEY=sk-...
python -m deep_research_tool.cli "2024 年の生成 AI セキュリティ動向" \
  --llm-provider openai \
  --iterations 3 \
  --additional-pdf path/to/whitepaper.pdf path/to/analysis.pdf
```
