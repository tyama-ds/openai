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

## sentence_fact_checker.py
文章を1文ごとに分割し、主張抽出→Web検索→別LLMで検証するファクトチェッカーです。

### インストール
```
pip install openai
pip install selenium
```

### 使い方
```
python sentence_fact_checker.py <入力テキスト> --format markdown
```

OpenAI API を使う場合は `OPENAI_API_KEY` を設定するか、
`--openai-api-key` で直接指定してください。

selenium でヘッドレス検索する場合は `--search-backend selenium` と
`--search-engine google` (または `bing`, `duckduckgo`) を指定してください。
ChromeDriver が必要なので、実行環境に合わせて用意してください。
検索結果のリンク先ページを取得して本文抽出を行いますが、広告と判断したページは除外します。
