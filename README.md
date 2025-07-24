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
