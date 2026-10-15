# 第三者コンポーネント・外部サービス・音声

このリポジトリの[MIT License](LICENSE)は、本プロジェクトのコードと文書に適用します。依存ライブラリ、外部API、音声モデル、生成音声、キャラクター等の権利や利用条件を変更するものではありません。それぞれの配布元・提供元の条件を確認してください。

## Jev / Typesafe

Jevの判定には外部APIを使用します。APIの利用契約・料金・データの扱いは提供元の条件に従います。本プロジェクトのMITライセンスは、サービスやモデルの利用権を付与しません。

- [公式APIドキュメント](https://docs.typesafe.ai/api)
- [公式モデルドキュメント](https://docs.typesafe.ai/models)

## OpenAI

音声認識・標準の返答生成・GPT音声合成にはOpenAI APIを使用します。APIの利用条件・料金・データの扱いは提供元の条件に従います。音声はAIが生成するもので、本アプリの画面でもその旨を示します。

生成済み音声は同梱しません。`prepare_fillers.bat`で生成したフィラーにも、外部サービスの利用条件が適用されます。

- [OpenAI公式APIドキュメント](https://developers.openai.com/api/docs/)
- [音声合成ガイド](https://developers.openai.com/api/docs/guides/text-to-speech)
- [OpenAIの利用規約・ポリシー](https://openai.com/policies/)

## Python・ブラウザテストの依存関係

Pythonの直接依存は`requirements.txt`、固定した依存関係は`requirements.lock.txt`に記載しています。ブラウザテストの開発依存は`package.json`とロックファイルを参照してください。各パッケージはそれぞれのライセンスに従います。依存パッケージをMITへ変更するものではありません。

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Pydantic](https://github.com/pydantic/pydantic)
- [aiohttp](https://github.com/aio-libs/aiohttp)
- [Playwright](https://github.com/microsoft/playwright)

本リポジトリには仮想環境、`node_modules/`、ブラウザ実行ファイルを同梱しません。インストールや再配布の際は、取得したパッケージに付属する通知とライセンスを確認してください。
