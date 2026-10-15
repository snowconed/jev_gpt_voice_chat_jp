# 開発・他の対話システムとの接続

## 処理の流れ

```text
ブラウザのマイク
  → 音声認識（OpenAI）
  → ターン・相槌・追加訂正の判定（Jev）
  → ChatCore → responder（標準: OpenAI Responses API）
  → 返答本文を変更せず音声合成（GPT音声）
  → ブラウザで再生 → 再生完了通知
```

発話が確定すると返答生成を開始します。フィラーを有効にし、その声のキャッシュが読み込まれている場合は、この段階で短いフィラーを再生します。本文の生成・合成は並行して進め、本文の再生だけをフィラーの完了まで待ちます。フィラー未送信のうちに本文音声が準備できた場合は、フィラーを省略します。

音声認識の区間終了は、ユーザーの発話確定とは別に扱います。Jevが文字起こしと会話の文脈から判断し、ランタイムが未確定の認識や入力配送を確認してから次の段階へ進めます。

## ファイルの役割

| ファイル・フォルダ | 役割 |
| --- | --- |
| `server.py` | localhostのHTTP・WebSocket、設定とセッションの管理 |
| `web/` | マイク、字幕、音声再生、開始前のフィラー選択 |
| `transcription_client.py` | 音声認識接続 |
| `jev_client.py` | Jevの分類リクエストと応答検証 |
| `jev_runtime.py` | 発話確定、準備中の追加、相槌・割込みの状態管理 |
| `voice_runtime.py`、`speech_runtime.py` | 入力・返答・TTS・再生確認の共通処理 |
| `filler_runtime.py` | 本文とは独立したフィラー再生状態 |
| `filler_cache.py`、`prepare_fillers.py` | フィラー設定の照合・読み込み・事前生成 |
| `chat_core.py` | 会話状態、配信結果、再開時の復旧 |
| `chat_model.py` | 差し替え可能な標準返答生成アダプタ |
| `chat_config.py`、`config.json`、`prompts/` | 会話モデルとプロンプト |
| `speech_client.py` | 本文をGPT音声にするアダプタ |
| `speech_segments.py` | 本文を保ったまま最大2区間に分割 |
| `tests/` | Pythonとブラウザの回帰テスト |
| `tools/check_publication.py` | 公開予定ファイルの検査 |

`live_client.py`と`live_context.py`は共通基底クラスの互換処理から参照されるため残しています。公開版のサーバー入口はJevのみを受け付け、Jev経路ではGPT-Liveとの対話接続を開始しません。音声認識用のOpenAI Realtime接続は引き続き使用します。Piperの起動入口・依存関係・モデルは含めていません。

## 返答生成システムを差し替える

まずは`ChatCore`へ`responder`を注入する方法を使えます。同期メソッド`respond(history, settings) -> str`が最小の境界です。

```python
from pathlib import Path

from aiohttp import web
from chat_config import load_settings
from chat_core import ChatCore
from server import create_app


class MyResponder:
    def respond(self, history, settings):
        # history: {"role": "user" | "assistant", "content": str} のリスト
        # ここを外部の対話システムへの問い合わせに置き換える。
        return "返答生成部分を接続できました。"

    def close(self):
        # 持続接続などがあれば終了する。
        pass


data_dir = Path("data_custom")
core = ChatCore(data_dir, settings=load_settings(), responder=MyResponder())
app = create_app(data_dir=data_dir, core=core, port=8793, tts="gpt")
web.run_app(app, host="127.0.0.1", port=8793)
```

`history`にはユーザー入力と、再生確認できたアシスタント本文だけを含めます。外部システムからは空でない2000文字以内の文字列を返します。本文に対してアプリ側の要約・言い換えは行いません。APIの例外本文や認証情報をユーザー向け返答に入れないようにしてください。

割込み時に外部問い合わせ自体も中断する場合は、任意の`respond_cancellable(history, settings, cancel_event) -> str`を実装します。`cancel_event`は`threading.Event`です。定期的に確認し、必要なら`chat_model.ChatGenerationCancelled`を送出して中断を表現します。外部サービスへのリクエスト取消はアダプタ側の責任です。

最小の`respond`だけでも接続できますが、処理中の問い合わせはそのまま走り続ける場合があります。アプリは古い返答を採用しないようにします。標準アダプタには、入力変更・割込み・終了時のキャンセルと、総待ち時間内の一度の再試行があります。

HTTP層を独自化したい場合は、`ChatCore`の`create`、`view`、`pending`、`submit`、`delivery`を直接呼べます。会話と診断ログの保存場所は同じ`data_dir`を渡してください。1つの保存フォルダを複数サーバープロセスで共有しないでください。

## 保持するべき状態と契約

- **返答本文:** TTSへ渡す文字列は元の返答を連結して再現できる区間に分けます。最初の区間を再生しながら残りを準備します。
- **聞こえた本文:** ブラウザから実再生完了通知を受けた区間だけを履歴へ反映します。途中で止めた区間の聞こえた文字数を経過時間で推定しません。
- **フィラー:** 本文と別のID・完了通知・タイムアウトを持ち、本文の字幕や履歴へ追加しません。フィラーの通知が失われた場合も、本文や会話全体を永久に待たせません。
- **明示的な停止:** 「待って」や短い否定だけの割込みでは読み上げと準備中の返答を止め、続きを待ちます。訂正内容まで続けば、その入力を会話へ渡します。
- **相槌:** 準備中の相槌や同じ内容の繰り返しでは元の返答を保持し、内容が変わる追加・訂正では更新します。音量だけで割込みを確定しません。
- **世代管理:** 入力・再生のIDと世代を照合し、遅れて届いた旧返答・旧フィラー・旧完了通知を採用しません。
- **失敗時:** 返答生成の失敗で音声接続全体を終了せず、同じ対話で次の入力を受けられるようにします。

この契約を変更すると、相槌による誤中断、訂正後の古い返答再生、フィラーと本文の二重再生が起きやすくなります。関連テストも併せて更新してください。

## ブラウザと音声

通信する音声は24 kHz・モノラル・符号付き16 bit little-endian PCMです。ブラウザ側はAudioWorkletで出力デバイスのレートに合わせます。音声の受信完了だけで再生済みにせず、出力タイミングを確認して通知します。

フィラーPCMは対話前に読み込みます。対話開始時の`filler_enabled`はセッションに固定します。候補の不足や読み込み失敗時は無効にし、対話開始後にAPIを呼んでフィラーを作りません。声や合成設定が異なるキャッシュを流用しないように、manifestの設定・ハッシュ・形式を照合します。

## 実行時ファイルと公開範囲

| パス | 内容 | Gitでの扱い |
| --- | --- | --- |
| `data_jev/` | 会話、配信状態、診断ログ | 除外 |
| `.runtime/` | テストなどの一時ファイル・出力 | 除外 |
| `.venv/`、`node_modules/` | インストールした依存関係 | 除外 |
| `assets/fillers/`内の生成ディレクトリ | PCMとmanifest | 除外 |
| `assets/fillers/README.txt` | 再生成の手順 | 含める |
| `.env`などの秘密設定 | このアプリでは自動読込しない | 除外 |

独自の`--data-dir`やレポート出力先を追加した場合、その場所も除外してください。フィラー候補に個人的な文や人名を追加した場合は、生成前に外部TTSへの送信内容を確認してください。

## テスト

リポジトリルートで実行します。JavaScriptのテストにはNode.js 20以降を使います。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
node --test tests/test_browser_audio.cjs tests/test_restart_ui.cjs
```

画面テストも実行する場合:

```powershell
npm install
npx playwright install chromium
npm run test:browser
```

PythonテストはAPIクライアントをモックし、状態遷移、通信異常、古い通知、本文の保持、キャンセル、フィラー中の停止・否定・相槌などを検証します。ブラウザテストはWorkletや再生通知、再起動時の設定取得、フィラーの選択・プリロード・表示を検証します。Node.jsだけで実行する2本に加え、全体実行ではPlaywrightのヘッドレスブラウザを使います。

モデルや声を変更した場合は、別途、少数の合成・実対話で品質と待ち時間を確認してください。API料金が発生する実験は通常の回帰テストに混ぜず、テスト入力と保存先を明示して行います。会話ログや音声を公開テストの固定入力へコピーしないでください。

公開前には`python tools/check_publication.py`を実行し、Gitの追加候補を`git add -n .`で確認します。
