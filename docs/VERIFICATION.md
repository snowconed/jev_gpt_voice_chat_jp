# 公開用ソースの検証

2026-10-15、Windows／Python 3.12で検証しました。

- Jev＋GPT音声専用に整理したPython回帰試験: **509件成功**（507件の全体試験と、フィラー生成・失敗時の後始末に関する追加2件）。
- 上記には公開対象検査の試験**10件**を含みます。無視ファイルを強制追加した場合や、ステージ済みの秘密を作業コピーからだけ削除した場合の検出も確認しています。
- ブラウザ・AudioWorklet・ヘッドレスChrome: **59件成功**。
- サーバーはJev＋GPT音声のみを受け付けます。`--turn-mode live`、`--tts voicevox`、`--tts piper`、削除した音声エンジン用オプションの拒否を確認しました。フィラー生成のバックエンドもGPTのみです。
- 公開用ファイルを検査し、既知の秘密情報形式・実行環境のAPIキーとの一致・個人のWindows絶対パスは検出しませんでした。

テストで使用する認証文字列は明示的なダミー値です。会話記録、実行ログ、APIキー、生成音声、モデル、仮想環境、元のGit履歴は同梱していません。フィラーは初回に生成してください。

この整理の検証では実APIへの会話リクエストを送信していません。実マイク・スピーカーによる品質評価や、すべてのAPIアカウントでのモデル利用可否はこの試験には含みません。ブラウザテストはPlaywrightを使い、今回の検証では`JEV_TEST_BROWSER_CHANNEL=chrome`でインストール済みChromeを指定しました。通常はREADMEの手順でPlaywright Chromiumを導入して実行します。

再検証コマンド:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
npm run test:browser
python tools/check_publication.py
git add -n .
```

公開対象の検査は現在のファイルとGitのステージを対象とする補助機能です。後から書き込んだ任意の個人情報や過去のGitコミットを網羅するものではありません。
