// Real DOM/layout checks in headless Chrome. No microphone, API, screenshots,
// production server, or visible preview is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const { chromium } = require("playwright");

const web = path.resolve(__dirname, "../web");
const html = fs.readFileSync(path.join(web, "index.html"), "utf8")
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
  .replace(/<link\b[^>]*>/g, "");
const css = fs.readFileSync(path.join(web, "style.css"), "utf8");
const app = fs.readFileSync(path.join(web, "app.js"), "utf8")
  .replace(/initialize\(\);\s*$/, "");

function progress(mode = "jev") {
  return { type: "processing_state", mode,
    turn: { code: "jev_judging", label: "話し終わりを確認中", detail: "入力の意味を確認しています。", elapsed_ms: 2100, remaining_ms: 650, wait_label: "次の確認まで" },
    reply: { code: "tts_preparing", label: "声を準備中", detail: "返答を音声にしています。", elapsed_ms: 1250 },
    recent: [{ label: "入力を受け取りました", detail: "返答を準備しています。" }] };
}

test("standalone voice chat has two accessible progress cards and no experience UI", async (t) => {
  const browser = await chromium.launch({ channel: process.env.JEV_TEST_BROWSER_CHANNEL || undefined, headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
  const requests = [];
  await page.route("**/*", (route) => { requests.push(route.request().url()); return route.abort(); });
  await page.setContent(html);
  await page.addStyleTag({ content: css });
  await page.addScriptTag({ content: app + `
    globalThis.__chatUi = {
      attempts: [],
      activate(mode = "jev") {
        if (active) stopAttempt(active);
        applyConfiguration({turn_mode:mode, roles:[{id:"chat",label:"雑談"}], voice:"marin", version:"1.0.0", csrf_token:"offline-token"});
        const attempt = createAttempt();
        attempt.socket = {readyState:WebSocket.OPEN, send() {}, close() {this.readyState=WebSocket.CLOSED;}};
        active = attempt;
        this.attempts.push(attempt);
        resetProcessing("connected");
        setControls();
      },
      deliver(message, index = this.attempts.length - 1) {
        handleServerMessage(this.attempts[index], {data:JSON.stringify(message)});
      },
      stop() {stopAttempt(active);}
    };` });

  await t.test("both modes preserve captions, chat role and progress without a third panel", async () => {
    for (const mode of ["live", "jev"]) {
      await page.evaluate((mode) => window.__chatUi.activate(mode), mode);
      await page.evaluate((state) => {
        window.__chatUi.deliver({type:"input_transcript", text:"こんにちは", replace:true});
        window.__chatUi.deliver({type:"output_transcript", text:"こんにちは。", replace:true});
        window.__chatUi.deliver(state);
      }, progress(mode));
      assert.equal(await page.locator("#processing-mode").textContent(), mode === "jev" ? "Jev方式" : "Live方式");
      assert.equal(await page.locator(".processing-card").count(), 2);
      assert.equal(await page.locator("#processing-experience, #processing-steps").count(), 0);
      assert.equal(await page.locator("#role").inputValue(), "chat");
      assert.equal(await page.locator("#role").isVisible(), false);
      assert.equal(await page.locator("#input-caption").textContent(), "こんにちは");
      assert.equal(await page.locator("#output-caption").textContent(), "こんにちは。");
      assert.match(await page.locator("#server-version").textContent(), /1\.0\.0/);
      assert.match(await page.locator("#processing-turn-timing").textContent(), /次の確認まで 約0\.7秒/);
    }
  });

  await t.test("timer changes do not rebuild history or repeat live announcements", async () => {
    await page.evaluate((state) => {
      window.__chatUi.activate(); window.__chatUi.deliver(state);
      window.__firstRecent = document.querySelector("#processing-recent-list li");
      window.__changes = 0;
      window.__observer = new MutationObserver((rows) => {window.__changes += rows.length;});
      window.__observer.observe(document.querySelector("#processing-announcement"), {subtree:true, childList:true, characterData:true});
    }, progress());
    const next = progress(); next.turn.remaining_ms = 100; next.reply.elapsed_ms = 2000;
    await page.evaluate((state) => window.__chatUi.deliver(state), next);
    assert.deepEqual(await page.evaluate(() => ({changes:window.__changes,
      stable:window.__firstRecent === document.querySelector("#processing-recent-list li")})), {changes:0, stable:true});
    await page.evaluate(() => window.__observer.disconnect());
  });

  await t.test("server strings remain text and history stays bounded", async () => {
    const state = progress();
    const text = '<img src="forbidden" onerror="window.__injected=true">';
    state.turn.label = text; state.reply.detail = text;
    state.recent = Array.from({length:10}, () => ({label:text, detail:text}));
    await page.evaluate((state) => window.__chatUi.deliver(state), state);
    assert.equal(await page.locator("#processing-turn-label").textContent(), text);
    assert.equal(await page.locator(".processing-panel img").count(), 0);
    assert.equal(await page.locator("#processing-recent-list li").count(), 6);
    assert.equal(await page.evaluate(() => window.__injected), undefined);
  });

  await t.test("stop and terminal server snapshots reject late progress", async () => {
    for (const action of ["stop", "closed", "error"]) {
      await page.evaluate(({state, action}) => {
        window.__chatUi.activate(); window.__chatUi.deliver(state);
        if (action === "stop") window.__chatUi.stop();
        else window.__chatUi.deliver({type:"processing_state", mode:"jev", turn:{code:action,label:"終了"}, reply:{code:action,label:"終了"}, recent:[]});
        window.__chatUi.deliver(state);
      }, {state:progress(), action});
      assert.equal(await page.locator(".processing-card.stopped").count(), 2);
      assert.equal(await page.locator(".processing-timing:not([hidden])").count(), 0);
      assert.equal(await page.locator("#processing-recent-list li").count(), 0);
      assert.notEqual(await page.locator("#processing-turn-label").textContent(), "話し終わりを確認中");
    }
  });

  await t.test("missing timing clears countdowns without NaN", async () => {
    await page.evaluate((state) => {window.__chatUi.activate(); window.__chatUi.deliver(state);}, progress());
    await page.evaluate(() => window.__chatUi.deliver({type:"processing_state", mode:"jev",
      turn:{label:"待機", elapsed_ms:-1, remaining_ms:"2000"}, reply:{label:"待機"}, recent:[]}));
    assert.equal(await page.locator(".processing-timing:not([hidden])").count(), 0);
    assert.equal((await page.locator(".processing-panel").textContent()).includes("NaN"), false);
  });

  await t.test("two desktop columns and a single narrow column have no overflow", async () => {
    const state = progress(); state.reply.detail = "長い説明".repeat(80);
    await page.evaluate((state) => window.__chatUi.deliver(state), state);
    for (const width of [1100, 375, 320]) {
      await page.setViewportSize({width, height:900});
      const layout = await page.evaluate(() => ({width:document.documentElement.scrollWidth, viewport:innerWidth,
        cards:[...document.querySelectorAll(".processing-card")].map((node) => {
          const box = node.getBoundingClientRect(); return {top:box.top,left:box.left};
        })}));
      assert(layout.width <= layout.viewport);
      if (width > 760) assert.equal(layout.cards[0].top, layout.cards[1].top);
      else {
        assert.equal(layout.cards[0].left, layout.cards[1].left);
        assert(layout.cards[1].top > layout.cards[0].top);
      }
    }
  });
  assert.deepEqual(requests, []);
});
