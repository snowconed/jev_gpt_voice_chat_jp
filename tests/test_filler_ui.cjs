// Headless DOM/layout checks only. No preview, microphone, synthesis or API.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const { chromium } = require("playwright");
const web = path.resolve(__dirname,"../web");
const html = fs.readFileSync(path.join(web,"index.html"),"utf8")
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,"").replace(/<link\b[^>]*>/g,"");
const css = fs.readFileSync(path.join(web,"style.css"),"utf8");
const app = fs.readFileSync(path.join(web,"app.js"),"utf8").replace(/initialize\(\);\s*$/,"");

test("filler option is labelled, keyboard operable, locked only during attempts, and fits narrow screens",async(t)=>{
  const browser=await chromium.launch({channel:process.env.JEV_TEST_BROWSER_CHANNEL || undefined,headless:true});t.after(()=>browser.close());
  const page=await browser.newPage({viewport:{width:375,height:900}});
  await page.route("**/*",route=>route.abort());await page.setContent(html);await page.addStyleTag({content:css});
  await page.addScriptTag({content:app+`
    globalThis.fillerUi={
      seed(supported=true){applyConfiguration({roles:[{id:"chat"}],turn_mode:"jev",filler_supported:supported});setControls();},
      begin(isTest=false){active=createAttempt(isTest);setControls();},
      stop(){stopAttempt(active);}
    };`});
  await page.evaluate(()=>fillerUi.seed());
  const checkbox=page.getByRole("checkbox",{name:"返答前に短いフィラーを入れる"});
  assert(await checkbox.isChecked());assert(await checkbox.isEnabled());
  assert.equal(await checkbox.getAttribute("aria-describedby"),"filler-help");
  await checkbox.focus();await page.keyboard.press("Space");assert(!(await checkbox.isChecked()));
  for(const isTest of [false,true]){
    await page.evaluate(isTest=>fillerUi.begin(isTest),isTest);assert(await checkbox.isDisabled());
    await page.evaluate(()=>fillerUi.stop());assert(await checkbox.isEnabled());assert(!(await checkbox.isChecked()));
  }
  const layout=await page.evaluate(()=>({
    width:document.documentElement.scrollWidth,viewport:window.innerWidth,
    targetHeight:document.querySelector(".filler-options label").getBoundingClientRect().height,
    focusOutline:getComputedStyle(document.querySelector("#filler-enabled")).outlineStyle,
  }));
  assert(layout.width<=layout.viewport,"mobile layout must not overflow horizontally");assert(layout.targetHeight>=44);
  await page.evaluate(()=>{document.documentElement.style.fontSize="200%";});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
  await page.evaluate(()=>fillerUi.seed(false));assert(!(await checkbox.isVisible()));
});
