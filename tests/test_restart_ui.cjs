// API-free regression tests for setup, same-page server restart, and stale callbacks.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8")
  .replace(/initialize\(\);\s*$/, "");
const configuration = (token, extra = {}) => ({csrf_token:token, roles:[{id:"first",label:"一番"},{id:"second",label:"二番"}], voice:"marin", turn_mode:"live", interruption_policy:"transcript", ...extra});
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };
const flush = () => new Promise((resolve) => setImmediate(resolve));

function harness(t) {
  const elements = new Map(), contexts = [], nodes = [], streams = [], sockets = [];
  const gets = [], posts = [], getQueue = [], postQueue = [], micQueue = [], timers = new Set(), clipGets = [], clipReplies = new Map();
  const element = () => ({value:"",textContent:"",hidden:false,disabled:false,checked:false,childNodes:[],events:new Map(),
    style:{setProperty(){}},classList:{add(){},remove(){},toggle(){}},
    addEventListener(type,fn){this.events.set(type,fn);},focus(){},append(...children){this.childNodes.push(...children);},
    replaceChildren(...children){this.childNodes=children;},scrollTop:0,scrollHeight:100,clientHeight:100});
  class Context {
    constructor(){ this.state="suspended";this.sampleRate=48000;this.destination={};this.audioWorklet={addModule:async()=>{}};contexts.push(this); }
    async resume(){this.state="running";this.resumed=true;}
    async close(){this.state="closed";}
    createMediaStreamSource(){return {connect(){},disconnect(){this.disconnected=true;}};}
  }
  class Worklet {
    constructor(){this.port={postMessage(){},onmessage:null};nodes.push(this);}
    connect(){} disconnect(){this.disconnected=true;}
  }
  class Socket {
    static OPEN=1;static CONNECTING=0;static CLOSED=3;
    constructor(url){this.url=new URL(url);this.readyState=0;this.bufferedAmount=0;this.events=new Map();this.sent=[];sockets.push(this);}
    addEventListener(type,fn){this.events.set(type,fn);}
    emit(type,event={}){this.events.get(type)?.(event);}
    open(){this.readyState=1;this.emit("open");}
    send(data){this.sent.push(data);}
    close(){this.readyState=3;this.emit("close");}
  }
  function stream(){
    const track={stopped:false,stop(){this.stopped=true;},addEventListener(type,fn){if(type==="ended")this.onended=fn;}};
    const value={track,getTracks:()=>[track],getAudioTracks:()=>[track]};streams.push(value);return value;
  }
  const response = (data, status=200) => ({ok:status<400,status,json:async()=>data});
  const sandbox = {
    document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element,createDocumentFragment:element},
    window:{AudioContext:Context,addEventListener(){}},navigator:{mediaDevices:{getUserMedia:async()=>micQueue.length?await micQueue.shift()():stream()}},
    WebSocket:Socket,AudioWorkletNode:Worklet,AbortController,ArrayBuffer,Int16Array,URL,TypeError,
    location:{href:"http://localhost:8783/",protocol:"http:"},performance:{now:()=>100},
    setTimeout(fn,ms){const timer=setTimeout(fn,ms);timers.add(timer);return timer;},clearTimeout(timer){clearTimeout(timer);timers.delete(timer);},
    async fetch(url,options){
      if(url.startsWith("/api/fillers/")){
        clipGets.push(url);const value=clipReplies.get(url);
        if(value instanceof Error)throw value;
        if(typeof value==="function")return await value(options);
        return {ok:Boolean(value),arrayBuffer:async()=>value.slice(0)};
      }
      if(url==="/api/config"){
        gets.push(options);assert.equal(options.cache,"no-store");
        assert(getQueue.length,"unexpected config request");return response(await getQueue.shift()());
      }
      assert.equal(url,"/api/sessions");posts.push(options);
      if(postQueue.length)return await postQueue.shift()();
      return response({id:`session-${posts.length}`,messages:[]});
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(source+`\nglobalThis.controls={startConversation,initialize,stop(){if(active)stopAttempt(active);},
    seed(data){applyConfiguration(data);setControls();},getActive(){return active;},getConfig(){return config;},
    capture(attempt){handleWorkletMessage(attempt,{type:"capture",level:0.02,buffer:new ArrayBuffer(960)});},
    preload:preloadFillers,catalog:fillerCatalog,testAudio,
    toggle(value){ui.fillerEnabled.checked=value;ui.fillerEnabled.events.get("change")();}};`,sandbox);
  t.after(()=>{sandbox.controls.stop();for(const timer of timers)clearTimeout(timer);});
  return {...sandbox.controls,elements,contexts,nodes,streams,sockets,gets,posts,getQueue,postQueue,micQueue,stream,response,clipGets,clipReplies};
}

function fillerConfig(token, voice="marin", extra={}) {
  return configuration(token,{turn_mode:"jev",interruption_policy:"semantic",filler_supported:true,
    filler_clips:[{id:`${voice}-content-hash`,url:`/api/fillers/${voice}-content-hash.pcm`,text:"ええと",audio_bytes:24000,audio_seconds:.5}],...extra});
}

test("Jev filler defaults on, preloads before the session, and locks its setting throughout a conversation",async(t)=>{
  const h=harness(t), config=fillerConfig("token");h.seed(config);
  h.clipReplies.set(config.filler_clips[0].url,new ArrayBuffer(24000));
  assert.equal(h.elements.get("filler-options").hidden,false);assert.equal(h.elements.get("filler-enabled").checked,true);
  assert.equal(h.elements.get("filler-enabled").disabled,false);
  h.getQueue.push(()=>config);const starting=h.startConversation();
  assert.equal(h.elements.get("filler-enabled").disabled,true);await starting;
  assert.equal(JSON.parse(h.posts[0].body).filler_enabled,true);assert.equal(h.getActive().fillerClips.size,1);
  h.sockets[0].open();assert.equal(h.elements.get("filler-enabled").disabled,true);
  h.stop();assert.equal(h.elements.get("filler-enabled").disabled,false);
  assert.match(h.elements.get("filler-help").textContent,/対話を始める前/);
});

test("filler off starts without clip requests and the preference survives restart",async(t)=>{
  const h=harness(t), config=fillerConfig("first");h.seed(config);h.toggle(false);
  h.getQueue.push(()=>config);await h.startConversation();assert.equal(JSON.parse(h.posts[0].body).filler_enabled,false);
  assert.equal(h.clipGets.length,0);h.stop();
  h.getQueue.push(()=>fillerConfig("second"));await h.startConversation();
  assert.equal(JSON.parse(h.posts[1].body).filler_enabled,false);assert.equal(h.elements.get("filler-enabled").checked,false);
});

test("each fresh voice gets its own PCM cache while unchanged voice is fetched only once",async(t)=>{
  const h=harness(t), first=fillerConfig("first","marin"), next=fillerConfig("next","coral");h.seed(first);
  for(const config of [first,next])h.clipReplies.set(config.filler_clips[0].url,new ArrayBuffer(24000));
  await h.preload(first);h.getQueue.push(()=>first);await h.startConversation();h.stop();
  assert.equal(h.clipGets.length,1);
  h.getQueue.push(()=>next);await h.startConversation();
  assert.equal(h.clipGets.length,2);assert(h.getActive().fillerClips.has("coral-content-hash"));
  assert(!h.getActive().fillerClips.has("marin-content-hash"));
});

test("failed or truncated filler download continues a normal session with filler disabled",async(t)=>{
  for(const reply of [new Error("offline"),new ArrayBuffer(4)]){
    const h=harness(t),config=fillerConfig("token");h.seed(config);h.clipReplies.set(config.filler_clips[0].url,reply);
    h.getQueue.push(()=>config);await h.startConversation();
    assert(h.getActive());assert.equal(h.getActive().fillerEnabled,false);
    assert.equal(JSON.parse(h.posts[0].body).filler_enabled,false);
    assert.match(h.elements.get("filler-help").textContent,/読み込めなかった/);h.stop();
  }
});

test("unsupported mode hides fillers and server disabling remains visible",async(t)=>{
  const h=harness(t);h.seed(configuration("live"));assert.equal(h.elements.get("filler-options").hidden,true);
  assert.equal(h.elements.get("filler-enabled").disabled,true);
  const config=fillerConfig("jev");h.seed(config);h.clipReplies.set(config.filler_clips[0].url,new ArrayBuffer(24000));
  h.getQueue.push(()=>config);h.postQueue.push(()=>h.response({id:"session",messages:[],filler_enabled:false}));
  await h.startConversation();assert.equal(h.getActive().fillerEnabled,false);
  assert.match(h.elements.get("filler-help").textContent,/サーバーのフィラー音声/);
});

test("cancelling a pending filler preload cannot create a stale session or affect a later attempt",async(t)=>{
  const h=harness(t),config=fillerConfig("old"),waiting=deferred();h.seed(config);
  h.clipReplies.set(config.filler_clips[0].url,()=>waiting.promise);h.getQueue.push(()=>config);
  const starting=h.startConversation();await flush();h.stop();h.toggle(false);
  h.getQueue.push(()=>fillerConfig("new"));await h.startConversation();const current=h.getActive();
  waiting.resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(24000)});await starting;
  assert.equal(h.getActive(),current);assert.equal(h.posts.length,1);assert.equal(current.csrfToken,"new");
});

test("filler catalogs reject cross-origin or oversized media without requesting it",async(t)=>{
  const h=harness(t);
  for(const change of [{url:"//other.example/clip.pcm"},{url:"https://other.example/clip.pcm"},{audio_bytes:9999999},{audio_seconds:20},{text:""}]){
    const config=fillerConfig("token");Object.assign(config.filler_clips[0],change);
    assert.equal((await h.preload(config)).size,0);
  }
  assert.equal(h.clipGets.length,0);
});

test("filler setting is disabled during the local audio test and becomes editable after stop",async(t)=>{
  const h=harness(t);h.seed(fillerConfig("token"));await h.testAudio();
  assert.equal(h.elements.get("filler-enabled").disabled,true);h.stop();
  assert.equal(h.elements.get("filler-enabled").disabled,false);assert.equal(h.posts.length,0);assert.equal(h.clipGets.length,0);
});

test("same-page restart refreshes tokens, preserves the role, and sends PCM through the new socket only", async(t)=>{
  const h=harness(t);h.seed(configuration("old"));h.elements.get("role").value="second";
  h.getQueue.push(()=>configuration("server-A"));await h.startConversation();
  const old=h.getActive(), first=h.sockets[0];first.open();h.capture(old);
  assert.equal(old.interruptionPolicy,"transcript");
  assert.equal(h.posts[0].headers["X-CSRF-Token"],"server-A");
  assert.equal(first.url.searchParams.get("token"),"server-A");
  assert.equal(JSON.parse(h.posts[0].body).role_id,"second");
  first.close();assert.equal(h.getActive(),null);assert(h.streams[0].track.stopped);
  assert.equal(h.contexts[0].state,"closed");assert.equal(h.nodes[0].port.onmessage,null);
  h.getQueue.push(()=>configuration("server-B",{turn_mode:"jev",interruption_policy:"semantic"}));await h.startConversation();
  const current=h.getActive(), second=h.sockets[1];second.open();
  assert.equal(h.posts[1].headers["X-CSRF-Token"],"server-B");
  assert.equal(second.url.searchParams.get("token"),"server-B");assert.equal(current.interruptionPolicy,"semantic");
  first.emit("error");first.emit("message",{data:JSON.stringify({type:"closed"})});h.streams[0].track.onended();
  h.capture(old);h.capture(current);
  assert.equal(h.getActive(),current);assert(!h.streams[1].track.stopped);
  assert.equal(first.sent.filter((v)=>v instanceof ArrayBuffer).length,1);
  assert.equal(second.sent.filter((v)=>v instanceof ArrayBuffer).length,1);
  assert.equal(h.posts.length,2);assert.equal(h.gets.length,2);
});

test("fresh Live config replaces legacy energy policy and missing policy stays transcript-based", async(t)=>{
  const h=harness(t);h.seed(configuration("old",{interruption_policy:"energy"}));
  h.getQueue.push(()=>configuration("current",{interruption_policy:undefined}));await h.startConversation();
  const current=h.getActive(), socket=h.sockets[0];socket.open();
  assert.equal(current.interruptionPolicy,"transcript");
  current.pendingPlayback.add("speaking");
  for(let index=0;index<20;index+=1)h.capture(current);
  assert.equal(current.generation,0);
  assert(current.pendingPlayback.has("speaking"));
  assert.equal(socket.sent.filter((value)=>value instanceof ArrayBuffer).length,20);
  assert.equal(socket.sent.filter((value)=>typeof value==="string"&&JSON.parse(value).type==="user_activity").length,0);
  assert.match(h.elements.get("connection-hint").textContent,/認識した発話/);
});

test("initially unavailable server can recover from the same start button without page reload",async(t)=>{
  const h=harness(t);h.getQueue.push(()=>{throw new TypeError("offline");});await h.initialize();
  assert.equal(h.getConfig(),null);assert.equal(h.elements.get("start").disabled,false);
  assert.equal(h.elements.get("start").textContent,"接続を確認して始める");
  h.getQueue.push(()=>configuration("recovered"));await h.startConversation();
  assert.equal(h.posts.length,1);assert.equal(h.posts[0].headers["X-CSRF-Token"],"recovered");
  assert.equal(h.elements.get("error").hidden,true);
});

test("cancelled config refresh cannot overwrite a new attempt or create a duplicate session",async(t)=>{
  const h=harness(t), stale=deferred();h.seed(configuration("initial"));h.getQueue.push(()=>stale.promise);
  const first=h.startConversation();assert(h.contexts[0].resumed,"resume is invoked before config returns");h.stop();
  h.getQueue.push(()=>configuration("current"));await h.startConversation();const active=h.getActive();
  stale.resolve(configuration("stale"));await first;
  assert.equal(h.getConfig().csrf_token,"current");assert.equal(h.getActive(),active);assert.equal(h.posts.length,1);
  assert.equal(h.streams.length,1);assert.equal(h.contexts[0].state,"closed");
});

test("a late initial config response cannot replace the running attempt's config",async(t)=>{
  const h=harness(t), initial=deferred();h.getQueue.push(()=>initial.promise);const initializing=h.initialize();
  h.getQueue.push(()=>configuration("current"));await h.startConversation();
  initial.resolve(configuration("initial-old"));await initializing;
  assert.equal(h.getConfig().csrf_token,"current");assert.equal(h.getActive().csrfToken,"current");
  assert.equal(h.posts.length,1);assert.equal(h.elements.get("start").disabled,true);
});

test("microphone permission completing after cancellation stops the old track without affecting the new microphone",async(t)=>{
  const h=harness(t), microphone=deferred();h.getQueue.push(()=>configuration("old"));h.micQueue.push(()=>microphone.promise);
  const first=h.startConversation();await flush();h.stop();
  h.getQueue.push(()=>configuration("new"));await h.startConversation();const current=h.getActive();
  const late=h.stream();microphone.resolve(late);await first;
  assert(late.track.stopped);assert(!current.stream.track.stopped);assert.equal(h.posts.length,1);
});

test("cancelled POST never opens an old socket; 403 does not retry session creation",async(t)=>{
  const h=harness(t), post=deferred();h.getQueue.push(()=>configuration("old"));h.postQueue.push(()=>post.promise);
  const first=h.startConversation();await flush();assert.equal(h.posts.length,1);h.stop();
  h.getQueue.push(()=>configuration("new"));await h.startConversation();const current=h.getActive();
  post.resolve(h.response({id:"old-session",messages:[]}));await first;
  assert.equal(h.sockets.length,1);assert.equal(h.getActive(),current);assert.equal(h.posts.length,2);
  h.stop();h.getQueue.push(()=>configuration("next"));h.postQueue.push(()=>h.response({error:"試験の403"},403));
  await h.startConversation();assert.equal(h.getActive(),null);assert.equal(h.posts.length,3);
  assert.equal(h.elements.get("start").disabled,false);assert(h.streams.at(-1).track.stopped);
});
