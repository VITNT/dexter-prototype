import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const configHandler = require("../../api/config.js");
const translateHandler = require("../../api/translate.js");
const { parseBilingualTranslation } = translateHandler;

function mockResponse() {
  return {
    headers: {},
    statusCode: 200,
    payload: null,
    setHeader(name, value) {
      this.headers[name] = value;
    },
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.payload = payload;
      return this;
    }
  };
}

const previousWlaslUrl = process.env.WLASL_API_URL;
process.env.WLASL_API_URL = "https://asl.example.test/";
const configResponse = mockResponse();
configHandler({ method: "GET" }, configResponse);
assert.equal(configResponse.statusCode, 200);
assert.equal(configResponse.payload.wlaslApiUrl, "https://asl.example.test");
assert.equal(configResponse.headers["Cache-Control"], "no-store");

delete process.env.WLASL_API_URL;
const defaultConfigResponse = mockResponse();
configHandler({ method: "GET" }, defaultConfigResponse);
assert.equal(defaultConfigResponse.payload.wlaslApiUrl, "");

const previousGeminiKey = process.env.GEMINI_API_KEY;
delete process.env.GEMINI_API_KEY;
const translateResponse = mockResponse();
await translateHandler({ method: "POST", body: { aslString: "ME GO" } }, translateResponse);
assert.equal(translateResponse.statusCode, 503);
assert.match(translateResponse.payload.error, /GEMINI_API_KEY/);

assert.deepEqual(
  parseBilingualTranslation('```json\n{"khmer":"ខ្ញុំទៅហាង។","english":"i am going to the store"}\n```'),
  { khmer: "ខ្ញុំទៅហាង។", english: "I am going to the store." }
);
assert.equal(parseBilingualTranslation('{"khmer":"ខ្ញុំទៅហាង។"}'), null);

let capturedGeminiRequest;
process.env.GEMINI_API_KEY = "test-key";
const previousFetch = globalThis.fetch;
globalThis.fetch = async (_url, options) => {
  capturedGeminiRequest = JSON.parse(options.body);
  return {
    ok: true,
    status: 200,
    async json() {
      return {
        candidates: [{
          content: {
            parts: [
              { thought: true, text: "Internal reasoning" },
              { text: '{"khmer":"ខ្ញុំទៅហាង។","english":"I am going to the store."}' }
            ]
          }
        }]
      };
    }
  };
};

const bilingualResponse = mockResponse();
await translateHandler({ method: "POST", body: { aslString: "ME GO STORE" } }, bilingualResponse);
assert.equal(bilingualResponse.statusCode, 200);
assert.deepEqual(bilingualResponse.payload.translation, {
  khmer: "ខ្ញុំទៅហាង។",
  english: "I am going to the store."
});
assert.match(capturedGeminiRequest.contents[0].parts[0].text, /Khmer and English/);
assert.match(capturedGeminiRequest.contents[0].parts[0].text, /CSL\/KSL gloss/);

globalThis.fetch = previousFetch;

if (previousWlaslUrl == null) delete process.env.WLASL_API_URL;
else process.env.WLASL_API_URL = previousWlaslUrl;
if (previousGeminiKey == null) delete process.env.GEMINI_API_KEY;
else process.env.GEMINI_API_KEY = previousGeminiKey;

console.log("Vercel API contracts passed: public runtime config and server-only Gemini key handling.");
