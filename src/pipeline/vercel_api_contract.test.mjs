import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const configHandler = require("../../api/config.js");
const translateHandler = require("../../api/translate.js");

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
assert.equal(defaultConfigResponse.payload.wlaslApiUrl, "/api/asl");

const previousGeminiKey = process.env.GEMINI_API_KEY;
delete process.env.GEMINI_API_KEY;
const translateResponse = mockResponse();
await translateHandler({ method: "POST", body: { aslString: "ME GO" } }, translateResponse);
assert.equal(translateResponse.statusCode, 503);
assert.match(translateResponse.payload.error, /GEMINI_API_KEY/);

if (previousWlaslUrl == null) delete process.env.WLASL_API_URL;
else process.env.WLASL_API_URL = previousWlaslUrl;
if (previousGeminiKey == null) delete process.env.GEMINI_API_KEY;
else process.env.GEMINI_API_KEY = previousGeminiKey;

console.log("Vercel API contracts passed: public runtime config and server-only Gemini key handling.");
