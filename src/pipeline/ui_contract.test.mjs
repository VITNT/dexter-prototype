import assert from "node:assert/strict";
import fs from "node:fs";

const pages = [
  { html: "index.html", script: "script.js" },
  { html: "recorder.html", script: "recorder.js" },
  { html: "trainer.html", script: "trainer.js" }
];

for (const page of pages) {
  const html = fs.readFileSync(new URL(`../../${page.html}`, import.meta.url), "utf8");
  const script = fs.readFileSync(new URL(`../../${page.script}`, import.meta.url), "utf8");

  const ids = [...html.matchAll(/\bid=["']([^"']+)["']/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length, `${page.html} contains duplicate element IDs.`);

  const handlers = [...html.matchAll(/\bonclick=["'](?:window\.)?([A-Za-z_$][\w$]*)/g)]
    .map(match => match[1]);
  for (const handler of handlers) {
    const escaped = handler.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const declaration = new RegExp(`(?:window\\.${escaped}\\s*=|function\\s+${escaped}\\s*\\()`);
    assert.match(script, declaration, `${page.html} calls missing handler ${handler} in ${page.script}.`);
  }

  const moduleSources = [...html.matchAll(/<script[^>]+src=["']([^"']+)["']/g)]
    .map(match => match[1])
    .filter(source => source.startsWith("./") || !source.includes("://"));
  for (const source of moduleSources) {
    const cleanSource = source.replace(/^\.\//, "");
    assert.ok(
      fs.existsSync(new URL(`../../${cleanSource}`, import.meta.url)),
      `${page.html} references missing script ${source}.`
    );
  }
}

console.log("UI contracts passed: unique IDs, callable click handlers, and local scripts are present.");
