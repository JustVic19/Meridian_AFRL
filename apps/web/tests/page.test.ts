import assert from "node:assert/strict";
import test from "node:test";

import { labTitle, researchDisclaimer } from "../lib/site.ts";

test("home page contract names the lab and the research boundary", () => {
  assert.equal(labTitle, "Financial Intelligence Lab");
  assert.match(researchDisclaimer, /not indicative of future results/);
  assert.match(researchDisclaimer, /No live capital/);
});
