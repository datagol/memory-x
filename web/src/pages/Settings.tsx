import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, getToken, setToken } from "../lib/api";
import { CATEGORY_LABEL, catVar, pct, topicLabel } from "../lib/meta";
import { Card, ErrorBox, Loading, useToast } from "../components/ui";

const Env = ({ n }: { n: string }) => <span className="env-hint">{n.split(" ").map((x) => `MEMX_${x}`).join(" · ")}</span>;
const PerScope = () => <span className="scope-chip" title="A scope can override this on the Memory page, under Settings for this scope">per scope</span>;

export function SettingsPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [token, setTokenValue] = useState(getToken());
  const config = useQuery({ queryKey: ["config"], queryFn: api.config });
  const c = config.data;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p>The models and thresholds this server runs with, for every scope. They are read-only here: they affect every agent at once, and some (the embedding model, categories, themes) can't change without reprocessing stored memory. Change them in <code>.env</code> with the variable shown, then run <code>docker compose up -d</code>. Values tagged <span className="scope-chip">per scope</span> can also be overridden for one scope and the scopes below it, on the Memory page.</p>
        </div>
      </div>
      <div className="grid cols-2">
        <Card title="API token" sub="Kept in this browser only">
          <form className="row" onSubmit={(e) => { e.preventDefault(); setToken(token.trim()); qc.invalidateQueries(); toast(token.trim() ? "Token saved" : "Token cleared"); }}>
            <input id="api-token" className="input" style={{ flex: 1 }} type="password" value={token} onChange={(e) => setTokenValue(e.target.value)} placeholder="MEMX_API_TOKEN" />
            <button className="btn primary" type="submit">Save</button>
          </form>
        </Card>
        <Card title="Models">
          {c ? (
            <dl className="kv">
              <dt>Extraction</dt><dd>{c.models.extractor}<Env n="EXTRACTOR_MODEL" /></dd>
              <dt>Embeddings</dt><dd>{c.models.embedding}<Env n="EMBEDDING_MODEL" /> <span className="small muted">changing it needs every fact re-embedded</span></dd>
              <dt>Judge</dt><dd>{c.models.judge} (Jev)<Env n="JEV_MODEL" /></dd>
              <dt>Insert mode</dt><dd>{c.insert_mode}<Env n="INSERT_MODE" /></dd>
              <dt>Window size</dt><dd>{c.window_tokens.toLocaleString()} tokens<Env n="WINDOW_TOKENS" /></dd>
            </dl>
          ) : config.error ? <ErrorBox error={config.error} /> : <Loading />}
        </Card>
      </div>
      {c && (
        <div className="grid cols-2">
          <Card title="Deduplication">
            <dl className="kv">
              <dt>Similarity bands</dt><dd>below {c.bands.low}: unrelated · {c.bands.low}–{c.bands.high}: Jev decides · above {c.bands.high}: duplicate if no new words<Env n="BAND_LOW BAND_HIGH" /></dd>
              <dt>Word overlap for auto</dt><dd>{pct(c.bands.guard_min_jaccard)}<Env n="GUARD_MIN_JACCARD" /></dd>
              <dt>“Same” needs</dt><dd>{pct(c.verdict.same_min)} from Jev<Env n="VERDICT_SAME_MIN" /></dd>
              <dt>Replace / covers needs</dt><dd>{pct(c.verdict.min_prob ?? c.verdict.replace_min)} / {pct(c.verdict.min_prob ?? c.verdict.covers_min)} from Jev, and the same attribute; otherwise kept as related<Env n="VERDICT_REPLACE_MIN VERDICT_COVERS_MIN" /></dd>
              {c.verdict.confirm && <><dt>Borderline check</dt><dd>{pct(c.verdict.confirm_band[0])}–{pct(c.verdict.confirm_band[1])}: replace only if Jev says both cannot be true<Env n="VERDICT_CONFIRM" /></dd></>}
              <dt>Category needs</dt><dd>{pct(c.sort.min_prob)} probability, else “Other”<Env n="SORT_MIN_PROB" /></dd>
              <dt>Injection</dt><dd>flag at {pct(c.sort.injection_flag)}, reject at {pct(c.sort.injection_reject)}<Env n="INJECTION_FLAG INJECTION_REJECT" /></dd>
            </dl>
          </Card>
          <Card title="Recall">
            <dl className="kv">
              <dt>Budget <PerScope /></dt><dd>{c.recall.budget_tokens.toLocaleString()} tokens, {pct(c.recall.budget_safety)} usable<Env n="BUDGET_TOKENS BUDGET_SAFETY" /></dd>
              <dt>Relevance cut-off</dt><dd>{pct(c.recall.grade_floor)}<Env n="GRADE_FLOOR" /></dd>
              <dt>Max facts</dt><dd>{c.recall.k_max}, from up to {c.recall.grade_max} graded<Env n="RECALL_K_MAX GRADE_MAX" /></dd>
              <dt>Grading</dt><dd>{c.recall.grade_enabled ? "Jev" : "off (search rank only)"}<Env n="GRADE_ENABLED" /></dd>
              <dt>Trust</dt><dd>user {c.trust.user} · assistant {c.trust.assistant} · documents {c.trust.text_default} <PerScope /><Env n="USER_TRUST ASSISTANT_TRUST DEFAULT_TEXT_TRUST" /></dd>
              <dt>Half-lives</dt><dd>{Object.entries(c.decay.half_lives).map(([k, v]) => `${CATEGORY_LABEL[k as keyof typeof CATEGORY_LABEL] ?? k} ${v ? `${v} d` : "∞"}`).join(" · ")}<Env n="HALF_LIVES" /></dd>
              <dt>Upcoming events <PerScope /></dt><dd>{c.event_ttl_days ? `expire ${c.event_ttl_days} days after their date` : "never hidden; they fade with the event half-life"}<Env n="EVENT_TTL_DAYS" /></dd>
            </dl>
          </Card>
        </div>
      )}
      {c && (
        <div className="grid cols-2">
          <Card title="Entities and themes" sub="Links between facts">
            <dl className="kv">
              <dt>Entities</dt><dd>{c.entities.enabled ? "extracted with every fact, shared within a tenant" : "off"}<Env n="ENTITIES_ENABLED" /></dd>
              <dt>Merging spellings</dt><dd>“Marcus” joins “Marcus Oyelaran” when Jev is {pct(c.entities.merge_min)} sure they are the same<Env n="ENTITY_MERGE_MIN" /></dd>
              <dt>Theme needs</dt><dd>{pct(c.entities.topic_min)} probability from Jev, else none<Env n="TOPIC_MIN" /></dd>
            </dl>
          </Card>
          <Card title="Walk at recall" sub="Expands ranked recall through shared entities">
            <dl className="kv">
              <dt>Walk <PerScope /></dt><dd>{c.walk.enabled ? `up to ${c.walk.hops} hops from the top ${c.walk.seeds} search hits` : "off"}<Env n="WALK_ENABLED WALK_HOPS WALK_SEEDS" /></dd>
              <dt>Per hop</dt><dd>score × {c.walk.decay}, split across the entity's other facts<Env n="WALK_DECAY" /></dd>
              <dt>Hub cut</dt><dd>entities in more than {pct(c.walk.hub_share)} of visible facts (and at least {c.walk.hub_min_degree}) are not walked<Env n="WALK_HUB_SHARE WALK_HUB_MIN_DEGREE" /></dd>
              <dt>Themes</dt><dd>hops through a shared theme count × {c.walk.topic_weight}<Env n="WALK_TOPIC_WEIGHT" /></dd>
              <dt>Adds at most</dt><dd>{c.walk.cap} facts to the grade request<Env n="WALK_CAP" /></dd>
              <dt>Walked facts need</dt><dd>{pct(c.walk.grade_floor)} relevance from Jev (“useful”), against {pct(c.recall.grade_floor)} for search hits<Env n="WALK_GRADE_FLOOR" /></dd>
              <dt>Final score</dt><dd>× {c.walk.score_factor} for a fact reached by a hop, so direct hits win ties<Env n="WALK_SCORE_FACTOR" /></dd>
            </dl>
          </Card>
        </div>
      )}
      {c && (
        <Card title="Themes" sub="Fixed list, part of the code. Jev gives every fact one theme, inside the same request that picks its category" flush>
          <table className="t"><tbody>
            {c.topics.map((t) => (
              <tr key={t.key}>
                <td style={{ width: 220 }}><span className="chip topic">{topicLabel(t.key)}</span></td>
                <td className="muted">{t.definition}</td>
              </tr>
            ))}
          </tbody></table>
        </Card>
      )}
      {c && (
        <Card title="Categories" sub="Fixed list, part of the code. Jev sorts every fact into one of these" flush>
          <table className="t"><tbody>
            {c.categories.map((cat) => (
              <tr key={cat.key}>
                <td style={{ width: 180 }}><span className="chip"><span className="sw" style={{ background: catVar(cat.key) }} />{CATEGORY_LABEL[cat.key]}</span></td>
                <td className="muted">{cat.definition}</td>
              </tr>
            ))}
          </tbody></table>
        </Card>
      )}
    </div>
  );
}
