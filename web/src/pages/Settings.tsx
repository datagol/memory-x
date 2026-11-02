import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, getToken, setToken } from "../lib/api";
import { CATEGORY_LABEL, catVar, pct, topicLabel } from "../lib/meta";
import { Card, ErrorBox, Loading, useToast } from "../components/ui";

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
          <p>Connection and the thresholds this memx server runs with. Thresholds are set with <code>MEMX_*</code> environment variables on the server.</p>
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
              <dt>Extraction</dt><dd>{c.models.extractor}</dd>
              <dt>Embeddings</dt><dd>{c.models.embedding}</dd>
              <dt>Judge</dt><dd>{c.models.judge} (Jev)</dd>
              <dt>Insert mode</dt><dd>{c.insert_mode}</dd>
              <dt>Window size</dt><dd>{c.window_tokens.toLocaleString()} tokens</dd>
            </dl>
          ) : config.error ? <ErrorBox error={config.error} /> : <Loading />}
        </Card>
      </div>
      {c && (
        <div className="grid cols-2">
          <Card title="Deduplication">
            <dl className="kv">
              <dt>Similarity bands</dt><dd>below {c.bands.low}: unrelated · {c.bands.low}–{c.bands.high}: Jev decides · above {c.bands.high}: duplicate if no new words</dd>
              <dt>Word overlap for auto</dt><dd>{pct(c.bands.guard_min_jaccard)}</dd>
              <dt>“Same” needs</dt><dd>{pct(c.verdict.same_min)} from Jev</dd>
              <dt>Replace / covers needs</dt><dd>{pct(c.verdict.min_prob)} from Jev, otherwise kept as related</dd>
              <dt>Category needs</dt><dd>{pct(c.sort.min_prob)} probability, else “Other”</dd>
              <dt>Injection</dt><dd>flag at {pct(c.sort.injection_flag)}, reject at {pct(c.sort.injection_reject)}</dd>
            </dl>
          </Card>
          <Card title="Recall">
            <dl className="kv">
              <dt>Budget</dt><dd>{c.recall.budget_tokens.toLocaleString()} tokens, {pct(c.recall.budget_safety)} usable</dd>
              <dt>Relevance cut-off</dt><dd>{pct(c.recall.grade_floor)}</dd>
              <dt>Max facts</dt><dd>{c.recall.k_max}, from up to {c.recall.grade_max} graded</dd>
              <dt>Grading</dt><dd>{c.recall.grade_enabled ? "Jev" : "off (search rank only)"}</dd>
              <dt>Trust</dt><dd>user {c.trust.user} · assistant {c.trust.assistant} · documents {c.trust.text_default}</dd>
              <dt>Half-lives</dt><dd>{Object.entries(c.decay.half_lives).map(([k, v]) => `${CATEGORY_LABEL[k as keyof typeof CATEGORY_LABEL] ?? k} ${v ? `${v} d` : "∞"}`).join(" · ")}</dd>
            </dl>
          </Card>
        </div>
      )}
      {c && (
        <div className="grid cols-2">
          <Card title="Entities and themes" sub="Links between facts">
            <dl className="kv">
              <dt>Entities</dt><dd>{c.entities.enabled ? "extracted with every fact, shared within a tenant" : "off"}</dd>
              <dt>Merging spellings</dt><dd>“Marcus” joins “Marcus Oyelaran” when Jev is {pct(c.entities.merge_min)} sure they are the same</dd>
              <dt>Theme needs</dt><dd>{pct(c.entities.topic_min)} probability from Jev, else none</dd>
            </dl>
          </Card>
          <Card title="Walk at recall" sub="Expands ranked recall through shared entities">
            <dl className="kv">
              <dt>Walk</dt><dd>{c.walk.enabled ? `up to ${c.walk.hops} hops from the top ${c.walk.seeds} search hits` : "off"}</dd>
              <dt>Per hop</dt><dd>score × {c.walk.decay}, split across the entity's other facts</dd>
              <dt>Hub cut</dt><dd>entities in more than {pct(c.walk.hub_share)} of visible facts (and at least {c.walk.hub_min_degree}) are not walked</dd>
              <dt>Themes</dt><dd>hops through a shared theme count × {c.walk.topic_weight}</dd>
              <dt>Adds at most</dt><dd>{c.walk.cap} facts to the grade request</dd>
              <dt>Walked facts need</dt><dd>{pct(c.walk.grade_floor)} relevance from Jev (“useful”), against {pct(c.recall.grade_floor)} for search hits</dd>
              <dt>Final score</dt><dd>× {c.walk.score_factor} for a fact reached by a hop, so direct hits win ties</dd>
            </dl>
          </Card>
        </div>
      )}
      {c && (
        <Card title="Themes" sub="Jev gives every fact one theme from this fixed list, inside the same request that picks its category" flush>
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
        <Card title="Categories" sub="Jev sorts every fact into one of these" flush>
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
