# Moodle <-> Python bridge (Layer 5): runbook

```
Browser ──AJAX──> Moodle PHP (web service, session + capability checked)
                       │  server-side HTTP, X-API-Token
                       ▼
                  FastAPI (WSL)  ──>  recommender/  model/  data/*.csv
```
The browser never talks to the Python API, so there is no CORS, no mixed content,
and no learner id supplied by the client.

## 1. Python side (WSL)

Keep `api/`, `collector/`, `moodle-collector.py`, `tests/` and `requirements.txt` in the project root (next to `model/`, `recommender/`, `data/`).
**Delete the old `api.py`.**

```bash
pip install -r requirements.txt
export API_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')   # keep this value
# The API runs `moodle-collector.py` before each pipeline rebuild by default.
# Start it with the same MOODLE_* database settings used by the collector.
# Optional override: API_COLLECTOR_COMMAND="python /path/to/custom_collector.py"
uvicorn api:create_app --factory --host 0.0.0.0 --port 8010
pytest tests -q          # uses synthetic data, never touches your data/
```
`--host 0.0.0.0` is required: `127.0.0.1` is unreachable from a Docker container.
The service refuses to start without `API_TOKEN` (16+ chars).

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | liveness; no learner data |
| `POST /run-pipeline` | token | collector -> `build_learner_model` -> reload. Collector failure is an error; stale CSVs are not accepted. 409 if already running |
| `GET /analytics/heatmap` | token | `{topics[], learners[{learner_id, topics{topic:{mastery_level,best_score,score_trend}}}], summary}`; NaN -> null |
| `GET /recommend/{learner_id}?k=&method=` | token | ranked list + `focus` + `learner_known` |

Changes to `data/learner_model.csv` (e.g. running `python -m model` by hand) are detected automatically.

## 2. Moodle side (5.2, Docker)

Plugins live under `public/` since Moodle 5.1. Copy the two folders:

```
moodle/public/local/instructor_analytics   ->  <moodle>/public/local/instructor_analytics
moodle/public/blocks/personalized_recommendation -> <moodle>/public/blocks/personalized_recommendation
```
Remove your old plugin files first (the old `lang/en/instructor_analytics.php` was misnamed and never loaded).
Example: `docker cp moodle/public/local/instructor_analytics <container>:<moodle>/public/local/` (same for the block),
then fix ownership to the web user. A bind mount is better for iteration.

1. Browser: **Site administration > Notifications** and run the upgrade (new capabilities/services).
2. **Site administration > Development > Purge caches.**
3. Configure each plugin (**Plugins > Local plugins > Recommendation analytics**, **Plugins > Blocks > Your Personalized Assistant**):
   API URL `http://host.docker.internal:8010`, the same API token, and (block) the method.
   Tip: to keep the token out of the database, set it in `config.php` via `$CFG->forced_plugin_settings`.
4. Docker networking (moodle-docker setup). Moodle's own webserver is published on host port **8000**, so the API
   uses **8010**. The container must be able to resolve the host: add this to `moodle-docker/local.yml`
   (the file `bin/moodle-docker-compose` already includes), then recreate the containers:
   ```yaml
   services:
     webserver:
       extra_hosts:
         - "host.docker.internal:host-gateway"
   ```
   ```bash
   bin/moodle-docker-compose up -d
   # with uvicorn running, this must print JSON:
   bin/moodle-docker-compose exec webserver curl -s http://host.docker.internal:8010/health
   ```
   If `curl` is missing in the image: `bin/moodle-docker-compose exec webserver php -r 'echo file_get_contents("http://host.docker.internal:8010/health");'`
5. Moodle can block server-side requests to internal hosts and non-standard ports. Check
   **Site administration > General > Security > HTTP security**: *cURL blocked hosts list* and
   *cURL allowed ports list*. If the dashboard says the service "could not be reached" although step 4 works,
   allow the host and port 8010 there (dev only; keep the restriction in production).
6. Add the block to the course page (**Add a block > Your Personalized Assistant**). Students see the card;
   teachers see the dashboard under the course **More** menu > *Recommendation analytics*.

Learner ids: the Python model's `learner_id` must equal the Moodle `user.id`
(true for the current test cohort: ids 3 and 5).

## 3. What was wrong in the original skeleton (and is fixed)

| # | Problem | Fix |
|---|---|---|
| 1 | Dashboard JS called `localhost:8000` from the browser: from a container/browser that is the wrong machine, plus CORS | Server-side PHP client + Moodle web services |
| 2 | `/run-pipeline` open to anyone; CORS `*` | Shared-secret token, no CORS, capability `runpipeline` |
| 3 | Heatmap crashed with `ValueError: Out of range float values` when `score_trend`/`best_score` were NaN (already true for view-only learners) | NaN -> `null`, strict-JSON tests |
| 4 | API read a relative path (`data/...`) so results depended on the launch directory | Absolute path from project root / `RECSYS_DATA_DIR` |
| 5 | Local plugin lang file misnamed (`instructor_analytics.php`) | `local_instructor_analytics.php` |
| 6 | No `db/access.php`: capability strings pointed at capabilities that did not exist | Defined (view / runpipeline / block view) |
| 7 | Inline `<script>`, `innerHTML` with raw JSON, Bootstrap-4 classes, `fontsize` typo | AMD modules, DOM `textContent`, BS5-safe markup, `styles.css` |
| 8 | Dashboard showed every learner in the model regardless of course | Only students enrolled in *this* course, names from Moodle, optional anonymise setting |
| 9 | Student block blocked page render on the API and had no failure state | Async shell; "temporarily unavailable" state; per-user, never client-supplied id |
| 10 | Recommendations could point at hidden/unavailable activities | Resolved through `get_fast_modinfo` (visible only, real URL) |
| 11 | `/run-pipeline` re-ran two runs at once; partial CSV reads possible | Single-run lock (409), snapshot reload, last good snapshot kept |

## 4. Not verified / known limits

* **PHP was syntax-parsed only** (no Moodle instance here). The first run in your Moodle 5.2 is the real test;
  please send me the exact error text if anything fails.
* The AMD builds in `amd/build/` are minified copies of `amd/src/` (named modules). If you edit `src`, rebuild with Moodle's `grunt amd` or re-minify.
* `/run-pipeline` is synchronous. The collector has a 300-second timeout; Moodle's request timeout defaults to 600 seconds so the HTTP request can cover collection and model rebuilding.
* `est_minutes` appears only if `activity_catalog.csv` has an `estimated_minutes` column (it does not today).
* Your Moodle test cohort has 2 learners, both `high` everywhere, so every recommendation is a popularity fallback
  (flagged `via_fallback`). `python -m tests.demo_data <dir>` writes an 11-learner cohort covering every band for demos.
* **`learner_model_oulad.csv` is broken upstream:** 173,912 of 203,140 rows (85.6%) have a blank `topic`
  and `sequence_position` (all are quiz-only rows). The ETL is not joining quiz features to topics. The API ignores these rows
  so it will not crash, but the OULAD recommender input needs fixing before Phase 2 evaluation.
