#!/usr/bin/env bash
# Rebuild the memx images and (re)start the API/dashboard, worker and MCP server as the Docker Compose project
# "datagol-memx", against the external Postgres in MEMX_DOCKER_DSN (see compose.local-db.yml). The bundled
# Postgres container is never started.
#
# Usage: ./deploy.sh            rebuild and restart everything
#        ./deploy.sh --no-cache rebuild images from scratch
#        ./deploy.sh down       stop and remove the containers (the database is untouched)
#        ./deploy.sh logs       follow API, worker and MCP logs
#        ./deploy.sh ps         show container status
set -euo pipefail

cd "$(dirname "$0")"

PROJECT=datagol-memx
SERVICES=(api worker mcp)
compose() { docker compose -p "$PROJECT" -f docker-compose.yml -f compose.local-db.yml "$@"; }

case "${1:-}" in
  down) compose down --remove-orphans; exit 0 ;;
  logs) compose logs -f "${SERVICES[@]}"; exit 0 ;;
  ps)   compose ps; exit 0 ;;
esac

BUILD_ARGS=()
[[ "${1:-}" == "--no-cache" ]] && BUILD_ARGS+=(--no-cache)

# --- Check settings ------------------------------------------------------------------------------------------------
if [[ ! -f .env ]]; then
  echo "error: .env is missing. Copy .env.example to .env and set OPENROUTER_API_KEY and MEMX_DOCKER_DSN." >&2
  exit 1
fi

env_value() { grep -E "^$1=" .env | tail -n1 | cut -d= -f2- || true; }

# .env is the source of truth. Compose prefers shell variables over .env when it fills in ${...}, so a stale
# OPENROUTER_API_KEY exported in a shell profile would otherwise silently replace the one in .env.
overridden=""
while IFS='=' read -r name _; do
  if [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -n "${!name+x}" ]]; then
    overridden="$overridden $name"
    unset "$name"
  fi
done < .env
[[ -z "$overridden" ]] || echo "==> Using .env instead of shell variables:$overridden"

if [[ "$(env_value MEMX_PROVIDERS)" != "demo" && -z "$(env_value OPENROUTER_API_KEY)" ]]; then
  echo "error: OPENROUTER_API_KEY is empty in .env (or set MEMX_PROVIDERS=demo)." >&2
  exit 1
fi

if [[ -z "$(env_value MEMX_DOCKER_DSN)" ]]; then
  echo "error: MEMX_DOCKER_DSN is empty in .env (the Postgres URL as the containers see it," >&2
  echo "       e.g. postgresql://user:password@host.docker.internal:5432/memx_db)." >&2
  exit 1
fi

PUBLISH="$(env_value MEMX_PUBLISH)"
PUBLISH="${PUBLISH:-127.0.0.1:8000}"
[[ "$PUBLISH" == *:* ]] || PUBLISH="127.0.0.1:$PUBLISH"   # a bare port publishes on all interfaces
API_URL="http://${PUBLISH/#0.0.0.0/127.0.0.1}"

# --- Build ---------------------------------------------------------------------------------------------------------
echo "==> Building images"
compose build ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} "${SERVICES[@]}"

# --- Migrations ----------------------------------------------------------------------------------------------------
# The services also migrate on start; doing it first surfaces database errors before anything is replaced.
echo "==> Applying migrations"
compose run --rm --no-deps api memx migrate

# --- App -----------------------------------------------------------------------------------------------------------
echo "==> Starting ${SERVICES[*]}"
compose up -d --no-deps --force-recreate --remove-orphans "${SERVICES[@]}"

echo "==> Waiting for the API at $API_URL"
for _ in $(seq 1 60); do
  if curl -fsS "$API_URL/readyz" >/dev/null 2>&1; then
    compose ps
    echo
    compose exec -T api memx banner || echo "memx is up: $API_URL"
    exit 0
  fi
  sleep 2
done

echo "error: the API did not become ready. Recent logs:" >&2
compose logs --tail 50 "${SERVICES[@]}" >&2
exit 1
