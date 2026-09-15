# Memoria frontend

The Memoria web UI uses Next.js App Router, React, TypeScript, Tailwind CSS,
and TanStack Query. It runs on port 5004 and connects to the FastAPI backend.
For the complete application, follow the [deployment guide](../README.md).

## Development

With Node.js 20+ and the backend running, execute from `frontend/`:

```bash
npm ci
npm run dev
```

Open [the web UI](http://localhost:5004). Requests to `/api/*` are proxied to
`http://127.0.0.1:8000` by default. Set `MEMORIA_INTERNAL_API_URL` before starting
or building if the backend runs elsewhere. `NEXT_PUBLIC_API_URL` can override
the browser API base; it is a build-time value and requires a rebuild to change.

## Production and validation

```bash
npm run lint
npm run build
npm start
```

The build produces a standalone server and copies public assets and generated
static files into its output. `npm start` defaults to `127.0.0.1:5004`;
`MEMORIA_WEB_HOST` and `MEMORIA_WEB_PORT` override that binding. The root startup
script and Docker image manage this build for full-stack deployments.
Next.js telemetry is disabled by the package scripts.

## Source layout

- `src/app/page.tsx` composes the application views.
- `src/components/` contains library, search, faces, map, settings, and diagnostics UI.
- `src/hooks/` contains data fetching, polling, and shared image-blob management.
- `src/lib/endpoints.ts` centralizes backend URLs.
- `src/lib/constants.ts` centralizes browser persistence keys.
- `src/types/` contains shared TypeScript interfaces.

Preserve grid virtualization and blob reference counting when changing image
loading. Server processing settings use API mutations; presentation preferences
are stored in the browser. Detailed telemetry is fetched when diagnostics opens.
