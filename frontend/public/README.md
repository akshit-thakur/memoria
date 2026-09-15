# Public assets

Files in this directory are served directly from the web root and copied into
the standalone production build. Keep only assets intended for public access;
do not put source photos, credentials, or runtime state here.

The application favicon lives in `src/app/favicon.ico`. UI icons are supplied
by `lucide-react`. Add static assets here only when a component references them,
and record any required attribution alongside the asset.
