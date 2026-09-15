// ─── LocationMap ────────────────────────────────────────────────────────────
// In plain English: the "Map" tab. The worker pipeline reads GPS data out of
// each photo's EXIF metadata during indexing; this view drops a dot on a
// world map for every photo that has coordinates. On first load the map
// auto-fits to all dots; if there's only one geotagged photo we zoom in on
// it. The tiles come from OpenStreetMap (network call to a/b/c.tile.osm.org).
//
/**
 * Memoria — Location Map
 *
 * Renders geotagged photos on a MapLibre GL map backed by OpenStreetMap
 * raster tiles. GPS coordinates come from each photo's EXIF, extracted
 * during indexing and stored under `exif_data._gps_lat` / `._gps_lng`.
 */

"use client";

import { useEffect, useRef } from "react";
import { MapPin } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import maplibregl, { type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { apiUrl } from "@/lib/api";
import { ENDPOINTS } from "@/lib/endpoints";
import type { LocationPoint } from "@/types";

interface LocationsResponse {
  points: LocationPoint[];
}

const OSM_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: [
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm-tiles", type: "raster", source: "osm" }],
};

export default function LocationMap() {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);

  const { data, isLoading } = useQuery<LocationsResponse>({
    queryKey: ["map-locations"],
    queryFn: async () => {
      const res = await fetch(apiUrl(ENDPOINTS.mapLocations));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (mapRef.current || !mapContainer.current) return;
    mapRef.current = new maplibregl.Map({
      container: mapContainer.current,
      style: OSM_STYLE,
      center: [0, 20],
      zoom: 1.5,
    });
    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  // Whenever the location list changes, wipe old markers and re-place them.
  // Then auto-zoom: if the points span an area, fitBounds (with 48px
  // padding and a max zoom of 10 so we don't zoom into a single building);
  // if there's exactly one point, flyTo it at zoom 8.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const points = data?.points ?? [];

    // Clear old markers before re-rendering.
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];
    if (points.length === 0) return;

    const bounds = new maplibregl.LngLatBounds();
    for (const p of points) {
      // Each marker is a hand-rolled <div> — keeps it cheap and lets
      // Tailwind classes pick up dark-mode colours.
      const el = document.createElement("div");
      el.className =
        "w-3 h-3 rounded-full bg-indigo-500 border-2 border-white shadow cursor-pointer";
      el.title = p.label ?? `Photo ${p.media_id}`;
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([p.lng, p.lat])
        .addTo(map);
      markersRef.current.push(marker);
      bounds.extend([p.lng, p.lat]);
    }
    // bounds.getNorthWest()/SouthEast() are null when no point was added,
    // and (nw.lng !== se.lng || nw.lat !== se.lat) ensures we don't pass a
    // zero-area bounds to fitBounds (which would zoom in indefinitely).
    const nw = bounds.getNorthWest();
    const se = bounds.getSouthEast();
    if (nw && se && (nw.lng !== se.lng || nw.lat !== se.lat)) {
      map.fitBounds(bounds, { padding: 48, maxZoom: 10, duration: 600 });
    } else if (points.length === 1) {
      map.flyTo({ center: [points[0].lng, points[0].lat], zoom: 8 });
    }
  }, [data]);

  const points = data?.points ?? [];

  return (
    <div className="flex-1 relative bg-neutral-950">
      <div ref={mapContainer} className="absolute inset-0" />

      <div className="absolute top-3 left-3 z-10 bg-black/60 backdrop-blur-sm rounded-lg px-3 py-1.5 pointer-events-none">
        <p className="text-xs text-neutral-300">
          <MapPin className="inline-block mr-1.5 -mt-0.5" size={12} />
          {isLoading
            ? "Loading…"
            : `${points.length} geotagged photo${points.length === 1 ? "" : "s"}`}
        </p>
      </div>

      {!isLoading && points.length === 0 && (
        <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
          <div className="bg-neutral-900/90 border border-neutral-800 rounded-lg px-5 py-4 text-center">
            <p className="text-neutral-400 text-sm">No geotagged photos yet.</p>
            <p className="text-xs text-neutral-600 mt-1">
              Photos with GPS EXIF data will appear here after indexing.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
