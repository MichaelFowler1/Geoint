const map = L.map("map", { zoomControl: true }).setView([38.9, -77.04], 9);

L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "&copy; OpenStreetMap &copy; CARTO",
  maxZoom: 19,
}).addTo(map);

// ---------- Live air tracks ----------
const acLayer = L.layerGroup().addTo(map);
const acMarkers = new Map();

function aircraftIcon(heading) {
  const rot = heading || 0;
  return L.divIcon({
    className: "",
    html: `<div style="transform:rotate(${rot}deg);color:#38e0a0;font-size:15px;line-height:1;">&#9650;</div>`,
    iconSize: [15, 15],
    iconAnchor: [7, 7],
  });
}

function renderTracks(aircraft) {
  const seen = new Set();
  for (const a of aircraft) {
    if (a.lat == null || a.lon == null) continue;
    seen.add(a.icao24);
    const label = a.callsign || a.icao24;
    const altft = a.altitude_m != null ? Math.round(a.altitude_m * 3.281) : "—";
    let m = acMarkers.get(a.icao24);
    if (m) {
      m.setLatLng([a.lat, a.lon]);
      m.setIcon(aircraftIcon(a.heading_deg));
    } else {
      m = L.marker([a.lat, a.lon], { icon: aircraftIcon(a.heading_deg) }).addTo(acLayer);
      acMarkers.set(a.icao24, m);
    }
    m.bindTooltip(`${label} · ${altft} ft`, { direction: "top", offset: [0, -8] });
  }
  for (const [id, m] of acMarkers) {
    if (!seen.has(id)) {
      acLayer.removeLayer(m);
      acMarkers.delete(id);
    }
  }
  document.getElementById("ac-count").textContent = acMarkers.size;
}

// ---------- WebSocket ----------
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/tracks`);
  const link = document.getElementById("link-state");
  ws.onopen = () => {
    link.textContent = "ON";
    link.style.color = "#38e0a0";
  };
  ws.onclose = () => {
    link.textContent = "··";
    link.style.color = "#ff5a4d";
    setTimeout(connectWS, 4000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "tracks") renderTracks(msg.aircraft);
  };
  setInterval(() => {
    if (ws.readyState === 1) ws.send("ping");
  }, 25000);
}
connectWS();

// ---------- Imagery detections ----------
const detLayer = L.layerGroup().addTo(map);
let imageOverlay = null;

function detIcon() {
  return L.divIcon({
    className: "",
    html: `<div style="width:9px;height:9px;border:2px solid #ffae42;border-radius:1px;box-shadow:0 0 6px #ffae42;"></div>`,
    iconSize: [9, 9],
    iconAnchor: [4, 4],
  });
}

function renderDetections(result) {
  document.getElementById("sitrep-block").hidden = false;
  document.getElementById("sitrep").textContent = result.sitrep;
  document.getElementById("scene").textContent = result.scene_assessment;

  const ul = document.getElementById("counts");
  ul.innerHTML = "";
  for (const [label, n] of Object.entries(result.counts)) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${label}</span><b>${n}</b>`;
    ul.appendChild(li);
  }

  // Clear the previous analysis so each upload renders cleanly.
  detLayer.clearLayers();
  if (imageOverlay) {
    map.removeLayer(imageOverlay);
    imageOverlay = null;
  }

  // Draw the aerial tile under the detections (georeferenced images only).
  if (result.tile_url && result.bounds) {
    imageOverlay = L.imageOverlay(result.tile_url, result.bounds).addTo(map);
  }

  const pts = [];
  for (const d of result.detections) {
    if (d.lat == null || d.lon == null) continue;
    const m = L.marker([d.lat, d.lon], { icon: detIcon() }).addTo(detLayer);
    m.bindTooltip(`${d.label} ${(d.confidence * 100).toFixed(0)}%`, { direction: "top" });
    pts.push([d.lat, d.lon]);
  }
  document.getElementById("det-count").textContent = result.detections.length;

  const status = document.getElementById("analyze-status");
  if (result.bounds) {
    map.fitBounds(result.bounds, { padding: [20, 20] });
    status.textContent = `${result.detections.length} objects · ${pts.length} geolocated`;
  } else if (pts.length) {
    map.fitBounds(pts, { maxZoom: 16, padding: [40, 40] });
    status.textContent = `${result.detections.length} objects · ${pts.length} geolocated`;
  } else {
    status.textContent = `${result.detections.length} objects · image not georeferenced`;
  }
}

document.getElementById("img-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById("upload-label").textContent = file.name;
  const status = document.getElementById("analyze-status");
  status.textContent = "Running detector + assessment…";
  const fd = new FormData();
  fd.append("image", file);
  try {
    const resp = await fetch("/api/detect", { method: "POST", body: fd });
    renderDetections(await resp.json());
  } catch (err) {
    status.textContent = "Error: " + err;
  }
});

// ---------- Clock ----------
setInterval(() => {
  document.getElementById("clock").textContent =
    new Date().toISOString().replace("T", " ").slice(0, 19) + "Z";
}, 1000);