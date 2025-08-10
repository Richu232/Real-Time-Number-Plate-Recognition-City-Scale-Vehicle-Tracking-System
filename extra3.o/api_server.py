












# api_server.py
from flask import Flask, request, jsonify, send_from_directory, render_template_string
import mysql.connector
from mysql.connector import Error
from plate_detection_final import PlateProcessor
from tp import ParquetIndex
from datetime import datetime
import os

DB_CONFIG = {
    "host": "localhost",
    "user": "edai_project_user",
    "password": "strongPassword123",
    "database": "edai_project_db"
}

app = Flask(__name__, static_folder='static')
processor = PlateProcessor()
parquet_index = ParquetIndex()

def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error:
        return None

@app.route("/search")
def search():
    q = request.args.get('q', '').strip().upper()
    fuzzy = request.args.get('fuzzy', '0') == '1'
    limit = int(request.args.get('limit', '100'))
    if not q:
        return jsonify({'error':'q param required'}), 400
    # quick prefix search in parquet index (sqlite)
    results = parquet_index.query_by_plate_prefix(q[:4], limit=limit)
    # if fuzzy requested, score using weighted_levenshtein
    if fuzzy:
        scored = []
        for r in results:
            score = processor.weighted_levenshtein(q, r.get('plate_number', ''))
            if score > 0.5:
                r['score'] = score
                scored.append(r)
        scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:limit]
        return jsonify({'count': len(scored), 'results': scored})
    else:
        return jsonify({'count': len(results), 'results': results})

@app.route("/recent_geojson")
def recent_geojson():
    minutes = int(request.args.get('minutes', '20'))
    # use parquet index to fetch recent records
    records = parquet_index.fetch_recent(minutes)
    features = []
    for r in records:
        try:
            lat = float(r.get('latitude', 0.0))
            lon = float(r.get('longitude', 0.0))
        except Exception:
            continue
        props = {
            'plate_number': r.get('plate_number'),
            'camera_id': r.get('camera_id'),
            'detected_at': r.get('detected_at'),
            'image': r.get('plate_image_path')
        }
        features.append({
            'type':'Feature',
            'geometry': {'type':'Point', 'coordinates':[lon, lat]},
            'properties': props
        })
    return jsonify({'type':'FeatureCollection','features': features})

# simple dashboard page (Leaflet)
DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ALPR Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css"/>
<style>html,body,#map { height: 100%; margin:0; padding:0; }</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([20.0, 78.0], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);
let markers = {};
async function refresh() {
  const res = await fetch('/recent_geojson?minutes=30');
  const data = await res.json();
  const features = data.features || [];
  // clear existing layer group
  for (let id in markers) { map.removeLayer(markers[id]); }
  markers = {};
  features.forEach(f => {
    const coords = f.geometry.coordinates;
    const props = f.properties;
    const key = props.plate_number + '_' + props.detected_at;
    const marker = L.marker([coords[1], coords[0]]).addTo(map);
    marker.bindPopup(`<b>${props.plate_number}</b><br>${props.detected_at}<br><img src="${props.image}" width="250">`);
    markers[key] = marker;
  });
}
setInterval(refresh, 5000);
refresh();
</script>
</body>
</html>
"""

@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
