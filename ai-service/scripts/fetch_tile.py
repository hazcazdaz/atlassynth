"""
Download a single OSM raster tile → call AtlasSynth /depth + /classify →
save results to disk.

Requires: requests, Pillow  (pip install requests pillow)
"""

import math, requests, pathlib
from io import BytesIO
from PIL import Image

# ---------- CONFIG ----------------------------------------------------------
LAT, LON, ZOOM = 37.7749, -122.4194, 13        # San Francisco, z=13 ≈ 9 km tile
OSM_HEADERS   = {"User-Agent": "AtlasSynth/0.1 (https://github.com/yourname)"}
API_BASE      = "http://127.0.0.1:8000"        # FastAPI server must be running
# ---------------------------------------------------------------------------


def latlon_to_tile_xy(lat: float, lon: float, zoom: int):
    """Convert WGS84 lat/lon → OSM tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad))
                 / math.pi) / 2 * n)
    return xtile, ytile


def download_tile(lat, lon, zoom) -> bytes:
    x, y = latlon_to_tile_xy(lat, lon, zoom)
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    print("→ Fetching", url)
    r = requests.get(url, headers=OSM_HEADERS, timeout=10)
    r.raise_for_status()                       # crash fast on 4xx/5xx
    if r.headers.get("Content-Type") != "image/png" or len(r.content) < 1000:
        raise RuntimeError("Server did not return a valid PNG tile")
    return r.content


def call_atlassynth(endpoint: str, img_bytes: bytes):
    files = {"file": ("tile.png", img_bytes, "image/png")}
    r = requests.post(f"{API_BASE}/{endpoint}", files=files, timeout=30)
    r.raise_for_status()
    return r


def main():
    img_bytes = download_tile(LAT, LON, ZOOM)
    pathlib.Path("tile.png").write_bytes(img_bytes)

    # Optional preview
    # Image.open(BytesIO(img_bytes)).show()

    print("→ Calling /depth")
    depth = call_atlassynth("depth", img_bytes).content
    pathlib.Path("height.tif").write_bytes(depth)
    print("  Saved height.tif")

    print("→ Calling /classify")
    tags = call_atlassynth("classify", img_bytes).json()
    print("  Top tags:", tags[:3])


if __name__ == "__main__":
    main()
