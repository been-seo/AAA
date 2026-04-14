"""
OSM 타일 맵 렌더러
- 타일 캐시 (메모리 + 디스크)
- 백그라운드 다운로더 스레드
- 줌/드래그 지원
"""
import math
import time
import io
import os
import queue
import threading

import pygame
import requests

from config import (
    TILE_SIZE, TILE_CACHE_DIR, MAX_MEM_CACHE_SIZE,
    OSM_TILE_URL, USER_AGENT, TILE_REQUEST_TIMEOUT,
    SCREEN_WIDTH, SCREEN_HEIGHT, GRAY,
)


class TileCache:
    def __init__(self, cache_dir=TILE_CACHE_DIR, max_mem=MAX_MEM_CACHE_SIZE):
        self.cache_dir = cache_dir
        self.max_mem = max_mem
        os.makedirs(cache_dir, exist_ok=True)
        self._mem = {}
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()

    def get(self, x, y, z):
        with self._lock:
            item = self._mem.get((x, y, z))
            if item:
                self._mem[(x, y, z)] = (item[0], time.time())
                return item[0]
        return None

    def put(self, x, y, z, surface):
        with self._lock:
            self._mem[(x, y, z)] = (surface, time.time())
            if len(self._mem) > self.max_mem:
                self._evict()

    def _evict(self):
        items = sorted(self._mem.items(), key=lambda kv: kv[1][1])
        for i in range(len(items) - self.max_mem):
            del self._mem[items[i][0]]

    def file_path(self, x, y, z):
        return os.path.join(self.cache_dir, str(z), str(x), f"{y}.png")

    def load_from_disk(self, x, y, z):
        fp = self.file_path(x, y, z)
        if os.path.exists(fp):
            with self._file_lock:
                try:
                    with open(fp, "rb") as f:
                        return f.read()
                except IOError:
                    return None
        return None

    def save_to_disk(self, x, y, z, data):
        fp = self.file_path(x, y, z)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with self._file_lock:
            try:
                with open(fp, "wb") as f:
                    f.write(data)
            except IOError:
                pass


class TileDownloader(threading.Thread):
    def __init__(self, req_q, res_q, cache, stop_event):
        super().__init__(daemon=True)
        self.req_q = req_q
        self.res_q = res_q
        self.cache = cache
        self.stop = stop_event

    def run(self):
        while not self.stop.is_set():
            try:
                x, y, z = self.req_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.cache.get(x, y, z):
                self.req_q.task_done()
                continue
            disk = self.cache.load_from_disk(x, y, z)
            if disk:
                try:
                    self.res_q.put_nowait((x, y, z, disk))
                except queue.Full:
                    pass
                self.req_q.task_done()
                continue
            url = OSM_TILE_URL.format(x=x, y=y, z=z)
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TILE_REQUEST_TIMEOUT)
                resp.raise_for_status()
                self.cache.save_to_disk(x, y, z, resp.content)
                try:
                    self.res_q.put_nowait((x, y, z, resp.content))
                except queue.Full:
                    pass
            except Exception:
                pass
            finally:
                self.req_q.task_done()


class Map:
    def __init__(self, width, height, center_lat, center_lon, initial_zoom,
                 tile_cache, req_q, res_q):
        self.width = width
        self.height = height
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_level_float = float(initial_zoom)
        self.tile_cache = tile_cache
        self.req_q = req_q
        self.res_q = res_q
        self.drag_start_pos = None
        self.latlon_at_drag_start = None
        self._last_update_time = 0
        self._known = set()
        self.update_needed_tiles()

    def get_current_zoom_level(self):
        return self.zoom_level_float

    def get_current_zoom_int(self):
        return max(1, int(math.floor(self.zoom_level_float)))

    def _world_coords(self, lat, lon, zoom_int, n):
        wx = (lon + 180.0) / 360.0 * n * TILE_SIZE
        lat_r = math.radians(max(-85.051128, min(85.051128, lat)))
        wy = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n * TILE_SIZE
        return wx, wy

    def latlon_to_screen(self, lat, lon):
        z = self.get_current_zoom_int()
        n = 2 ** z
        sf = 2 ** (self.zoom_level_float - z)
        wx, wy = self._world_coords(lat, lon, z, n)
        cx, cy = self._world_coords(self.center_lat, self.center_lon, z, n)
        return (wx - cx + self.width / 2 / sf) * sf, (wy - cy + self.height / 2 / sf) * sf

    def screen_to_latlon(self, sx, sy):
        z = self.get_current_zoom_int()
        n = 2 ** z
        sf = 2 ** (self.zoom_level_float - z)
        cx, cy = self._world_coords(self.center_lat, self.center_lon, z, n)
        tlx = cx - self.width / 2 / sf
        tly = cy - self.height / 2 / sf
        twx = tlx + sx / sf
        twy = tly + sy / sf
        lon = twx / (n * TILE_SIZE) * 360.0 - 180.0
        yf = max(0.0, min(1.0 - 1e-9, twy / (n * TILE_SIZE)))
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yf))))
        return lat, lon

    def zoom(self, factor, mouse_pos):
        lat_b, lon_b = self.screen_to_latlon(*mouse_pos)
        self.zoom_level_float = max(3.0, min(18.99, self.zoom_level_float * factor))
        lat_a, lon_a = self.screen_to_latlon(*mouse_pos)
        self.center_lat -= lat_a - lat_b
        self.center_lon = (self.center_lon - (lon_a - lon_b) + 180) % 360 - 180
        self.update_needed_tiles()

    def update_needed_tiles(self):
        z = self.get_current_zoom_int()
        if z == 0:
            return
        n = 2 ** z
        sf = 2 ** (self.zoom_level_float - z)
        cx, cy = self._world_coords(self.center_lat, self.center_lon, z, n)
        tlx = cx - self.width / 2 / sf
        tly = cy - self.height / 2 / sf
        sx = int(math.floor(tlx / TILE_SIZE)) - 1
        sy = int(math.floor(tly / TILE_SIZE)) - 1
        nx = int(math.ceil(self.width / (TILE_SIZE * sf))) + 3
        ny = int(math.ceil(self.height / (TILE_SIZE * sf))) + 3
        for i in range(nx):
            xt = (sx + i) % n
            for j in range(ny):
                yt = sy + j
                if yt < 0 or yt >= n:
                    continue
                key = (xt, yt, z)
                if key in self._known:
                    continue
                if self.tile_cache.get(xt, yt, z) is None:
                    disk = self.tile_cache.load_from_disk(xt, yt, z)
                    if disk is None:
                        try:
                            self.req_q.put_nowait(key)
                        except queue.Full:
                            pass
                    else:
                        try:
                            self.res_q.put_nowait(key + (disk,))
                        except queue.Full:
                            pass
                self._known.add(key)

    def process_tile_results(self):
        while True:
            try:
                x, y, z, data = self.res_q.get_nowait()
                try:
                    surf = pygame.image.load(io.BytesIO(data)).convert()
                    self.tile_cache.put(x, y, z, surf)
                except Exception:
                    pass
            except queue.Empty:
                break

    def draw(self, screen):
        z = self.get_current_zoom_int()
        n = 2 ** z
        sf = 2 ** (self.zoom_level_float - z)
        cx, cy = self._world_coords(self.center_lat, self.center_lon, z, n)
        tlx = cx - self.width / 2 / sf
        tly = cy - self.height / 2 / sf
        stx = int(math.floor(tlx / TILE_SIZE))
        sty = int(math.floor(tly / TILE_SIZE))
        tsz = TILE_SIZE * sf
        if tsz <= 0:
            tsz = 1
        ntx = int(math.ceil(self.width / tsz)) + 1
        nty = int(math.ceil(self.height / tsz)) + 1
        fdx = (stx * TILE_SIZE - tlx) * sf
        fdy = (sty * TILE_SIZE - tly) * sf
        for i in range(ntx):
            xt = (stx + i) % n
            for j in range(nty):
                yt = sty + j
                if yt < 0 or yt >= n:
                    continue
                dim = (int(round(tsz)) + 1, int(round(tsz)) + 1)
                pos = (int(round(fdx + i * tsz)), int(round(fdy + j * tsz)))
                surf = self.tile_cache.get(xt, yt, z)
                if surf is None:
                    # 메모리에서 evict된 타일 → 디스크에서 복원
                    disk = self.tile_cache.load_from_disk(xt, yt, z)
                    if disk:
                        try:
                            surf = pygame.image.load(io.BytesIO(disk)).convert()
                            self.tile_cache.put(xt, yt, z, surf)
                        except Exception:
                            surf = None
                if surf:
                    try:
                        scaled = pygame.transform.scale(surf, dim)
                        screen.blit(scaled, pos)
                        overlay = pygame.Surface(dim, pygame.SRCALPHA)
                        overlay.fill((255, 255, 255, 50))
                        screen.blit(overlay, pos)
                    except Exception:
                        pass
                else:
                    try:
                        pygame.draw.rect(screen, GRAY, (pos, dim))
                    except Exception:
                        pass

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.drag_start_pos = event.pos
                self.latlon_at_drag_start = self.screen_to_latlon(*event.pos)
                return False
            elif event.button in (4, 5):
                self.zoom(1.2 if event.button == 4 else 1 / 1.2, event.pos)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.drag_start_pos and (abs(event.pos[0] - self.drag_start_pos[0]) > 5
                                        or abs(event.pos[1] - self.drag_start_pos[1]) > 5):
                self.update_needed_tiles()
                self.drag_start_pos = None
                return True
            self.drag_start_pos = None
            return False
        elif event.type == pygame.MOUSEMOTION and self.drag_start_pos:
            lat, lon = self.screen_to_latlon(*event.pos)
            self.center_lat -= lat - self.latlon_at_drag_start[0]
            self.center_lon = (self.center_lon - (lon - self.latlon_at_drag_start[1]) + 180) % 360 - 180
            if time.time() - self._last_update_time > 0.1:
                self.update_needed_tiles()
                self._last_update_time = time.time()
            return True
        return False
