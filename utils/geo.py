"""지리/항법 계산 유틸리티"""
import math


def calculate_distance(lat1, lon1, lat2, lon2):
    """두 위경도 간 거리 (해리, Haversine)"""
    R_nm = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_nm * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """두 지점 간 방위각 (도, True North 기준)"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def calculate_traffic_density(aircraft, other_aircraft_list, radius_nm=15.0, alt_range_ft=5000):
    """주변 트래픽 밀도 (0.0 ~ 1.0)"""
    count = sum(
        1 for other in other_aircraft_list
        if other is not aircraft
        and calculate_distance(aircraft.lat, aircraft.lon, other.lat, other.lon) < radius_nm
        and abs(aircraft.alt_current - other.alt_current) < alt_range_ft
    )
    return min(1.0, count / 5.0)


def alt_normal_rate_factor(progress):
    """고도 변경 곡선 계수 (0~1 → 0.2~1.0, 사인 곡선)"""
    progress = max(0.0, min(1.0, progress))
    min_factor = 0.2
    return min_factor + (1.0 - min_factor) * (1.0 - math.cos(progress * math.pi)) / 2.0


def dms_to_decimal(dms_str, latorlon):
    """DMS 문자열 → 십진 위경도. latorlon: 0=위도(DDMMSS), 1=경도(DDDMMSS)"""
    try:
        sign = 1
        if isinstance(dms_str, str) and dms_str.startswith("-"):
            sign = -1
            dms_str = dms_str[1:]
        if not isinstance(dms_str, str):
            return None
        if latorlon == 0:
            if len(dms_str) != 6:
                return None
            d, m, s = int(dms_str[:2]), int(dms_str[2:4]), int(dms_str[4:6])
        elif latorlon == 1:
            if len(dms_str) != 7:
                return None
            d, m, s = int(dms_str[:3]), int(dms_str[3:5]), int(dms_str[5:7])
        else:
            return None
        return sign * (d + m / 60.0 + s / 3600.0)
    except (ValueError, IndexError):
        return None


def decimal_to_dms(decimal_deg, symbol=False, latorlon_hint=None):
    """십진 위경도 → DMS 문자열"""
    sign_str = "-" if decimal_deg < 0 else ""
    abs_deg = abs(decimal_deg)
    degrees = int(abs_deg)
    minutes_decimal = (abs_deg - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = int(round((minutes_decimal - minutes) * 60))
    if seconds >= 60:
        minutes += seconds // 60
        seconds %= 60
    if minutes >= 60:
        degrees += minutes // 60
        minutes %= 60
    if symbol:
        return f"{sign_str}{degrees}\u00b0{minutes:02d}'{seconds:02d}\""
    is_lon = latorlon_hint == 1 or abs_deg >= 100
    fmt = f"{degrees:03d}" if is_lon else f"{degrees:02d}"
    return f"{sign_str}{fmt}{minutes:02d}{seconds:02d}"


def latlon_to_tile(lat_deg, lon_deg, zoom):
    """위경도 → OSM 타일 좌표"""
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    ytile = max(0, min(ytile, n - 1))
    return xtile, ytile, zoom


def tile_to_latlon(xtile, ytile, zoom):
    """OSM 타일 좌표 → 타일 좌상단 위경도"""
    n = 2 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return math.degrees(lat_rad), lon_deg
