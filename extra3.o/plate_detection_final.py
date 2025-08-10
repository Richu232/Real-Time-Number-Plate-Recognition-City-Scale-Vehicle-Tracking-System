



# plate_detection_final.py
import cv2
import numpy as np
import re
import easyocr
from datetime import datetime, timedelta
from typing import List, Tuple
from math import radians, sin, cos, sqrt, atan2

class PlateProcessor:
    def __init__(self, lang_list: List[str] = ["en"]):
        # init EasyOCR
        self.reader = easyocr.Reader(lang_list, gpu=False, verbose=False)
        # substitution costs used in weighted_levenshtein
        self.sub_costs = {('8','B'):0.3, ('0','O'):0.3, ('1','I'):0.3, ('5','S'):0.4, ('2','Z'):0.6}

    # ---------- Weighted Levenshtein ----------
    def weighted_levenshtein(self, a: str, b: str) -> float:
        a = a or ""
        b = b or ""
        la, lb = len(a), len(b)
        if la == 0 and lb == 0:
            return 1.0
        if la == 0 or lb == 0:
            return 0.0
        dp = [[0.0]*(lb+1) for _ in range(la+1)]
        for i in range(la+1):
            dp[i][0] = i
        for j in range(lb+1):
            dp[0][j] = j
        for i in range(1, la+1):
            for j in range(1, lb+1):
                cost = 0 if a[i-1] == b[j-1] else self.sub_costs.get((a[i-1], b[j-1]), self.sub_costs.get((b[j-1], a[i-1]), 1.0))
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        dist = dp[la][lb]
        sim = 1 - (dist / max(la, lb))
        return max(0.0, min(1.0, sim))

    # ---------- Gamma correction ----------
    def auto_gamma_correction(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        med = float(np.median(gray))
        if med <= 0:
            gamma = 1.0
        else:
            gamma = np.log(0.5) / (np.log(med / 255.0 + 1e-8))
            gamma = float(max(0.5, min(2.5, gamma)))
        inv_gamma = 1.0 / gamma if gamma != 0 else 1.0
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype('uint8')
        corrected = cv2.LUT(gray, table)
        return corrected, gamma

    # ---------- CLAHE ----------
    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        tx = max(2, min(8, w // 100))
        ty = max(2, min(8, h // 100))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tx, ty))
        return clahe.apply(gray)

    # ---------- Gaussian-weighted adaptive threshold ----------
    def gaussian_weighted_threshold(self, gray: np.ndarray, w: int = 11, C: float = 2.0) -> np.ndarray:
        if w % 2 == 0:
            w += 1
        k1d = cv2.getGaussianKernel(w, sigma=w/4.0)
        kernel = np.outer(k1d, k1d)
        kernel = kernel / kernel.sum()
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        thresh = (local_mean - C).astype(np.uint8)
        binary = (gray >= thresh).astype(np.uint8) * 255
        return binary

    # ---------- OCR candidate extraction ----------
    def ocr_candidates(self, img, max_candidates: int = 5):
        try:
            results = self.reader.readtext(img)
        except Exception:
            results = []
        if not results:
            return []
        results = sorted(results, key=lambda x: x[2], reverse=True)
        candidates = []
        for bbox, text, conf in results[:max_candidates]:
            cleaned = re.sub(r'\s+', '', text)
            candidates.append((cleaned, float(conf)))
        return candidates

    def consolidate_candidates(self, candidates: List[Tuple[str, float]]):
        if not candidates:
            return "", 0.0
        max_len = max(len(t[0]) for t in candidates)
        padded = [(text.ljust(max_len, '-'), conf) for (text, conf) in candidates]
        consolidated = []
        total_conf = 0.0
        for pos in range(max_len):
            votes = {}
            for text, conf in padded:
                ch = text[pos]
                if ch == '-':
                    continue
                votes[ch] = votes.get(ch, 0.0) + conf
            if not votes:
                consolidated.append('')
            else:
                consolidated.append(max(votes.items(), key=lambda x: x[1])[0])
                total_conf += sum(votes.values())
        cons = ''.join(consolidated).strip('-')
        avg_conf = (total_conf / (len(candidates) * max_len)) if max_len > 0 else 0.0
        return cons, avg_conf

    # ---------- Format/regex scoring ----------
    PLATE_REGEXES = [
        r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',
        r'^[A-Z]{2}\d{1,2}[A-Z]{0,2}\d{1,4}$',
        r'^[A-Z]{1,3}\d{1,4}$'
    ]
    def format_score(self, plate: str) -> float:
        p = (plate or "").upper()
        for rx in self.PLATE_REGEXES:
            if re.match(rx, p):
                return 1.0
        L = len(p) if len(p) > 0 else 1
        letters = sum(ch.isalpha() for ch in p)
        digits = sum(ch.isdigit() for ch in p)
        ratio_letters = letters / L
        ratio_digits = digits / L
        score = 0.5
        if ratio_letters > 0.6 and ratio_digits > 0.2:
            score += 0.2
        elif ratio_digits > 0.8:
            score -= 0.3
        if not (4 <= L <= 12):
            score -= 0.2
        return max(0.0, min(1.0, score))

    def normalize_plate_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).upper()
        corrections = {'0':'O','1':'I','5':'S','8':'B'}
        return ''.join([corrections.get(c, c) for c in cleaned])

    # ---------- Haversine ----------
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # returns meters
        R = 6371000.0
        phi1 = radians(lat1); phi2 = radians(lat2)
        dphi = radians(lat2 - lat1); dlambda = radians(lon2 - lon1)
        a = sin(dphi/2.0)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2.0)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    # ---------- Geo-duplicate check ----------
    def is_geo_duplicate(self, candidate_plate: str, lat, lon, recent_records: List[dict], time_window_seconds=300, distance_meters=200):
        """
        recent_records: list of {'plate_number':..., 'detected_at': datetime, 'latitude':..., 'longitude':...}
        returns True if any record in recent_records within time_window_seconds and distance_meters matches plate.
        Uses weighted_levenshtein + haversine.
        """
        now = datetime.utcnow()
        for rec in recent_records:
            try:
                rec_time = rec.get('detected_at') if isinstance(rec.get('detected_at'), datetime) else datetime.strptime(rec.get('detected_at'), "%Y-%m-%d %H:%M:%S")
            except Exception:
                rec_time = now
            if (now - rec_time).total_seconds() > time_window_seconds:
                continue
            db_plate = (rec.get('plate_number') or "").upper()
            sim = self.weighted_levenshtein(candidate_plate, db_plate)
            if sim > 0.75:
                d = self.haversine_distance(lat, lon, rec.get('latitude', lat), rec.get('longitude', lon))
                if d <= distance_meters:
                    return True
        return False

    # ---------- Full extraction pipeline ----------
    def extract_plate_text(self, plate_img):
        try:
            h, w = plate_img.shape[:2]
            scale = 1.0
            if max(h, w) < 120:
                scale = 2.0
            elif max(h, w) > 800:
                scale = 0.7
            if scale != 1.0:
                plate_proc = cv2.resize(plate_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            else:
                plate_proc = plate_img.copy()
            gray = cv2.cvtColor(plate_proc, cv2.COLOR_BGR2GRAY)
            gray_gamma, _ = self.auto_gamma_correction(gray)
            gray_clahe = self.apply_clahe(gray_gamma)
            binary = self.gaussian_weighted_threshold(gray_clahe, w=11, C=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            candidates = []
            candidates += self.ocr_candidates(cleaned, max_candidates=5)
            candidates += self.ocr_candidates(gray_clahe, max_candidates=5)
            candidates += self.ocr_candidates(plate_proc, max_candidates=3)
            uniq = {}
            for t, c in candidates:
                t2 = re.sub(r'\s+', '', t)
                if t2 not in uniq or uniq[t2] < c:
                    uniq[t2] = c
            candidates = [(t, uniq[t]) for t in uniq]
            consolidated_text, avg_conf = self.consolidate_candidates(candidates)
            if not consolidated_text and candidates:
                consolidated_text, avg_conf = candidates[0][0], candidates[0][1]
            normalized = self.normalize_plate_text(consolidated_text)
            if not (4 <= len(normalized) <= 12):
                if candidates:
                    fb, fbc = max(candidates, key=lambda x: x[1])
                    normalized_fb = self.normalize_plate_text(fb)
                    if 4 <= len(normalized_fb) <= 12:
                        normalized = normalized_fb
                        avg_conf = fbc
            fscore = self.format_score(normalized)
            combined_score = 0.6 * avg_conf + 0.4 * fscore
            if combined_score < 0.55:
                return None
            return re.sub(r'[^A-Z0-9]', '', normalized.upper())
        except Exception as e:
            print("extract_plate_text error:", e)
            return None
