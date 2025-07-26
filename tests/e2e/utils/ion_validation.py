import re, csv, statistics
from pathlib import Path

NUM_RX = {
    "avg":  re.compile(r"TW\s*(?:AVG|Average|평균)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    "low":  re.compile(r"(?:LOW\s*LIMIT|하한|Low Limit)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    "high": re.compile(r"(?:HIGH\s*LIMIT|상한|High Limit)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I),
}
KW_MUST_SECTIONS = [
    r"(이상 여부|판단|Anomaly|Abnormal)",
    r"(원인|Cause|해석)",
    r"(조치|Action|권고|대응)",
]
KW_DOMAIN = ["TW","Tilt","Dose","Energy","calibration","drift","분포","장비"]

def parse_number(rx, text):
    m = rx.search(text)
    return float(m.group(1)) if m else None

def derive_limits_from_csv(csv_path: Path):
    vals=[]
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k,v in row.items():
                if k and v and k.strip().lower() in ("tw","tw_avg","twavg","taper","taperwidth"):
                    try: vals.append(float(v))
                    except: pass
    if not vals:
        return None
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals)>1 else 0.0
    low  = mu-3*sd if sd>0 else mu*0.95
    high = mu+3*sd if sd>0 else mu*1.05
    return round(mu,3), round(low,3), round(high,3)

def soft_close(a,b, rel=0.2, abs_eps=0.05):
    return abs(a-b) <= max(rel*abs(b), abs_eps)

def validate_sections(text: str):
    import re
    misses=[p for p in KW_MUST_SECTIONS if not re.search(p, text, re.I)]
    return misses

def validate_domain_keywords(text: str):
    lows = [kw for kw in KW_DOMAIN if kw.lower() not in text.lower()]
    return lows

def validate_logic(text: str, avg, low, high):
    import re
    issues=[]
    if avg is None or low is None or high is None:
        return issues
    outside = (avg<low) or (avg>high)
    if outside:
        if not re.search(r"(이상|초과|abnormal|beyond limit)", text, re.I):
            issues.append("outside-limit-but-no-abnormal-mention")
    else:
        if not re.search(r"(모니터링|추가 계측|trend|트렌드|follow[-\s]*up)", text, re.I):
            issues.append("inside-limit-but-no-monitoring-mention")
    if not re.search(r"(장비 간|장비별|분포|산포|영점|calibration|drift)", text, re.I):
        issues.append("no-equipment-dispersion-mention")
    if not re.search(r"(tuning|calibration|점검|청소|deposition 제거|focus|accel|corrector|recipe 조정|조치|권고)", text, re.I):
        issues.append("no-actionable-recommendations")
    return issues