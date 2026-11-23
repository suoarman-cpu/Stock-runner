"""미국 S&P500·나스닥 종합 종가 수집(3개 출처 비교) + 코스피·코스닥 지수 갱신. 출력: OUT 폴더(csv·json). 확인 안 되는 자료는 만들지 않음."""
import os, io, sys, json, subprocess
import pandas as pd, numpy as np, requests
OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True); H = {"User-Agent": "Mozilla/5.0"}
def pipi(p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
def yf(sym):
    try: import yfinance
    except Exception: pipi("yfinance"); import yfinance
    d = yfinance.download(sym, start="2014-12-01", progress=False, auto_adjust=False); s = d["Close"]
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()
def stooq(sym):
    r = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", headers=H, timeout=40)
    return pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")["Close"].dropna()
def fred(sid):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers=H, timeout=40)
    d = pd.read_csv(io.StringIO(r.text)); d.columns = ["d", "v"]; d["d"] = pd.to_datetime(d["d"]); d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.set_index("d")["v"].dropna()
SRC = {"sp500": [("yfinance", yf, "^GSPC"), ("stooq", stooq, "^spx"), ("fred", fred, "SP500")],
       "nasdaq": [("yfinance", yf, "^IXIC"), ("stooq", stooq, "^ndq"), ("fred", fred, "NASDAQCOM")]}
rep = {"미국": {}}; chosen = {}
for nm, lst in SRC.items():
    got = {}
    for sn, f, a in lst:
        try:
            s = f(a); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize(); s = s[~s.index.duplicated()].astype(float).sort_index(); got[sn] = s
            rep["미국"][f"{nm}/{sn}"] = dict(기호=a, 시작=str(s.index.min().date()), 끝=str(s.index.max().date()), 일수=int(len(s)))
        except Exception as e:
            rep["미국"][f"{nm}/{sn}"] = dict(기호=a, 실패=type(e).__name__)
    # 가격 일치도(2025-01-01 이후 공통일)
    names = list(got)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = got[names[i]].loc["2025-01-01":], got[names[j]].loc["2025-01-01":]; c = a.index.intersection(b.index)
            if len(c): rel = (a[c] / b[c] - 1).abs(); rep["미국"][f"{nm} 일치 {names[i]}-{names[j]}"] = dict(공통일=int(len(c)), 최대상대차_pct=float(rel.max() * 100), 평균상대차_pct=float(rel.mean() * 100), 만분의1초과일=int((rel > 1e-4).sum()))
    # 사전 고정 우선순위 yfinance → stooq → FRED. 시작이 2015-01 이전이고 끝이 최근 7일 안일 때만 채택
    for sn in ("yfinance", "stooq", "fred"):
        s = got.get(sn)
        if s is not None and s.index.min() <= pd.Timestamp("2015-01-10") and s.index.max() >= pd.Timestamp.now().normalize() - pd.Timedelta(days=7):
            chosen[nm] = (sn, s); break
rep["채택"] = {k: v[0] for k, v in chosen.items()}
ok = len(chosen) == 2
if ok:
    df = pd.concat({k: v[1] for k, v in chosen.items()}, axis=1).dropna().reset_index(); df.columns = ["date", "sp500", "nasdaq"]; df.to_csv(f"{OUT}/us_closes.csv", index=False)
    rep["미국_채택_행수"] = int(len(df)); rep["미국_채택_끝"] = str(df.date.max().date())
else:
    rep["미국_결과"] = "수집 실패 — 채택 가능한 출처 없음(임의 생성 금지)"
# 한국 지수 갱신: 기존 파일 이후 구간만 pykrx 로 받아 겹치는 구간 일치 확인 후 이어붙임
kr = {}
try:
    try: from pykrx import stock
    except Exception: pipi("pykrx"); from pykrx import stock
    kd = os.environ.get("KR_SRC", "idx")
    for code in ("1001", "2001"):
        old = pd.read_pickle(f"{kd}/index_{code}.pkl"); t0 = (old.index.max() - pd.Timedelta(days=10)).strftime("%Y%m%d"); t1 = pd.Timestamp.now().strftime("%Y%m%d")
        new = stock.get_index_ohlcv_by_date(t0, t1, code, name_display=False)["종가"]; new.index = pd.DatetimeIndex(new.index)
        c = old.index.intersection(new.index); rel = (old[c] / new[c] - 1).abs() if len(c) else pd.Series(dtype=float)
        merged = pd.concat([old, new[new.index > old.index.max()]]); merged = merged[~merged.index.duplicated()].sort_index()
        merged.to_pickle(f"{OUT}/index_{code}.pkl")
        kr[code] = dict(기존끝=str(old.index.max().date()), 새끝=str(merged.index.max().date()), 겹침일=int(len(c)), 겹침최대상대차_pct=float(rel.max() * 100) if len(rel) else None, 추가일수=int(len(merged) - len(old)))
except Exception as e:
    kr["오류"] = f"{type(e).__name__}"
rep["한국지수"] = kr
json.dump(rep, open(f"{OUT}/fetch_report.json", "w"), ensure_ascii=False, indent=1)
print("fetch 완료", "미국 OK" if ok else "미국 실패", "한국지수", "OK" if "오류" not in kr else "실패")
sys.exit(0 if ok else 3)
