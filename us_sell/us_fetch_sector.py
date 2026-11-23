"""미국 S&P500·나스닥 종합·업종 ETF 11개 종가 수집(yfinance 단독, 다른 출처 시도 결과는 보고). 임의 생성 없음."""
import os, sys, json, subprocess
import pandas as pd
OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
try: import yfinance
except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False); import yfinance
SYM = {"sp500": "^GSPC", "nasdaq": "^IXIC", "XLK": "XLK", "XLF": "XLF", "XLV": "XLV", "XLY": "XLY", "XLP": "XLP", "XLE": "XLE", "XLI": "XLI", "XLB": "XLB", "XLU": "XLU", "XLRE": "XLRE", "XLC": "XLC"}
rep = {}; cols = {}
for k, a in SYM.items():
    try:
        d = yfinance.download(a, start="2014-12-01", progress=False, auto_adjust=False)["Close"]; s = (d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d).dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize(); s = s[~s.index.duplicated()].astype(float).sort_index()
        rep[k] = dict(기호=a, 시작=str(s.index.min().date()), 끝=str(s.index.max().date()), 일수=int(len(s)), 가격종류="종가(배당 미반영, auto_adjust=False)"); cols[k] = s
    except Exception as e:
        rep[k] = dict(기호=a, 실패=type(e).__name__)
need = {"sp500", "nasdaq"}
ok = need <= set(cols) and all(cols[k].index.max() >= pd.Timestamp.now().normalize() - pd.Timedelta(days=7) for k in need)
if ok:
    df = pd.concat(cols, axis=1).sort_index().reset_index(); df.columns = ["date"] + list(cols); df.to_csv(f"{OUT}/us_closes.csv", index=False)
rep["핵심_수집_성공"] = bool(ok)
json.dump(rep, open(f"{OUT}/fetch_report.json", "w"), ensure_ascii=False, indent=1); print("fetch", "OK" if ok else "FAIL"); sys.exit(0 if ok else 3)
