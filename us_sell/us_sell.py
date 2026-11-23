"""미국·한국 증시 관계 기반 매도 학습 (사전등록 v2: analysis_ab/candidates_us_sell_DRAFT_v2.json → 확정본 candidates_us_sell.json).
기존 T3 상위10 매수·기존 매도 규칙(tools/t3s2.py outcomes 원문 실행)은 읽기만. 결과는 별도 폴더(OUT)에만 저장.
환경변수: OUT(결과 폴더), US_CSV(미국 종가 csv: date,sp500,nasdaq), SMOKE=1(가짜 미국 자료·결과는 시험용), SYNTH_EFFECT=x(가짜 효과 주입 시험)"""
import os, sys, json, time, hashlib
sys.dont_write_bytecode = True
ROOT = os.getcwd(); assert os.path.exists("tools/t3s2.py"), "Stock 작업 폴더에서 실행"
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
t00 = time.time()
OUT = os.environ.get("OUT", "/tmp/us_sell_out"); os.makedirs(OUT, exist_ok=True)
SMOKE = bool(os.environ.get("SMOKE")); SYNTH = float(os.environ.get("SYNTH_EFFECT", "0"))
NXT0 = pd.Timestamp("2025-03-04"); MINTRAIN = int(os.environ.get("MINTRAIN", "2000"))
def log(*a):
    print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)
# ── 기존 코드 원문 실행(자료·outcomes 정의까지, 1~104행). 읽기만 ──
src = open("tools/t3s2.py", encoding="utf-8").read().split("\n")
g = {"__name__": "t3s2_slice", "__file__": os.path.abspath("tools/t3s2.py")}
exec(compile("\n".join(src[:104]), "tools/t3s2.py[1:104]", "exec"), g)
globals().update({k: g[k] for k in ("T", "N", "C", "O", "H", "cal", "keep", "tr", "halt", "taxes", "imp", "imp_b", "E_", "trE", "limup", "nxs", "at", "sh", "ti", "jj",
                                    "cand", "lastC", "last", "outcomes", "sellable", "te0", "t_hi", "rows", "HOLD", "FEE_MAIN", "K", "tax_rate")})
from dante_v1 import im_test, bh
assert not SMOKE or True
z = np.load("data/cnn/model_ta3.npz", allow_pickle=True); assert list(z["keep"]) == list(keep)
P_T3 = z["T"].astype(np.float64); assert not np.isfinite(P_T3[:int(np.searchsorted(cal.values, np.datetime64("2020-01-02")))]).any()
rows = np.zeros(T, bool); rows[te0:T - (HOLD + 2) + 1] = True
popT = np.isfinite(P_T3) & tr & rows[:, None]
S = np.where(popT, P_T3, -np.inf); part = np.argsort(-S, axis=1)[:, :10]
selT = np.zeros((T, N), bool); selT[np.arange(T)[:, None], part] = True; selT &= popT; okT = popT.sum(1) >= 10
selT &= okT[:, None]
oc = outcomes(FEE_MAIN, "con")
log(f"[기존 T3] 상위10 선택 {int(selT.sum()):,}건, 신호일 {int(okT.sum())}일")
# ── 기존 규칙 최종 청산(outcomes 와 같은 식, 목표 +10% 하한 기준)에서 청산일 복원 ──
fee = FEE_MAIN; bc = 1 - fee - imp_b; sc_t = 1 - fee - taxes[:, None] - imp
tgt = E_ * 1.10 / (bc * sc_t); hk = np.full((T, N), -1, np.int16)
for k in range(HOLD):
    d_tr = sh(tr, 1 + k, False); d_H = sh(H, 1 + k, np.nan); m = (hk < 0) & d_tr & (d_H >= tgt); hk[m] = k
hd = ti + 1 + hk.astype(int); reach = hk >= 0
s = at(nxs, hd + 1, T).astype(int); s = np.where(reach, np.minimum(s, T), -1)
pxs = at(O, s); net_s = pxs / E_ * bc * (1 - fee - taxes[np.clip(s, 0, T - 1)] - at(imp, s)) - 1
ok_s = reach & (s >= 0) & (s < T) & np.isfinite(pxs)
x = ti + HOLD; xs = at(nxs, x, T).astype(int)
dead = (xs >= T) & cand[None, :]; unres = (xs >= T) & ~cand[None, :]
px_x = np.where(xs == x, at(C, x), at(O, xs)); px_x = np.where(dead, lastC[None, :] * 0.7, px_x)
net_x = np.where(dead, px_x / E_ * bc - 1, px_x / E_ * bc * (1 - fee - taxes[np.clip(xs, 0, T - 1)] - at(imp, xs)) - 1)
filled = oc["filled"]
net_ex = np.where(reach & ok_s, net_s, np.where(reach, np.nan, net_x)); net_ex = np.where(filled, net_ex, 0.0)
assert np.allclose(np.nan_to_num(net_ex[selT], nan=9), np.nan_to_num(oc["net"][selT], nan=9)), "청산 복원이 기존 outcomes 와 다름"
# 청산일 X: 도달=매도일 s(시가), 미도달=xs(종가 x 또는 다음 시가), 폐지=마지막 거래일(폐지 청산일 정의: 마지막 거래일), 미청산=제외
Xidx = np.where(reach & ok_s, s, np.where(reach, -1, np.where(dead, last[None, :], np.where(unres, -1, np.minimum(xs, T - 1)))))
# ── 거래 목록 ──
ts, js = np.where(selT); trade = pd.DataFrame(dict(ti=ts, j=js))
trade["filled"] = filled[ts, js]; trade["net_ex"] = net_ex[ts, js]; trade["X"] = Xidx[ts, js]
trade["dead"] = dead[ts, js] & ~reach[ts, js]; trade["unres"] = unres[ts, js] & ~reach[ts, js]
trade["E"] = E_[ts, js]; trade["bc"] = bc[ts, js]; trade["tb"] = ts + 1
trade["sigdate"] = cal.values[ts]
log(f"[거래] 전체 {len(trade):,} · 미체결 {int((~trade.filled).sum())} · 미청산 {int(trade.unres.sum())} · 폐지 {int(trade.dead.sum())} · 비용미확인(기존) {int((trade.filled & trade.net_ex.isna()).sum())}")
# ── 미국·한국 지수 (결정일 d 시가 이전에 확정되는 값만) ──
us = pd.read_csv(os.environ["US_CSV"], parse_dates=["date"]).sort_values("date") if os.environ.get("US_CSV") else None
if SMOKE and us is None:
    rng = np.random.default_rng(7); dts = pd.bdate_range("2015-01-01", cal[-1]); us = pd.DataFrame(dict(date=dts, sp500=100 * np.exp(np.cumsum(rng.normal(0, .01, len(dts)))), nasdaq=100 * np.exp(np.cumsum(rng.normal(0, .012, len(dts))))))
    log("[시험] 가짜 미국 자료 사용 — 결과는 판정에 쓰지 않음")
assert us is not None, "미국 자료 없음 — 임의 생성·대체 금지"
us = us.dropna().reset_index(drop=True); udt = us["date"].values.astype("datetime64[D]")
def us_feat(usd, usc, calv):
    """한국 거래일 d: L(d)=미국 세션 날짜<d 중 최신. 수익률=C[L(d)]/C[L(d-1)]-1 (직전 한국 거래일 이후 새로 마감한 세션 누적). 새 세션 없으면 NaN+플래그"""
    L = np.searchsorted(usd, calv.astype("datetime64[D]"), side="left") - 1
    Lp = np.r_[-1, L[:-1]]; new = (L > Lp) & (Lp >= 0)
    r = np.full(len(calv), np.nan); r[new] = usc[L[new]] / usc[Lp[new]] - 1
    return r, new.astype(float), L
spr, newsess, L1 = us_feat(udt, us["sp500"].values, cal.values)
nqr, _, _ = us_feat(udt, us["nasdaq"].values, cal.values)
assert (udt[L1[L1 >= 0]] < cal.values[L1 >= 0].astype("datetime64[D]")).all(), "미국 날짜 누출"
idx = {}
for nm, f in (("kospi", "1001"), ("kosdaq", "2001")):
    sx = pd.read_pickle(f"data/bt_cache/index_{f}.pkl") if not os.environ.get("KR_DIR") else pd.read_pickle(f"{os.environ['KR_DIR']}/index_{f}.pkl")
    v = sx.reindex(cal).values.astype(float); r = np.full(T, np.nan); r[2:] = v[1:-1] / v[:-2] - 1; idx[nm] = r    # d 의 값 = (d-1 종가)/(d-2 종가)-1
sgn = lambda a: np.where(np.isnan(a), np.nan, np.sign(a))
F = dict(sp=spr, nq=nqr, sp_s=sgn(spr), nq_s=sgn(nqr), newsess=newsess, kospi=idx["kospi"], kosdaq=idx["kosdaq"], kospi_s=sgn(idx["kospi"]), kosdaq_s=sgn(idx["kosdaq"]))
F["rel_sp_kospi"] = 3 * (F["sp_s"] + 1) + (F["kospi_s"] + 1); F["rel_nq_kosdaq"] = 3 * (F["nq_s"] + 1) + (F["kosdaq_s"] + 1)
F["d_sp_kospi"] = spr - F["kospi"]; F["d_nq_kosdaq"] = nqr - F["kosdaq"]
# ── 결정 행 ──
R = []
for r_ in trade[trade.filled & trade.net_ex.notna() & (trade.X >= 0) & ~trade.unres].itertuples():
    d = np.arange(r_.tb + 1, r_.X)                                          # 기존 청산일 X 이전(X>d)
    d = d[(d >= 1) & (cal.values[d] >= np.datetime64(NXT0))]
    if len(d) == 0: continue
    d = d[sellable[d, r_.j] & tr[d, r_.j]]
    for dd in d: R.append((r_.Index, r_.j, r_.ti, dd))
dec = pd.DataFrame(R, columns=["tr", "j", "ti", "d"]); dec["X"] = trade.X.values[dec.tr]; dec["E"] = trade.E.values[dec.tr]; dec["bc"] = trade.bc.values[dec.tr]
dec["net_ex"] = trade.net_ex.values[dec.tr]
dj = dec.j.values; dd_ = dec.d.values
dec["net_open"] = O[dd_, dj] / dec.E * dec.bc * (1 - fee - taxes[dd_] - imp[dd_, dj]) - 1          # 실제 d 의 비용
dec["y"] = dec.net_open - dec.net_ex
dec["held"] = dd_ - trade.tb.values[dec.tr]; dec["prevret"] = C[dd_ - 1, dj] / dec.E - 1
for k_, v_ in F.items(): dec[k_] = v_[dd_]
if SYNTH:
    dec["net_open"] = dec.net_open + SYNTH * (dec.sp < 0); dec["y"] = dec.net_open - dec.net_ex; log(f"[시험] 가짜 효과 주입 {SYNTH}")
dec["unk"] = ~np.isfinite(dec.net_open)
dec["qtr"] = pd.PeriodIndex(cal.values[dd_], freq="Q").astype(str)
log(f"[결정 행] {len(dec):,} (비용 미확인 {int(dec.unk.sum())}) · 거래 {dec.tr.nunique():,}")
dec.groupby("qtr").agg(행=("y", "size"), 거래=("tr", "nunique")).to_csv(f"{OUT}/decision_rows_by_quarter.csv")
# ── 변수 집합 (사전등록) ──
STATE = ["held", "prevret"]
M = {"M1": ["sp", "nq", "sp_s", "nq_s", "newsess"] + STATE,
     "M2": ["kospi", "kosdaq", "kospi_s", "kosdaq_s"] + STATE,
     "M3": ["sp", "nq", "sp_s", "nq_s", "newsess", "kospi", "kosdaq", "kospi_s", "kosdaq_s", "rel_sp_kospi", "rel_nq_kosdaq", "d_sp_kospi", "d_nq_kosdaq"] + STATE}
CATS = {"rel_sp_kospi", "rel_nq_kosdaq"}
import lightgbm as lgb
from model_a_lib import PARAM
PR = dict(PARAM, objective="regression", num_threads=os.cpu_count()); NROUND = 104            # T3 1단 기록 그루 수 104, 튜닝 없음
quarters = sorted(set(dec.qtr)); xd = np.where(dec.X >= 0, dec.X, 10 ** 9)
def qstart(q):
    p = pd.Period(q, freq="Q").start_time; return int(np.searchsorted(cal.values, np.datetime64(p)))
pred = {m: np.full(len(dec), np.nan) for m in M}; fold_info = []
lab = dec.y.notna().values
for q in quarters:
    V = qstart(q); te = (dec.qtr == q).values
    trm = lab & (dec.d.values < V) & (xd < V)                                 # 청산일 기준 purge: 학습 행의 최종 청산 < 검증 분기 시작
    assert (dec.X.values[trm] < V).all() and (dec.d.values[trm] < V).all()
    info = dict(분기=q, 학습행=int(trm.sum()), 검증행=int(te.sum()))
    if trm.sum() >= MINTRAIN:
        for m, cols in M.items():
            ds = lgb.Dataset(dec.loc[trm, cols].values.astype(float), dec.y.values[trm], feature_name=cols, categorical_feature=[c for c in cols if c in CATS], free_raw_data=False)
            b = lgb.train(PR, ds, NROUND); pred[m][te] = b.predict(dec.loc[te, cols].values.astype(float))
        info["학습"] = "실행"
    else: info["학습"] = f"표본 부족(학습 {int(trm.sum())} < {MINTRAIN}) — 평가 제외"
    fold_info.append(info); log(info)
# ── 정책 평가: 거래별 d 시간순, 예측>0 첫 날 시가 매도, 없으면 기존 ──
def evaluate(m):
    p = pred[m]; df = dec[["tr", "d", "net_open", "qtr"]].copy(); df["p"] = p
    ev_tr = df.groupby("tr").p.apply(lambda s: s.notna().all())               # 모든 결정일이 평가 가능한 거래만
    df = df[df.tr.map(ev_tr)]; df = df[np.isfinite(df.net_open)]              # 비용 미확인 일은 매도 불가일(0% 대체 없음)
    first = df[df.p > 0].sort_values(["tr", "d"]).groupby("tr").first()
    tt = trade.loc[ev_tr[ev_tr].index]; tt = tt[tt.net_ex.notna()].copy()
    tt["early"] = tt.index.isin(first.index); tt["net_pol"] = tt.net_ex
    tt.loc[first.index.intersection(tt.index), "net_pol"] = first.net_open.reindex(first.index.intersection(tt.index))
    tt["diff"] = tt.net_pol - tt.net_ex; tt["blk"] = pd.PeriodIndex(tt.sigdate, freq="Q").astype(str)
    return tt
res = {}
for m in M:
    tt = evaluate(m)
    if len(tt) == 0: res[m] = dict(판정="평가 가능한 거래 없음"); continue
    day = tt.groupby("ti")["diff"].mean(); blk = tt.assign(dm=tt.ti.map(day)).drop_duplicates("ti").groupby("blk").dm.mean()
    E = list(blk.values); p = im_test(E)[0] if len(E) >= 3 else None
    def sm(col): v = tt[col].values; w = v[v < 0]; return dict(평균_pct=float(v.mean() * 100), 중위_pct=float(np.median(v) * 100), 달성률_pct=float(np.mean(v >= 0.10 - 1e-12) * 100),
                                                                손실비율_pct=float(np.mean(v < 0) * 100), 손실평균_pct=float(w.mean() * 100) if len(w) else None, 하위5_pct=float(np.percentile(v, 5) * 100))
    res[m] = dict(거래수=int(len(tt)), 신호일수=int(tt.ti.nunique()), 분기묶음수=int(len(blk)), 분기별_평균차이_pct={k: round(float(v) * 100, 4) for k, v in blk.items()}, 분기별_거래수=tt.groupby("blk").size().to_dict(),
                  기존=sm("net_ex"), 개선=sm("net_pol"), 평균차이_pct=float(tt["diff"].mean() * 100), 조기매도건수=int(tt.early.sum()), 조기매도비율_pct=float(tt.early.mean() * 100), IM_p=p,
                  판정=("판정 불가(분기 묶음 3개 미만)" if p is None else "검정 수행"))
    tt.drop(columns=["j"]).to_csv(f"{OUT}/trades_{m}.csv", index=False)           # 종목코드 없음(인덱스만)
ps = [(m, r["IM_p"]) for m, r in res.items() if r.get("IM_p") is not None]
if ps:
    qs = bh([p for _, p in ps]); 
    for (m, _), qv in zip(ps, qs): res[m]["BH_q"] = float(qv); res[m]["유의(q<0.05)"] = bool(qv < 0.05 and res[m]["평균차이_pct"] > 0)
# ── 누출 점검: 미국 자료를 d 이전으로 절단해 다시 계산한 값과 비교(무작위 60일) ──
rng = np.random.default_rng(0); chk = rng.choice(np.where(np.isfinite(spr))[0], min(60, int(np.isfinite(spr).sum())), replace=False); bad = 0
for d in chk:
    cut = udt < cal.values[d].astype("datetime64[D]"); r1, _, _ = us_feat(udt[cut], us["sp500"].values[cut], cal.values[: d + 1]); bad += int(not np.isclose(r1[d], spr[d], equal_nan=True))
miss = dict(결정행=int(len(dec)), 비용미확인행=int(dec.unk.sum()), 라벨결측행=int((~lab).sum()), 미국수익률결측행=int(dec.sp.isna().sum()), 새세션없음행=int((dec.newsess == 0).sum()), 한국지수결측행=int(dec.kospi.isna().sum()))
R_ = dict(모드=("시험(가짜 미국 자료)" if SMOKE else "실제"), 가짜효과=SYNTH, 최소학습행=MINTRAIN, 모델=dict(PR, 그루=NROUND, 변수집합=M), 폴드=fold_info, 결과=res, 누출절단시험=dict(검사일=len(chk), 불일치=bad),
          결측=miss, 거래전체=int(len(trade)), 기간=dict(미국=[str(us.date.min().date()), str(us.date.max().date())], 한국가격끝=str(cal[-1].date()), 결정일범위=[str(cal[dec.d.min()].date()), str(cal[dec.d.max()].date())]),
          lightgbm=lgb.__version__, 입력_sha256=dict(model_ta3=hashlib.sha256(open("data/cnn/model_ta3.npz", "rb").read()).hexdigest()[:16]))
json.dump(R_, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("[완료]", json.dumps({m: {k: r.get(k) for k in ("거래수", "분기묶음수", "평균차이_pct", "IM_p", "BH_q", "판정")} for m, r in res.items()}, ensure_ascii=False, default=str))
