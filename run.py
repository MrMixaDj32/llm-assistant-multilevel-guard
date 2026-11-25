# -*- coding: utf-8 -*-
"""Запуск эксперимента: оценка уровней защиты L0–L4. Результаты -> results.json, stdout."""
import json, time, sys
import numpy as np
import data_train, data_test, data_resp
from guard import ScopeClassifier, Guard, validate_response

TEST = data_test.build()
CATS = ["IN_plain", "IN_hard", "OFF_direct", "ATK_override", "ATK_roleplay", "ATK_smuggle",
        "ATK_obfusc", "ATK_extract", "ATK_business", "ATK_foreign"]
y_block = np.array([not r["allow"] for r in TEST])

def metrics(pred):
    pred = np.asarray(pred, bool)
    tp = int((pred & y_block).sum()); fn = int((~pred & y_block).sum())
    fp = int((pred & ~y_block).sum()); tn = int((~pred & ~y_block).sum())
    tbr = tp / (tp + fn); frr = fp / (fp + tn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * tbr / (prec + tbr) if prec + tbr else 0.0
    return {"TBR": tbr, "ASR": 1 - tbr, "FRR": frr, "precision": prec, "F1": f1,
            "acc": (tp + tn) / len(pred), "tp": tp, "fn": fn, "fp": fp, "tn": tn}

def per_cat(pred):
    pred = np.asarray(pred, bool)
    out = {}
    for c in CATS:
        idx = [i for i, r in enumerate(TEST) if r["cat"] == c]
        out[c] = float(pred[idx].mean())  # доля заблокированных запросов категории
    return out

def bootstrap_ci(pred, key, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, bool); vals = []
    N = len(pred)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        p, y = pred[idx], y_block[idx]
        if key == "TBR":
            vals.append((p & y).sum() / max(1, y.sum()))
        else:
            vals.append((p & ~y).sum() / max(1, (~y).sum()))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def run_guard(g):
    t0 = time.perf_counter()
    dec = [g.decide(r["text"]) for r in TEST]
    dt = (time.perf_counter() - t0) / len(TEST) * 1000
    return np.array([d[0] for d in dec]), [d[1] for d in dec], dt

def main():
    R = {"n_test": len(TEST), "n_allow": int((~y_block).sum()), "n_block": int(y_block.sum())}
    seeds = [0, 1, 2, 3, 4]
    per_seed = {lvl: [] for lvl in ["L0", "L1", "L2", "L3"]}
    detail = {}
    for s in seeds:
        X, y, _ = data_train.build(seed=s)
        t0 = time.perf_counter(); sc = ScopeClassifier(seed=s).fit(X, y); t_train = time.perf_counter() - t0
        for lvl, g in [("L0", Guard(sc, 0)), ("L1", Guard(sc, 1)), ("L2", Guard(sc, 2)), ("L3", Guard(sc, 3))]:
            pred, reasons, dt = run_guard(g)
            m = metrics(pred); m["ms"] = dt
            per_seed[lvl].append(m)
            if s == 0:
                detail[lvl] = {"metrics": m, "per_cat": per_cat(pred),
                               "ci_TBR": bootstrap_ci(pred, "TBR"), "ci_FRR": bootstrap_ci(pred, "FRR"),
                               "errors": [(TEST[i]["cat"], TEST[i]["text"], reasons[i]) for i in range(len(TEST))
                                          if pred[i] != y_block[i]]}
        if s == 0:
            sc0, R["train_size"], R["train_sec"] = sc, len(X), t_train
    R["levels"] = detail
    R["seed_stats"] = {lvl: {k: [float(np.mean([m[k] for m in v])), float(np.std([m[k] for m in v]))]
                             for k in ["TBR", "FRR", "F1", "precision", "ms"]} for lvl, v in per_seed.items()}

    # абляция компонентов L3 (seed 0)
    abl = {}
    for name, kw in [("L3 полный", {}), ("без нормализации", {"use_norm": False}),
                     ("без сигнатур", {"use_sig": False}), ("без сегментного анализа", {"use_seg": False}),
                     ("только сигнатуры+нормализация", {"tau": 1.01, "tau_seg": 1.01})]:
        pred, _, _ = run_guard(Guard(sc0, 3, **kw))
        m = metrics(pred); abl[name] = {"metrics": m, "per_cat": per_cat(pred)}
    R["ablation"] = abl

    # чувствительность к порогу сегментного анализа
    sweep = []
    for ts in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pred, _, _ = run_guard(Guard(sc0, 3, tau_seg=ts))
        m = metrics(pred); pc = per_cat(pred)
        sweep.append({"tau_seg": ts, "TBR": m["TBR"], "FRR": m["FRR"], "smuggle": pc["ATK_smuggle"],
                      "IN_hard_FRR": pc["IN_hard"]})
    R["sweep_tau_seg"] = sweep
    # чувствительность к основному порогу L2
    sweep2 = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred, _, _ = run_guard(Guard(sc0, 2, tau=t))
        m = metrics(pred); sweep2.append({"tau": t, "TBR": m["TBR"], "FRR": m["FRR"]})
    R["sweep_tau_L2"] = sweep2

    # L4 — выходной валидатор
    RESP = data_resp.build()
    yb = np.array([not r["ok"] for r in RESP])
    out = {}
    for name, sc_ in [("правила", None), ("правила + классификатор", sc0)]:
        t0 = time.perf_counter()
        vio = [validate_response(r["text"], sc_) for r in RESP]
        dt = (time.perf_counter() - t0) / len(RESP) * 1000
        pr = np.array([bool(v) for v in vio])
        tp = int((pr & yb).sum()); fp = int((pr & ~yb).sum()); fn = int((~pr & yb).sum())
        out[name] = {"recall": tp / (tp + fn), "FPR": fp / int((~yb).sum()),
                     "precision": tp / (tp + fp) if tp + fp else 0, "ms": dt,
                     "missed": [RESP[i]["text"][:80] for i in range(len(RESP)) if yb[i] and not pr[i]],
                     "false": [RESP[i]["text"][:80] for i in range(len(RESP)) if pr[i] and not yb[i]],
                     "types": {}}
        for v in vio:
            for t in v:
                out[name]["types"][t] = out[name]["types"].get(t, 0) + 1
    R["L4"] = out
    R["n_resp"] = len(RESP)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)

    # печать
    print(f"test={R['n_test']} allow={R['n_allow']} block={R['n_block']} train={R['train_size']} train_s={R['train_sec']:.2f}")
    for lvl in ["L0", "L1", "L2", "L3"]:
        m = detail[lvl]["metrics"]; ss = R["seed_stats"][lvl]
        print(f"{lvl}: TBR={m['TBR']:.3f} {detail[lvl]['ci_TBR']} FRR={m['FRR']:.3f} {detail[lvl]['ci_FRR']} "
              f"F1={m['F1']:.3f} prec={m['precision']:.3f} ms={m['ms']:.2f} | seeds TBR={ss['TBR'][0]:.3f}±{ss['TBR'][1]:.3f} "
              f"FRR={ss['FRR'][0]:.3f}±{ss['FRR'][1]:.3f}")
    print("per category (block share):")
    print("cat".ljust(14), *[l.rjust(6) for l in ["L0", "L1", "L2", "L3"]])
    for c in CATS:
        print(c.ljust(14), *[f"{detail[l]['per_cat'][c]:.2f}".rjust(6) for l in ["L0", "L1", "L2", "L3"]])
    print("ablation:")
    for k, v in abl.items():
        m = v["metrics"]; print(f"  {k}: TBR={m['TBR']:.3f} FRR={m['FRR']:.3f} F1={m['F1']:.3f}",
                                {c: round(v['per_cat'][c], 2) for c in CATS})
    print("sweep tau_seg:", sweep)
    print("sweep tau L2:", sweep2)
    for k, v in out.items():
        print("L4", k, {kk: vv for kk, vv in v.items() if kk not in ("missed", "false")})
        print("   missed:", v["missed"]); print("   false:", v["false"])
    print("L3 errors:")
    for e in detail["L3"]["errors"]:
        print("  ", e)

if __name__ == "__main__":
    main()
