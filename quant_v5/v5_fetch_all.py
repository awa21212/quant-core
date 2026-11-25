# -*- coding: utf-8 -*-
"""v5_fetch_all.py —— P1 全量下载（多线程 + 断点续跑）。

单线程实测 23 秒/只（714 只 = 4.5 小时），太慢。
改成 6 个线程并发，各自带重试；已缓存的直接跳过（断点续跑）。

用法：
    python v5_fetch_all.py                  # 全量 714 只
    python v5_fetch_all.py --workers 4
    python v5_fetch_all.py --limit 50       # 试水
    python v5_fetch_all.py --fin-only       # 只补财务
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v5_data as D  # noqa: E402

_lock = threading.Lock()
_cnt = {"px_ok": 0, "px_bad": 0, "fin_ok": 0, "fin_bad": 0, "done": 0}


def _exists(path: str, min_size: int) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > min_size


def work(code: str, fin_only: bool, px_only: bool) -> None:
    r = {"px_ok": 0, "px_bad": 0, "fin_ok": 0, "fin_bad": 0}
    if not fin_only:
        for adj in ("", "hfq"):
            p = D._px_path(code, adj)
            if _exists(p, 200):
                r["px_ok"] += 1
                continue
            df = D.fetch_px(code, adjust=adj)
            if df is not None and len(df):
                r["px_ok"] += 1
            else:
                r["px_bad"] += 1
    if not px_only:
        p = D._fin_path(code)
        if _exists(p, 50):
            r["fin_ok"] = 1
        else:
            df = D.fetch_fin(code)
            if df is not None and len(df):
                r["fin_ok"] = 1
            else:
                r["fin_bad"] = 1
    with _lock:
        for k in r:
            _cnt[k] += r[k]
        _cnt["done"] += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fin-only", action="store_true")
    ap.add_argument("--px-only", action="store_true")
    a = ap.parse_args(argv)

    uni = D.universe()
    codes = uni["code"].tolist()
    if a.limit:
        codes = codes[:a.limit]

    logf = os.path.join(D.C.OUT, "fetch_all.log")
    t0 = time.time()
    print(f"[start] {len(codes)} 只，{a.workers} 线程，{t0 and datetime.now():%H:%M:%S}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, c, a.fin_only, a.px_only): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [err] {futs[f]} {type(e).__name__}", flush=True)
            if i % 25 == 0 or i == len(codes):
                el = time.time() - t0
                eta = el / i * (len(codes) - i)
                with _lock:
                    line = (f"[{i}/{len(codes)}] px_ok={_cnt['px_ok']} px_bad={_cnt['px_bad']} "
                            f"fin_ok={_cnt['fin_ok']} fin_bad={_cnt['fin_bad']} "
                            f"用时 {el/60:.1f}min 剩余 {eta/60:.1f}min")
                print(line, flush=True)
                with open(logf, "a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now():%H:%M:%S} {line}\n")

    print(f"[done] px {_cnt['px_ok']} 成 / {_cnt['px_bad']} 败｜"
          f"fin {_cnt['fin_ok']} 成 / {_cnt['fin_bad']} 败｜总用时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if _cnt["px_bad"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
