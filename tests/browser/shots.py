"""Open every tab of the site in headless Chromium, record console errors, save screenshots.

usage: python tests/browser/shots.py [--url http://localhost:8123/] [--out DIR] [--tabs a,b] [--dark] [--mobile]
"""
import argparse, json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8123/")
ap.add_argument("--out", default="shots")
ap.add_argument("--tabs", default="")
ap.add_argument("--dark", action="store_true")
ap.add_argument("--mobile", action="store_true")
ap.add_argument("--full", action="store_true", help="full-page screenshots")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 420, "height": 900} if a.mobile else {"width": 1440, "height": 1000},
                        color_scheme="dark" if a.dark else "light", device_scale_factor=1)
    page = ctx.new_page()
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url}"))
    page.goto(a.url)
    page.wait_for_selector(".tab:not([hidden])", timeout=30000)
    ids = [t.get_attribute("id")[4:] for t in page.query_selector_all(".tab:not([hidden])")]
    want = [t for t in a.tabs.split(",") if t] or ids
    for tid in want:
        page.click(f"#tab-{tid}")
        t0 = time.time()
        try:
            page.wait_for_selector("#loading", state="hidden", timeout=120000)
        except Exception as e:
            errors.append(f"{tid}: loading overlay stuck: {e}")
        page.wait_for_timeout(1200)
        tag = ("dark" if a.dark else "light") + ("-m" if a.mobile else "")
        page.screenshot(path=str(out / f"{tid}-{tag}.png"), full_page=a.full)
        print(f"{tid}: {time.time()-t0:.1f}s")
    b.close()
print("ERRORS:" if errors else "no console errors")
for e in errors:
    print("  ", e[:300])
sys.exit(1 if errors else 0)
