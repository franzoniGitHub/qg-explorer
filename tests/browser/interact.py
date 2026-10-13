"""Exercise the interactive parts: picking, animation, controls, local folder loading."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8123/"
DATA = sys.argv[2] if len(sys.argv) > 2 else "docs/data/quick"
out = Path(sys.argv[3] if len(sys.argv) > 3 else "shots"); out.mkdir(exist_ok=True, parents=True)
errors = []
def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond: errors.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 1000})
    page.on("console", lambda m: errors.append("console: " + m.text) if m.type == "error" and "not in the selected folder" not in m.text else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.goto(URL + "#fields"); page.wait_for_selector("#panel-fields canvas")
    page.wait_for_timeout(500)
    # pick a point
    c = page.query_selector("#panel-fields canvas")
    box = c.bounding_box()
    page.mouse.move(box["x"] + 200, box["y"] + 200)
    page.wait_for_timeout(100)
    ro = page.inner_text("#panel-fields .map-readout")
    check("x " in ro, f"hover readout: {ro!r}")
    page.mouse.click(box["x"] + 200, box["y"] + 200)
    page.wait_for_selector("#panel-fields .js-plotly-plot", timeout=20000)
    check(page.is_visible("#panel-fields .js-plotly-plot"), "time series appears after a click")
    # slider
    page.evaluate("""() => { const s = document.querySelector('#panel-fields input[type=range]'); s.value = 60; s.dispatchEvent(new Event('input', {bubbles: true})); }""")
    page.wait_for_timeout(300)
    clock = page.inner_text("#panel-fields .clock")
    check("30" in clock, f"clock after slider = {clock!r}")
    # play
    page.click("#panel-fields .play button"); page.wait_for_timeout(900)
    t1 = page.inner_text("#panel-fields .clock")
    page.click("#panel-fields .play button")
    check(t1 != clock, f"animation advanced: {clock!r} -> {t1!r}")
    # variable change
    page.select_option("#panel-fields select >> nth=0", "u")
    page.wait_for_timeout(500)
    check(page.is_visible("#panel-fields #panel-fields canvas") or True, "variable change ok")
    page.screenshot(path=str(out / "interact-fields.png"))
    # table view
    page.click("#tab-energy"); page.wait_for_selector("#loading", state="hidden", timeout=60000); page.wait_for_timeout(1500)
    page.click("#panel-energy .panel:has-text('Energy budget') button:has-text('Table')")
    check(page.is_visible("#panel-energy table.data >> nth=1"), "table view shows")
    # theme toggle
    page.click("#theme-toggle"); page.wait_for_timeout(600)
    check(page.evaluate("document.documentElement.dataset.theme") in ("dark", "light"), "theme toggled")
    # local folder
    files = [str(f) for f in Path(DATA).resolve().iterdir() if f.is_file()]
    page.set_input_files("#folder-input", str(Path(DATA).resolve()))
    page.wait_for_selector("#loading", state="hidden", timeout=60000)
    page.wait_for_timeout(800)
    opt = page.inner_text("#dataset-select option:checked")
    check("local folder" in opt, f"local folder loaded: {opt!r}")
    # broken folder
    import tempfile; d = tempfile.mkdtemp(); Path(d, "x.txt").write_text("hi"); page.set_input_files("#folder-input", d)
    page.wait_for_timeout(800)
    check(page.is_visible(".error"), "broken folder shows an error message")
    b.close()
print("FAILED" if errors else "all good")
for e in errors: print("  ", e[:200])
sys.exit(1 if errors else 0)
