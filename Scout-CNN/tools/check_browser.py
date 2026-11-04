"""Capture the actual Gradio UI at monitor resolution, using the smoke fixture."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width":1920,"height":1080}, device_scale_factor=1)
    errors, uploads = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: uploads.append(request.url) if "/upload" in request.url else None)
    page.goto("http://127.0.0.1:7860", wait_until="domcontentloaded")
    page.locator(".title").wait_for(timeout=30000)
    report = {}
    def capture(name):
        page.wait_for_timeout(300)  # Let output transitions finish before measuring/screenshotting.
        page.screenshot(path=str(OUT / (name + ".jpg")), quality=85)
        metrics = page.evaluate('''() => ({width:innerWidth,height:innerHeight,scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight,
          overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&getComputedStyle(e).overflowY==='auto'&&e.scrollHeight>e.clientHeight+2}).map(e=>({tag:e.tagName,id:e.id,height:e.clientHeight,scroll:e.scrollHeight}))})''')
        report[name] = metrics
    capture("train-ready")
    page.get_by_role("tab",name="Test",exact=True).click()
    page.locator(".empty:visible").filter(has_text="Train or load a model first").wait_for()
    capture("test-empty")
    page.get_by_role("tab",name="Inside",exact=True).click()
    page.locator(".empty:visible").filter(has_text="Train or load a model first").wait_for()
    capture("inside-empty")
    page.get_by_role("tab",name="Train",exact=True).click()
    page.get_by_role("button",name="Start",exact=True).click()
    page.locator(".status").filter(has_text="Epoch 1/1").wait_for(timeout=60000)
    capture("train")
    page.locator(".matrix td").evaluate_all("cells => cells.forEach(cell => cell.textContent = '1250')")
    capture("train-layout-stress")
    page.get_by_role("tab",name="Test",exact=True).click()
    page.locator(".demo-pick").first.click()
    page.locator(".verdict").wait_for(timeout=30000)
    capture("test")
    assert "Truth: Hidden" in page.locator(".prediction").inner_text()
    page.get_by_label("Reveal",exact=True).check()
    page.get_by_text("Truth: CAT",exact=True).wait_for(timeout=30000)
    page.get_by_role("tab",name="Inside",exact=True).click()
    page.get_by_role("heading",name="Feature space",exact=True).wait_for(timeout=30000)
    capture("inside")
    page.get_by_role("tab",name="Test",exact=True).click()
    page.get_by_label("Camera/Upload",exact=True).check()
    capture("camera")
    # Actual browser upload must travel as base64 textbox data, not /upload.
    from PIL import Image
    import io
    buffer=io.BytesIO(); Image.new("RGB",(200,100),"orange").save(buffer,format="PNG")
    page.locator("#capture-file").set_input_files({"name":"secret-cat-name.png","mimeType":"image/png","buffer":buffer.getvalue()})
    page.get_by_text("Truth: Unknown",exact=True).wait_for(timeout=30000)
    assert not uploads, uploads
    assert "secret-cat-name" not in page.content()
    assert page.locator("#capture-data textarea").input_value() == ""
    capture("upload")
    report["page_errors"] = errors
    report["upload_requests"] = uploads
    (OUT / "browser-report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert not errors, errors
    for key, value in report.items():
        if isinstance(value,dict):
            assert value["scrollHeight"] <= 1080 and value["scrollWidth"] <= 1920, (key,value)
            assert not value["overflow"], (key,value)
    browser.close()
