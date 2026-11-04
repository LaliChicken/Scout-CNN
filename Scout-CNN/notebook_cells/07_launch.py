#@title 7 · Launch { display-mode: "form" }
if not SMOKE_TEST:
    _, local_url, share_url = app.launch(share=True, auth=("scouts", SHARE_PASSWORD) if SHARE_PASSWORD else None,
                                         show_api=False, show_error=False, inline=False, prevent_thread_lock=True)
    assert share_url, "Share tunnel unavailable; rerun this cell."
    share_url += "?__theme=light"
    display(HTML(f'<div style="font: bold 40px Arial; padding:24px"><a href="{html.escape(share_url)}" target="_blank">{html.escape(share_url)}</a><br>open on the monitor, press F11</div>'))
else:
    print("SMOKE_TEST: launch skipped")
