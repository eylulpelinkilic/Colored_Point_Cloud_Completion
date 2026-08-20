"""docs/results.html (A4 baski) -> docs/results_web.html (ekran + iki tema).

Tek kaynak: govde HTML'i results.html'den aliniyor, sadece olcek ve palet
katmani degisiyor. Rapor guncellenince bu script tekrar kosulur."""
import re, sys

src = open("docs/results.html").read()
style = src[src.index("<style>"): src.rindex("</style>") + 8]
body  = src[src.index("<body>") + 6: src.index("</body>")]

# baski birimlerini (pt) ekran birimlerine tasi -- 1pt ~ 1.34px okunur boyutta
style = re.sub(r"(\d+(?:\.\d+)?)pt", lambda m: f"{float(m.group(1)) * 1.34:.1f}px", style)
style = style.replace("@page{ size:A4; margin:17mm 16mm 15mm; }", "")

SCREEN = """
<style>
  /* --- ekran katmani: baski dosyasinin uzerine --- */
  :root{
    --bg:#f3f5f6;
    --shadow:0 1px 2px rgba(20,40,60,.05), 0 10px 34px rgba(20,40,60,.07);
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --ink:#e9eef1; --muted:#a2b0b8; --faint:#71808a;
      --line:#28343d; --line-soft:#1a242b; --surface:#151e24;
      --accent:#3fb6cf; --accent-ink:#93dceb; --accent-soft:#0f2d37;
      --warn-soft:#241f14; --warn-line:#6f5828;
      --bg:#0c1216;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 40px rgba(0,0,0,.45);
    }
  }
  :root[data-theme="dark"]{
    --ink:#e9eef1; --muted:#a2b0b8; --faint:#71808a;
    --line:#28343d; --line-soft:#1a242b; --surface:#151e24;
    --accent:#3fb6cf; --accent-ink:#93dceb; --accent-soft:#0f2d37;
    --warn-soft:#241f14; --warn-line:#6f5828;
    --bg:#0c1216;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 40px rgba(0,0,0,.45);
  }
  body{
    background:var(--bg); color:var(--ink);
    padding:clamp(14px, 3.5vw, 48px) clamp(12px, 4vw, 40px);
    font-size:15px; line-height:1.62;
  }
  .sheet{
    max-width:60rem; margin:0 auto; background:var(--surface-page, #fff);
    padding:clamp(20px, 4.5vw, 60px); border-radius:3px; box-shadow:var(--shadow);
  }
  :root{ --surface-page:#fff }
  @media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){ --surface-page:#101a20 } }
  :root[data-theme="dark"]{ --surface-page:#101a20 }

  h1{font-size:2.45rem}
  h2{font-size:1.16rem; margin-top:2.1rem}
  h3{font-size:1rem}
  .sub{font-size:1.03rem}
  table{font-size:.86rem}
  .grid27{font-size:.79rem}
  th{font-size:.68rem}
  .eq,.rule{font-size:.83rem; overflow-x:auto}
  .v .n{font-size:1.7rem}
  /* genis tablolar sayfayi yana kaydirmasin */
  table{display:block; overflow-x:auto; white-space:normal}
  .verdict{grid-template-columns:repeat(auto-fit, minmax(15rem, 1fr))}
  .pb{break-before:auto}
  a:focus-visible, [tabindex]:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  @media print{
    body{background:#fff; padding:0; font-size:9.7pt}
    .sheet{max-width:none; box-shadow:none; padding:0; border-radius:0}
  }
</style>"""

out = ("<title>Part conditioning evidence</title>\n" + style + SCREEN +
       '\n<div class="sheet">' + body + "</div>\n")
open("docs/results_web.html", "w").write(out)
print("yazildi: docs/results_web.html", len(out), "bayt")
