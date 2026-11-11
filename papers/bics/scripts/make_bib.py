"""Write papers/bics/latex/references.bib from docs/references.md (the SSOT) for the TMLR LaTeX build.
  python3 -X utf8 papers/bics/scripts/make_bib.py [--fetch]   (from the PJ root)

Title, venue and year come from docs/references.md. Only the full author names come from the primary metadata
(arXiv API, Crossref, or the citation_author tags of the proceedings page), because the SSOT abbreviates them.
--fetch queries those services and writes papers/bics/latex/ref_authors.json (with the source of each list);
without it the cached file is used, so the build is offline and reproducible. The BibTeX key of [n] is ref<n>,
so the [n] of the manuscript maps one to one onto \\citep{ref<n>}.
"""
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LATEX = ROOT / "papers/bics/latex"
CACHE = LATEX / "ref_authors.json"
UA = {"User-Agent": "bib-builder/0.1 (research script)"}


def cited():
    ids = set()
    for f in ("manuscript_anon.md", "supplement_anon.md"):
        txt = (ROOT / "papers/bics" / f).read_text(encoding="utf-8").split("\n## References")[0]
        ids |= {int(x) for x in re.findall(r"\[(\d+)\]", txt)}
    return sorted(ids)


def ssot():
    out = {}
    for line in (ROOT / "docs/references.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \[(\d+)\] \| ([^|]+) \|([^|]+)\|([^|]+)\|([^|]+)\|", line)
        if m:
            n, key, at, venue, url = int(m[1]), m[2].strip(), m[3].strip(), m[4].strip(), m[5].strip()
            short, title = at.split(". ", 1) if ". " in at else ("", at)
            out[n] = dict(key=key, short=short, title=re.sub(r"（.*?）", "", title).strip(),
                          venue=re.sub(r"（.*?）", "", venue).strip(), url=url)
    return out


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def fetch_authors(key, url):
    if key.startswith("arXiv:"):
        x = ET.fromstring(get(f"https://export.arxiv.org/api/query?id_list={key[6:]}"))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        names = [e.text.strip() for e in x.findall("a:entry/a:author/a:name", ns)]
        return names, "arXiv API"
    if key.startswith("DOI:"):
        d = json.loads(get(f"https://api.crossref.org/works/{key[4:]}"))["message"]
        names = [f"{a.get('given', '')} {a.get('family', '')}".strip() if "family" in a else a.get("name", "")
                 for a in d.get("author", [])]
        return names, "Crossref"
    html = get(url)
    names = re.findall(r'<meta\s+name="?citation_author"?\s+content="([^"]+)"', html) or \
        re.findall(r'<meta\s+content="([^"]+)"\s+name="?citation_author"?', html)
    names = [n.replace(" Carnegie Mellon University", "") for n in names]  # PMLR R12 appends the affiliation
    names = [" ".join(reversed(n.split(", ", 1))) if ", " in n else n for n in names]
    return names, "citation_author tags of " + url


def tex_escape(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def year_of(venue):
    y = re.findall(r"(?<!\d)(?:19|20)\d\d(?!\d)", venue)
    return y[-1] if y else None


def main():
    ids, refs = cited(), ssot()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if "--fetch" in sys.argv:
        for n in ids:
            if str(n) in cache:
                continue
            names, src = fetch_authors(refs[n]["key"], refs[n]["url"])
            if not names:
                sys.exit(f"[{n}] no authors from {src}")
            cache[str(n)] = {"authors": names, "source": src, "key": refs[n]["key"]}
            print(n, len(names), names[0], "…", src)
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(1)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    out = []
    for n in ids:
        r, a = refs[n], cache[str(n)]
        venue = r["venue"]
        year = year_of(venue)
        if year is None:  # [60]: no year in the SSOT ("To appear"); use the arXiv identifier's year
            year = "20" + r["key"][6:8]
            venue = venue + ", to appear"
        venue_txt = re.sub(r",?\s+(?:v\d+\s+)?(?:19|20)\d\d\b", "", venue).strip(" ,")
        names = [("Qwen Team" if x == "Qwen" else x) for x in a["authors"] if x != ":"]  # arXiv 2412.15115 lists "Qwen", ":"
        if len(names) > 10:  # long consortium lists: first ten, then "others" (et al.)
            names = names[:10] + ["others"]
        authors = " and ".join("{" + x + "}" if x.lower().startswith(("team", "qwen")) else x for x in names)
        fields = {"author": authors, "title": "{" + r["title"] + "}", "year": year}
        if r["key"].startswith("arXiv:") and "arXiv" in venue:
            fields["journal"] = f"arXiv preprint arXiv:{r['key'][6:]}"
        else:
            fields["journal"] = venue_txt
        fields["url"] = r["url"]
        body = ",\n".join(f"  {k} = {{{tex_escape(v) if k != 'url' else v}}}" for k, v in fields.items())
        out.append(f"@article{{ref{n},\n{body}\n}}\n")
    (LATEX / "references.bib").write_text("\n".join(out), encoding="utf-8")
    print(f"{len(ids)} entries -> {LATEX / 'references.bib'}")


if __name__ == "__main__":
    main()
