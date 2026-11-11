"""WordNet 3.0 sense inventories for method A and the fake inventory of control (iv) (preregistration v4, Sec. 5).

- A: the glosses (definitions only, no usage examples) of all senses of the lemma and part of speech,
  permuted with a per-item seed.
- (iv): each gloss is replaced by a gloss of a different lemma with the same part of speech. The number of
  senses is kept, and the replacement length (in characters) is within +/-20%. Lemmas that share a synset
  with the target lemma, and lemmas of its direct hypernyms and hyponyms, are excluded as donors. If no
  donor satisfies the length condition, the tolerance is relaxed to +/-40%, and then to the nearest length.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from common import item_seed

POS = {"N": "n", "V": "v"}
LEN_TOL = (0.20, 0.40)  # copied to config.json at freeze 2


def get_wn(nltk_dir=None):
    """WordNet 3.0. With nltk_dir, the reader is built on <nltk_dir>/corpora/wordnet.zip itself: the default
    loader prefers an extracted corpora/wordnet/ folder anywhere on nltk.data.path over the zip, so the file
    whose digest was checked would not be the one read."""
    import nltk
    if nltk_dir:
        from nltk.corpus.reader.wordnet import WordNetCorpusReader
        nltk.data.path[:] = [str(Path(nltk_dir).resolve())]  # nothing else is searched
        zp = Path(nltk_dir).resolve() / "corpora" / "wordnet.zip"
        wn = WordNetCorpusReader(nltk.data.ZipFilePathPointer(str(zp), "wordnet/"), None)
        assert str(wn.root).startswith(str(zp)), wn.root
    else:
        from nltk.corpus import wordnet as wn
    assert wn.get_version() == "3.0", wn.get_version()
    return wn


def senses(wn, lemma: str, pos: str):
    """Senses of exactly this lemma and part of speech (Sec. 5). wn.synsets() is not used because it applies
    morphy and would add the senses of other lemmas (e.g. "saw" would get the senses of "see")."""
    return list(dict.fromkeys(l.synset() for l in wn.lemmas(lemma, POS[pos])))  # one entry per synset


def inventory_A(wn, data: str, item_id: str, lemma: str, pos: str):
    """Permuted glosses and synset names. Returns None for fewer than two senses (excluded, Sec. 5)."""
    ss = senses(wn, lemma, pos)
    if len(ss) < 2:
        return None
    order = np.random.default_rng(item_seed(data, item_id, "A")).permutation(len(ss))
    return {"definitions": [ss[i].definition() for i in order], "synsets": [ss[i].name() for i in order]}


class DonorPool:
    """All synsets of each part of speech, sorted by gloss length."""

    def __init__(self, wn):
        self.wn = wn
        self.by_pos = {}
        self.lemmas = {}
        self.defn = {}
        for p, wp in POS.items():
            rows = []
            for s in wn.all_synsets(wp):
                n = s.name()
                self.lemmas[n] = frozenset(l.lower() for l in s.lemma_names())
                self.defn[n] = s.definition()
                rows.append((len(self.defn[n]), n))
            rows.sort()
            self.by_pos[p] = ([r[0] for r in rows], [r[1] for r in rows])

    def banned_lemmas(self, lemma, pos):
        wn = self.wn
        ban = {lemma.lower()}
        for s in senses(wn, lemma, pos):
            ban |= {l.lower() for l in s.lemma_names()}
            for r in s.hypernyms() + s.instance_hypernyms() + s.hyponyms() + s.instance_hyponyms():
                ban |= {l.lower() for l in r.lemma_names()}
        return ban

    def candidates(self, pos, length, tol):
        lens, names = self.by_pos[pos]
        lo, hi = length * (1 - tol), length * (1 + tol)
        a, b = bisect.bisect_left(lens, lo), bisect.bisect_right(lens, hi)
        return names[a:b]


def inventory_iv(wn, pool: DonorPool, data: str, item_id: str, lemma: str, pos: str, inv_A: dict):
    """Replace each gloss of the A inventory, keeping its order. Donor synsets are not reused within an item."""
    rng = np.random.default_rng(item_seed(data, item_id, "iv"))
    ban = pool.banned_lemmas(lemma, pos)
    used = set()
    out_defs, out_syn, tol_used = [], [], []

    def ok(name):
        return name not in used and not (pool.lemmas[name] & ban)

    for d in inv_A["definitions"]:
        L = len(d)
        pick, tol_hit = None, None
        for tol in LEN_TOL:
            cands = [n for n in pool.candidates(pos, L, tol) if ok(n)]
            if cands:
                pick, tol_hit = cands[int(rng.integers(len(cands)))], tol
                break
        if pick is None:  # fall back to the nearest gloss length
            lens, names = pool.by_pos[pos]
            order = sorted(range(len(lens)), key=lambda i: (abs(lens[i] - L), names[i]))
            pick = next(names[i] for i in order if ok(names[i]))
            tol_hit = "nearest"
        used.add(pick)
        out_defs.append(pool.defn[pick])
        out_syn.append(pick)
        tol_used.append(tol_hit)
    return {"definitions": out_defs, "synsets": out_syn, "tolerance": tol_used}
