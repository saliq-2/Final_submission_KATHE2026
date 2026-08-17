# diacritics.py
# Data-derived combining-mark inventory for Kashmiri Perso-Arabic, used ONLY
# by diacritic_restore.py / density-measurement / Phase 2 diacritizer code.
# metric.py deliberately keeps using KashmiriNormalizer's incomplete
# KASHMIRI_DIACRITICS (11 marks) because it mirrors the organizers' real
# scoring script -- do not point scoring at this module.
import unicodedata

# Honorifics / pen-name markers. These are CONTENT tied to specific named
# entities, not orthographic vowel marking. They must be transparent to
# bare-form keying (so a word with and without one share a key), but must
# NEVER be predicted or generated. Permanent exclusion from any generation
# target.
HONORIFICS = {
    "ؐ",  # SALLALLAHOU ALAYHE WASSALLAM
    "ؒ",  # RAHMATULLAH ALAYHE
    "ؓ",  # RADI ALLAHOU ANHU
    "ؔ",  # TAKHALLUS
}

# Corruption / mixed-script artifacts. Strip from the corpus entirely.
JUNK = {
    "̺",  # COMBINING INVERTED BRIDGE BELOW (Latin block)
    "्",  # DEVANAGARI SIGN VIRAMA
}

def normalize_text(s):
    """Remove corruption marks. Apply before anything else."""
    return "".join(c for c in str(s) if c not in JUNK)

def is_mark(c):
    """True for any combining mark we care about, derived from Unicode
    rather than a hardcoded list."""
    return unicodedata.combining(c) != 0 and c not in JUNK

def strip_diacritics(s):
    """Bare form for dictionary keying. Removes ALL marks including
    honorifics -- that is deliberate, it is what makes keys merge."""
    return "".join(c for c in normalize_text(s) if not is_mark(c))
