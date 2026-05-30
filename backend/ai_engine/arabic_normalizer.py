"""
Arabic/Egyptian Dialect Normalization Layer
============================================

This module preprocesses Arabic text before it enters the AI pipeline.
It standardizes Egyptian Arabic dialect variations into consistent forms
so the intent detection, RAG retrieval, and context systems can work
with clean, normalized input.

Why Normalize?
  Egyptian Arabic (العامية المصرية) has many spelling variations for the
  same word. For example:
    - "إيه" / "ايه" / "اي" → all mean "what"
    - "ازاى" / "ازاي" / "إزاي" → all mean "how"
    - "عايز" / "عاوز" → both mean "want"

  Without normalization, the ML pipeline treats these as different words,
  reducing accuracy. By standardizing before processing, all variations
  map to the same representation.

Pipeline Position:
  User Input → [ArabicNormalizer] → Tool Dispatch → Intent Detection → RAG → Response

Components:
  1. Egyptian Dialect Standardization — common dialect word forms
  2. Arabic Character Normalization — alef/hamza/taa marbuta variants
  3. Diacritics Removal — strip tashkeel (harakat)
  4. Tatweel Removal — remove kashida stretching
  5. Whitespace Normalization — collapse multiple spaces

Safety:
  Non-Arabic text (English, etc.) passes through unchanged.
  The normalizer only modifies Arabic Unicode characters and known dialect patterns.
"""

import re


class ArabicNormalizer:
    """
    Arabic/Egyptian dialect text normalizer.

    Standardizes Egyptian Arabic dialect variations and Arabic character
    forms into a consistent representation for downstream NLP processing.

    Usage:
        normalizer = ArabicNormalizer()
        clean_text = normalizer.normalize("إيه الأخبار يا صاحبي؟")
        # → "ايه الاخبار يا صاحبي؟"
    """

    def __init__(self):
        """
        Initialize the normalizer with dialect mappings and character tables.
        """
        # Egyptian dialect word/phrase mappings
        # Format: (original_variation, standardized_form)
        # Ordered from longer phrases to shorter ones to avoid partial replacements
        self._build_dialect_map()

        # Arabic diacritics (tashkeel) Unicode range
        # These are the harakat: fatha, damma, kasra, shadda, sukun, etc.
        self.diacritics_pattern = re.compile(
            r'[ؗ-ًؚ-ْٰ]'
        )

        # Tatweel (kashida) character — used for stretching Arabic text visually
        self.tatweel_pattern = re.compile(r'ـ')

        # Multiple whitespace collapse
        self.whitespace_pattern = re.compile(r'\s+')

    def normalize(self, text):
        """
        Apply full normalization pipeline to the input text.

        Pipeline order:
          1. Diacritics removal (tashkeel)
          2. Tatweel removal (kashida)
          3. Egyptian dialect standardization (word forms) — BEFORE character normalization
          4. Arabic character normalization (alef, hamza, taa marbuta)
          5. Whitespace normalization

        Note: Dialect normalization runs BEFORE character normalization because
        dialect patterns are written with specific character forms (e.g., "إيه" has
        إ specifically). If we normalized إ→ا first, the dialect patterns wouldn't match.

        Args:
            text (str): Raw input text (can be Arabic, English, or mixed).

        Returns:
            str: Normalized text ready for AI processing.
                 Non-Arabic text passes through unchanged.
        """
        if not text or not text.strip():
            return text

        # Step 1: Remove diacritics (tashkeel/harakat)
        text = self._remove_diacritics(text)

        # Step 2: Remove tatweel (kashida stretching)
        text = self._remove_tatweel(text)

        # Step 3: Apply Egyptian dialect standardization FIRST
        # (dialect patterns contain specific character forms like إ، ة، ى)
        text = self._normalize_dialect(text)

        # Step 4: Normalize Arabic characters (alef variants, hamza, etc.)
        text = self._normalize_characters(text)

        # Step 5: Normalize whitespace
        text = self._normalize_whitespace(text)

        return text

    # =========================================================================
    # STEP 1: DIACRITICS REMOVAL
    # =========================================================================

    def _remove_diacritics(self, text):
        """
        Remove Arabic diacritical marks (tashkeel/harakat).

        Removes: fatha (َ), damma (ُ), kasra (ِ), shadda (ّ), sukun (ْ),
                 tanween forms, and superscript alef (ٰ).

        Example:
            "مُحَمَّد" → "محمد"
            "كِتَابٌ" → "كتاب"
        """
        return self.diacritics_pattern.sub('', text)

    # =========================================================================
    # STEP 2: TATWEEL REMOVAL
    # =========================================================================

    def _remove_tatweel(self, text):
        """
        Remove tatweel (kashida) characters used for visual text stretching.

        Example:
            "كـــتاب" → "كتاب"
            "مـحـمـد" → "محمد"
        """
        return self.tatweel_pattern.sub('', text)

    # =========================================================================
    # STEP 3: ARABIC CHARACTER NORMALIZATION
    # =========================================================================

    def _normalize_characters(self, text):
        """
        Normalize Arabic character variants to their base forms.

        Normalizations:
          - Alef variants: أ / إ / آ → ا
          - Alef maksura: ى → ي
          - Waw with hamza: ؤ → و
          - Yaa with hamza: ئ → ي

        Note: Taa marbuta (ة) is NOT normalized to haa (ه) because it would
        break word endings like "مساعدة" (help) → "مساعده" which then gets
        caught by dialect patterns. Taa marbuta carries grammatical meaning.

        Why normalize alef?
          Arabic has multiple forms of alef depending on grammatical rules.
          For NLP matching, we want one canonical form so "أحمد" and "احمد"
          match the same person's name.
        """
        # Alef variants → bare alef (ا)
        text = text.replace('أ', 'ا')  # Alef with hamza above
        text = text.replace('إ', 'ا')  # Alef with hamza below
        text = text.replace('آ', 'ا')  # Alef with madda

        # Alef maksura → yaa
        text = text.replace('ى', 'ي')

        # Waw with hamza → waw
        text = text.replace('ؤ', 'و')

        # Yaa with hamza → yaa
        text = text.replace('ئ', 'ي')

        return text

    # =========================================================================
    # STEP 4: EGYPTIAN DIALECT STANDARDIZATION
    # =========================================================================

    def _normalize_dialect(self, text):
        """
        Standardize Egyptian Arabic dialect variations to canonical forms.

        Maps common Egyptian colloquial spellings and expressions to
        consistent standardized forms that the AI engine can recognize.

        Uses word-boundary-aware matching for short patterns to prevent
        accidental replacement inside longer words (e.g., "ده" inside "مساعده").

        The mappings are applied longest-first to prevent partial matches.
        """
        for original, standard, use_boundary in self.dialect_map:
            if use_boundary:
                # Use regex word boundary for short patterns
                # Arabic word boundary: start/end of string, space, or non-Arabic char
                pattern = r'(?<![^\s؀-ۿ])' + re.escape(original) + r'(?![^\s؀-ۿ])'
                # Simpler: match as standalone word (surrounded by spaces or at edges)
                text = re.sub(r'(?:^|(?<=\s))' + re.escape(original) + r'(?=\s|$)', standard, text)
            else:
                # Direct replacement for longer, unambiguous patterns
                text = text.replace(original, standard)

        return text

    def _build_dialect_map(self):
        """
        Build the Egyptian dialect normalization mapping table.

        Organized by category for maintainability. Entries are stored as
        (original, standardized, use_word_boundary) tuples, sorted longest-first
        to prevent partial replacements.

        use_word_boundary=True for short patterns (<=3 chars) that could appear
        inside other words. False for longer, unambiguous patterns.
        """
        mappings = []

        # --- Multi-word Phrases (always safe, no boundary needed) ---
        mappings.extend([
            ("مش عارفه", "لا اعرف", False),
            ("مش عارف", "لا اعرف", False),
            ("عامله ايه", "كيف حالك", False),
            ("عامل ايه", "كيف حالك", False),
            ("مش قادر", "لا استطيع", False),
            ("ليه كده", "لماذا هكذا", False),
            ("ايه ده", "ما هذا", False),
            ("في اين", "فين", False),
            ("فى اين", "فين", False),
        ])

        # --- Question Words ---
        mappings.extend([
            ("إيه", "ايه", False),
            ("ازاى", "ازاي", False),
            ("إزاى", "ازاي", False),
            ("إزاي", "ازاي", False),
            ("إمتى", "امتي", False),
            ("امتى", "امتي", False),
            ("إمتي", "امتي", False),
        ])

        # --- Common Verbs (longer patterns, safe) ---
        mappings.extend([
            ("معرفش", "لا اعرف", False),
            ("مقدرش", "لا استطيع", False),
            ("عايزه", "اريد", False),
            ("عاوزه", "اريد", False),
            ("عايز", "اريد", False),
            ("عاوز", "اريد", False),
            ("مفيش", "لا يوجد", False),
            ("ماكنش", "لم يكن", False),
            ("بيعمل", "يعمل", False),
            ("دلوقتي", "الان", False),
            ("دلوقت", "الان", False),
            ("علشان", "لان", False),
            ("عشان", "لان", False),
            ("بتاعي", "خاصي", False),
            ("بتاعت", "خاصة", False),
            ("بتاع", "خاص", False),
        ])

        # --- Greetings (longer patterns, safe) ---
        mappings.extend([
            ("ازيكم", "كيف حالكم", False),
            ("ازيك", "كيف حالك", False),
        ])

        # --- Common Expressions (longer patterns, safe) ---
        mappings.extend([
            ("كمان", "ايضا", False),
            ("برضو", "ايضا", False),
            ("برضه", "ايضا", False),
            ("خلاص", "حسنا", False),
            ("ماشي", "حسنا", False),
            ("اقدر", "استطيع", False),
            ("بعمل", "اعمل", False),
            ("هعمل", "ساعمل", False),
        ])

        # --- Short patterns (NEED word boundary to avoid matching inside words) ---
        mappings.extend([
            ("كده", "هكذا", True),
            ("لسه", "لم بعد", True),
            ("تمام", "بخير", True),
            ("طيب", "حسنا", True),
            ("فين", "اين", True),
            ("لية", "لماذا", True),
            ("ازى", "ازاي", True),
            ("دول", "هؤلاء", True),
            ("دي", "هذه", True),
            ("ده", "هذا", True),
            ("مش", "ليس", True),
            ("بس", "فقط", True),
            ("فى", "في", True),
        ])

        # Sort by length (longest first) to prevent partial matches
        mappings.sort(key=lambda x: len(x[0]), reverse=True)

        # Remove no-op mappings
        self.dialect_map = [(orig, std, boundary) for orig, std, boundary in mappings if orig != std]

    # =========================================================================
    # STEP 5: WHITESPACE NORMALIZATION
    # =========================================================================

    def _normalize_whitespace(self, text):
        """
        Collapse multiple whitespace characters into a single space
        and strip leading/trailing whitespace.

        Example:
            "مرحبا    كيف   حالك" → "مرحبا كيف حالك"
        """
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()


# Module-level convenience instance for quick usage
_default_normalizer = None


def normalize_arabic(text):
    """
    Module-level convenience function for quick normalization.

    Usage:
        from ai_engine.arabic_normalizer import normalize_arabic
        clean = normalize_arabic("إيه الأخبار؟")
    """
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = ArabicNormalizer()
    return _default_normalizer.normalize(text)
