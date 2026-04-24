#!/usr/bin/env python3

# expands a search query with synonym substitutions using WordNet + a safe fallback list
# used by cli_test.py to generate richer retrieval queries for fact-checking

import re


MAX_QUERIES = 3
MAX_REPLACEMENTS = 2

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}

# WordNet can be noisy for short news claims. These safe fallbacks cover common
# fact-checking verbs while WordNet adds extra lexical coverage when available.
SAFE_SYNONYMS = {
    "ban": ["prohibit"],
    "banned": ["prohibited"],
    "claim": ["assert"],
    "claims": ["asserts"],
    "drop": ["fall", "decline"],
    "dropped": ["fell", "declined"],
    "fake": ["false"],
    "help": ["assist"],
    "rise": ["increase"],
    "rose": ["increased"],
    "says": ["states"],
}


# strips punctuation and lowercases a token for lookup
def _normalise_token(token: str) -> str:
    return re.sub(r"[^A-Za-z-]", "", token).lower()


# returns True if a token is worth expanding (not a stopword, number, or acronym)
def _is_expandable(token: str) -> bool:
    cleaned = _normalise_token(token)
    return (
        len(cleaned) >= 4
        and cleaned not in STOPWORDS
        and not any(char.isdigit() for char in token)
        and not token.isupper()
    )


# looks up synonyms via WordNet, returns empty list if nltk is unavailable
def _wordnet_synonyms(word: str, max_synonyms: int = 2) -> list[str]:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return []

    synonyms = []
    try:
        synsets = wn.synsets(word)
    except LookupError:
        return []

    for synset in synsets[:3]:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word and synonym.isascii() and synonym not in synonyms:
                synonyms.append(synonym)
            if len(synonyms) >= max_synonyms:
                return synonyms

    return synonyms


# returns synonyms for a word, preferring the safe fallback list over WordNet
def get_query_synonyms(word: str, max_synonyms: int = 2) -> list[str]:
    cleaned = _normalise_token(word)
    synonyms = SAFE_SYNONYMS.get(cleaned, []).copy()

    for synonym in _wordnet_synonyms(cleaned, max_synonyms=max_synonyms):
        if synonym not in synonyms:
            synonyms.append(synonym)

    return synonyms[:max_synonyms]


# generates up to max_queries variants of a query by swapping in synonyms one word at a time
def expand_search_queries(query: str, max_queries: int = MAX_QUERIES) -> list[str]:
    words = query.split()
    queries = [query]

    for index, token in enumerate(words):
        if len(queries) >= max_queries:
            break
        if not _is_expandable(token):
            continue

        for synonym in get_query_synonyms(token):
            if len(queries) >= max_queries:
                break
            replacement = words.copy()
            replacement[index] = synonym
            expanded = " ".join(replacement)
            if expanded != query and expanded not in queries:
                queries.append(expanded)

        if len(queries) >= MAX_REPLACEMENTS + 1:
            break

    return queries[:max_queries]
