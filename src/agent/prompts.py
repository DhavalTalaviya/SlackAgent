"""Loads prompt templates from the project-root prompts/ folder into named
string constants, used with str.format(...) at each call site. Keeping
prompt text in plain files instead of f-strings buried inside node logic
makes them easy to find and tweak without touching graph code, and makes a
prompt-wording change show up as its own clean diff."""

import os

# This file lives at src/agent/prompts.py -- three levels up (agent -> src ->
# project root) to reach the root-level prompts/ folder.
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts"
)


def _load(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
        # Strip the trailing newline convention text files end with -- keeps
        # the loaded prompt text identical to what the original inline
        # f-strings produced, not off by one character.
        return f.read().rstrip("\n")


CONTEXTUALIZE_QUESTION = _load("contextualize_question.txt")
CONTEXTUALIZE_PRIOR_RANGE = _load("contextualize_prior_range.txt")
ANALYZE_AND_REWRITE = _load("analyze_and_rewrite.txt")
SYNTHESIZE_ANSWER = _load("synthesize_answer.txt")
