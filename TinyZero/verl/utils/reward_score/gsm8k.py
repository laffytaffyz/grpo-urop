# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math
from collections import defaultdict

def extract_solution(solution_str, method='strict'):
    marker = "Let me solve this step by step."
    lower_str = solution_str.lower()
    marker_lower = marker.lower()
    marker_idx = lower_str.rfind(marker_lower)
    if marker_idx != -1:
        after_marker = solution_str[marker_idx + len(marker):]
        think_close = after_marker.lower().find("</think>")
        if think_close != -1:
            solution_str = after_marker[think_close + len("</think>"):]
        else:
            solution_str = after_marker
    matches = re.findall(r"-?\d+(?:\.\d+)?", solution_str)
    return matches[-1] if matches else None

    # assert method in ['strict', 'flexible']

    # if method == 'strict':
    #     # this also tests the formatting of the model
    #     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    #     if solution is None:
    #         final_answer = None
    #     else:
    #         final_answer = solution.group(0)
    #         final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    # elif method == 'flexible':
    #     answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    #     final_answer = None
    #     if len(answer) == 0:
    #         # no reward is there is no answer
    #         pass
    #     else:
    #         invalid_str = ['', '.']
    #         # find the last number that is not '.'
    #         for final_answer in reversed(answer):
    #             if final_answer not in invalid_str:
    #                 break
    # return final_answer

_MATH_INLINE = [
    (r"\$\$.*?\$\$", re.DOTALL),                 # $$ ... $$
    (r"\$.*?\$", 0),                             # $ ... $
    (r"\\\((?:.|\n)*?\\\)", 0),                  # \( ... \)
    (r"\\\[(?:.|\n)*?\\\]", 0),                  # \[ ... \]
    (r"\\boxed\{(?:.|\n)*?\}", 0),               # \boxed{...}
    (r"\\frac\{(?:.|\n)*?\}\{(?:.|\n)*?\}", 0),  # \frac{...}{...}
]
_TAGS_RE = re.compile(r"</?\s*(?:think|answer)\s*>", flags=re.IGNORECASE)
_NUM = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:[,_]\d{3})*|\d+)(?:\.\d+)?(?!\w)")
_MATH_SYM = re.compile(r"[+\-*/=^%°×÷∙·≈≃≅≠≤≥<>(){}\[\]|\\:]")
_LATEX_CMDS = re.compile(r"\\[A-Za-z]+")  # \alpha, \cdots, etc.

def remove_math(text):
    for pat, flags in _MATH_INLINE:
        text = re.sub(pat, " ", text, flags=flags)
    text = _LATEX_CMDS.sub(" ", text)
    text = _NUM.sub(" ", text)
    text = _MATH_SYM.sub(" ", text)
    return text

# simple sentence splitter (replace with your NLTK wrapper if you kept it)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_BULLET_SPLIT = re.compile(r'(?:^|\n)\s*[-•*]\s+(?=\S)')

SCRIPT_RANGES = {
    # Latin (basic + extended)
    "latin": [
        (0x0041, 0x005A), (0x0061, 0x007A),         # A-Z, a-z
        (0x00C0, 0x00FF), (0x0100, 0x024F),         # Latin-1 Supplement, Extended-A/B
        (0x1E00, 0x1EFF),                           # Latin Extended Additional
    ],
    "cyrillic": [
        (0x0400, 0x04FF), (0x0500, 0x052F),
        (0x2DE0, 0x2DFF), (0xA640, 0xA69F),
    ],
    "greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "hebrew": [(0x0590, 0x05FF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "thaana": [(0x0780, 0x07BF)],
    "nko": [(0x07C0, 0x07FF)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "oriya": [(0x0B00, 0x0B7F)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "sinhala": [(0x0D80, 0x0DFF)],
    "thai": [(0x0E00, 0x0E7F)],
    "lao": [(0x0E80, 0x0EFF)],
    "tibetan": [(0x0F00, 0x0FFF)],
    "myanmar": [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)],
    "georgian": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
    "ethiopic": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)],
    "cherokee": [(0x13A0, 0x13FF), (0xAB70, 0xABBF)],
    "canadian_aboriginal": [(0x1400, 0x167F)],
    "ogham": [(0x1680, 0x169F)],
    "runic": [(0x16A0, 0x16FF)],
    "tagalog": [(0x1700, 0x171F)],
    "hanunoo": [(0x1720, 0x173F)],
    "buhid": [(0x1740, 0x175F)],
    "tagbanwa": [(0x1760, 0x177F)],
    "khmer": [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    "mongolian": [(0x1800, 0x18AF)],
    "yi": [(0xA000, 0xA48F), (0xA490, 0xA4CF)],
    "bopomofo": [(0x3100, 0x312F), (0x31A0, 0x31BF)],
    "hiragana": [(0x3040, 0x309F)],
    "katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "hangul": [
        (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F),
        (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF),
    ],
    # CJK (Han) including extensions
    "cjk": [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # Ext A
        (0x20000, 0x2A6DF), # Ext B
        (0x2A700, 0x2B73F), # Ext C
        (0x2B740, 0x2B81F), # Ext D
        (0x2B820, 0x2CEAF), # Ext E
        (0x2CEB0, 0x2EBEF), # Ext F
        (0x30000, 0x3134F), # Ext G/H
    ],
    # Symbols/Emoji (coarse)
    "emoji": [
        (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F), (0x1F780, 0x1F7FF), (0x1F800, 0x1F8FF),
        (0x1F900, 0x1F9FF), (0x1FA00, 0x1FAFF), (0x1FB00, 0x1FBFF),
        (0x2600, 0x26FF), (0x2700, 0x27BF),
        (0x1F1E6, 0x1F1FF), # regional indicator flags
        (0xFE0F, 0xFE0F),   # VS16 (emoji presentation)
    ],
    # Fullwidth / CJK punctuation (helps separate “looks like CJK”)
    "cjk_punct_fullwidth": [(0x3000, 0x303F), (0xFF00, 0xFFEF)],
    # Numbers & punctuation (optional buckets)
    "digits": [(0x0030, 0x0039)],
    "ascii_punct": [(0x0020, 0x002F), (0x003A, 0x0040), (0x005B, 0x0060), (0x007B, 0x007E)],
}

# Optional: count only “letters” buckets for normalization
LETTER_BUCKETS = {
    "latin","cyrillic","greek","hebrew","arabic","thaana","nko","devanagari","bengali",
    "gurmukhi","gujarati","oriya","tamil","telugu","kannada","malayalam","sinhala","thai","lao",
    "tibetan","myanmar","georgian","ethiopic","cherokee","canadian_aboriginal","ogham","runic",
    "tagalog","hanunoo","buhid","tagbanwa","khmer","mongolian","yi","bopomofo","hiragana",
    "katakana","hangul","cjk"
}

def _in_any_range(cp: int, ranges):
    for a, b in ranges:
        if a <= cp <= b:
            return True
    return False

def script_coverage(text: str, normalize="letters", include_symbols=True):
    """
    Returns {script: share} normalized either by:
      - 'letters' (default): only letter scripts in denominator
      - 'all': denominator is all counted chars (letters + digits + punctuation + emoji/fullwidth if included)
    """
    t = text
    counts = defaultdict(int)
    total_letters = 0
    total_all = 0

    for ch in t:
        cp = ord(ch)

        found = None
        for name, ranges in SCRIPT_RANGES.items():
            if name not in LETTER_BUCKETS and not include_symbols:
                continue
            if _in_any_range(cp, ranges):
                found = name
                counts[name] += 1
                break

        # Track totals
        if found:
            total_all += 1
            if found in LETTER_BUCKETS:
                total_letters += 1

    denom = total_letters if normalize == "letters" else total_all
    denom = max(denom, 1)

    # Normalize
    coverage = {k: v / denom for k, v in counts.items() if (normalize != "letters") or (k in LETTER_BUCKETS)}
    # Add a catch-all bucket if you want to see what's left
    if normalize == "all":
        other = max(len(t) - total_all, 0) / max(len(t), 1)
        coverage["other_unclassified"] = other

    # Sort high to low
    return sorted(coverage.items(), key=lambda kv: kv[1], reverse=True)

def process_languages(text,topk=10):
    '''
    returns language and script distribution in decreasing order
    '''

    # removes math
    text = remove_math(text)    

    # collapse whitespace a bit
    text = re.sub(r"\s+", " ", text).strip()
    
    # processes languages in clean text
    if not text:
        return {
            "english_coverage": 0.0,
            "top" : []
        }

    cleaned = text

    # 2) split into sentences/chunks
    chunks = _SENT_SPLIT.split(cleaned) if _SENT_SPLIT.search(cleaned) else [cleaned]
    sents = [b.strip() for c in chunks for b in _BULLET_SPLIT.split(c) if b.strip()]

    # 3) run langdetect per sentence and aggregate by length
    prob_sum = defaultdict(float)
    total_len = 0
    for s in sents:
        L = max(len(s), 1)
        total_len += L
        # try:
        #     for lp in detect_langs(s):
        #         prob_sum[lp.lang] += float(lp.prob) * L
        # except Exception:
        #     # if langdetect fails on a short/noisy sentence, ignore it
        #     pass

    if total_len == 0:
        return [], {}

    # normalize to probabilities
    total_prob = sum(prob_sum.values()) or 1.0
    probs = {lang: p / total_prob for lang, p in prob_sum.items()}

    scripts = script_coverage(cleaned)

    langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topk]

    return langs, scripts

def language_diversity_reward(response, k=3):
    # k = temperature, increases concavity
    # Returns reward for language diversity that falls in [0,1]
    # Note: can function as penalty by taking 1 - reward

    p_langs, p_scripts = process_languages(response)
    lang_count = 0

    # for lang, prob in p_langs: 
    #     if prob > 1e-5: 
    #         lang_count += 1
    #     else:
    #         break 
    
    for lang, prob in p_scripts:
        if prob > 1e-5:
            lang_count += 1
        else: break

    return (math.atan(lang_count)/(math.pi/2) - 0.5) * 2 # zero reward with one language, zero to one
    # return (1 - 1 / (lang_count**(1/k))) if lang_count > 1 else 0


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1., do_print=False, extra_info=None):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    do_print = extra_info is not None and extra_info['do_print']
    
    print('do_print value is', do_print)
    do_print = True # debug
    if do_print:
        print(f"--------------------------------")
        print(f"Desired answer: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        # score = language_diversity_reward(solution_str)
        score = 0
        if do_print:
            print(f"No answer found, score:", score)
        return score
    else:
        if answer == ground_truth:
            # score = language_diversity_reward(solution_str)
            score = 1
            if do_print:
                print("Extracted answer correct, score:", score)
            return score
        else:
            score = 0.1
            # score = language_diversity_reward(solution_str)
            print('Extracted answer incorrect, score:', score)
            return score