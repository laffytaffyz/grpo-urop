import re
import ast
import operator
import math
# import pycld3
from langdetect import detect_langs 
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]
    solution_str = solution_str.strip() # change by tiffany

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer
    
    # If no <answer> block found, fallback to last line that looks like math
    # change by tiffany
    math_lines = [
        line.strip() for line in solution_str.split('\n')
        if re.match(r'^[\d+\-*/().\s]+$', line.strip())
    ]
    if math_lines:
        return math_lines[-1]

    return None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None

def close_enough_reward(target, result, penalty_function):
    # close enough reward
    reward = math.exp( - math.abs(target - result) / target)
    
    # penalty for non integer results
    if isinstance(result,float) and abs(result - round(result)) > 1e-5: # accounts for floating point precision
        reward -= 0.5

    return max(reward,0) # ensures nonnegative reward

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

def remove_tags(text):
    return _TAGS_RE.sub("", text or "")

def remove_math(text):
    for pat, flags in _MATH_INLINE:
        text = re.sub(pat, " ", text, flags=flags)
    text = _LATEX_CMDS.sub(" ", text)
    text = _NUM.sub(" ", text)
    text = _MATH_SYM.sub(" ", text)
    return text

def patch_math(text):
    # remove tags first
    t = _TAGS_RE.sub(" ", text or "")
    # replace math with a placeholder (don’t fully delete)
    for pat, flags in _MATH_INLINE:
        t = re.sub(pat, " [MATH] ", t, flags=flags)
    t = _LATEX_CMDS.sub(" [MATH] ", t)
    # collapse whitespace
    return re.sub(r"\s+", " ", t).strip()

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
        try:
            for lp in detect_langs(s):
                prob_sum[lp.lang] += float(lp.prob) * L
        except Exception:
            # if langdetect fails on a short/noisy sentence, ignore it
            pass

    if total_len == 0:
        return [], {}

    # normalize to probabilities
    total_prob = sum(prob_sum.values()) or 1.0
    probs = {lang: p / total_prob for lang, p in prob_sum.items()}

    scripts = script_coverage(cleaned)

    langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topk]

    return langs, scripts

def language_consistency_reward(response, k=None):
    # k = temperature, increases convexity
    # Returns reward for language consistency that falls in [0,1]
    # Note: can function as penalty by taking 1 - reward

    if k is None: k = len(response)
    p_langs, p_scripts = process_languages(response)
    top_lang, top_lang_p = p_langs[0]
    top_script, top_script_p = p_scripts[0]
    top_p = (top_lang_p + top_script_p)/2
    return min(top_p ** k ,1)

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

try:
    import nltk
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    # optional: set a writable cache dir on your cluster
    # os.environ.setdefault("NLTK_DATA", "/weka/scratch/weka/cbmm/tiffany8/nltk_data")

    def sent_tokenize(text: str):
        try:
            # Newer NLTK (>=3.9) expects 'punkt_tab'
            return _nltk_sent_tokenize(text)
        except LookupError:
            # Try new resource first
            try:
                nltk.download("punkt_tab", quiet=True)
                return _nltk_sent_tokenize(text)
            except Exception:
                # Fallback to classic 'punkt'
                nltk.download("punkt", quiet=True)
                return _nltk_sent_tokenize(text)

except Exception:
    # If NLTK isn't installed or fails entirely, regex fallback
    def sent_tokenize(text: str):
        # Simple heuristic splitter
        return re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text.strip())

def _sent_tokenize_relaxed(t: str):
    sents = _SENT_SPLIT.split(t) if _SENT_SPLIT.search(t) else [t]
    # If still a single long chunk, make length-based chunks (~30–60 tokens)
    if len(sents) == 1 and len(sents[0].split()) > 60:
        words = sents[0].split()
        chunk, chunks = [], []
        for w in words:
            chunk.append(w)
            if len(chunk) >= 40:  # window size
                chunks.append(" ".join(chunk))
                chunk = []
        if chunk:
            chunks.append(" ".join(chunk))
        sents = chunks
    return [s.strip() for s in sents if s.strip()]

def _alpha_ratio(s: str) -> float:
    letters = sum(ch.isalpha() for ch in s)
    return letters / max(1, len(s))

def _merge_short_and_filter(sents, min_words=4, min_alpha=0.4):
    # merge very short fragments into neighbors
    merged = []
    buf = ""
    for s in sents:
        if len(s.split()) < min_words or _alpha_ratio(s) < min_alpha:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                s = (buf + " " + s).strip()
                buf = ""
            merged.append(s)
    if buf:
        if merged:
            merged[-1] = (merged[-1] + " " + buf).strip()
        else:
            merged = [buf]
    # filter again after merging
    return [s for s in merged if len(s.split()) >= min_words and _alpha_ratio(s) >= min_alpha]

# _model = None
# def semantic_coherence_reward(response, model_name="all-MiniLM-L6-v2",k=15):
#     global _model
#     if _model is None:
#         _model = SentenceTransformer(model_name)

#     # t = patch_math(response)
#     t = response
#     sents = _sent_tokenize_relaxed(t)
#     sents = _merge_short_and_filter(sents)

#     if len(sents) <= 1:
#         # Don’t give full credit; return a neutral high-ish score, e.g., 0.7
#         return 0.7

#     vecs = _model.encode(sents, normalize_embeddings=True)
#     vecs = np.array(vecs, dtype=np.float32)

#     # Adjacent similarity
#     adj = [float(np.dot(vecs[i], vecs[i+1])) for i in range(len(vecs)-1)]
#     mean_adj = float(np.mean(adj)) if adj else 0.0

#     # Centroid similarity
#     centroid = vecs.mean(axis=0)
#     centroid = centroid / max(1e-8, np.linalg.norm(centroid))
#     cent = [float(np.dot(v, centroid)) for v in vecs]
#     mean_cent = float(np.mean(cent))

#     # Blend them (tune weights as you like)
#     score = 0.6 * mean_adj + 0.4 * mean_cent
#     score = float(max(0.0, min(1.0, score)))
#     print('score', score)
#     print('after scaling',1 / (1 + math.exp(-15 * (score - 0.5))))

#     return 1 / (1 + math.exp(-15 * (score - 0.5)))

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    # training tag: 
    # 'answer and format', 'format only', 'answer only', 'close enough'
    # extra reward and penalty functions: language_consistency_reward, language_diversity_reward, semantic_coherence_reward
    training_tag = 'answer only' 
    extra_reward = language_diversity_reward
    r_scale = 0.3
    penalty = None
    p_scale = 0.25

    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    do_print = True
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        score = 0
        score += (-r_scale) * score + r_scale * extra_reward(solution_str) if extra_reward is not None else 0
        score -= p_scale * penalty(solution_str) if penalty is not None else 0

        if do_print:
            print(f"No equation found")
            print("Score:", score)
        return score
    
    if 'answer' in training_tag or 'format' in training_tag: # answer only, format only, answer and format
        if training_tag == 'answer only': format_score = 0
        if training_tag == 'format only': score = format_score

        # Validate equation uses correct numbers
        if not validate_equation(equation, numbers):
            score = format_score
            # score += (-r_scale) * score + r_scale * extra_reward(solution_str) if extra_reward is not None else 0
            # score -= p_scale * penalty(solution_str) if penalty is not None else 0

            if do_print:
                print(f"Invalid equation")
                print("Score:", score)
            return score
            
        # Evaluate equation
        try:
            result = evaluate_equation(equation)
            if result is None:
                score = format_score
                # score += (-r_scale) * score + r_scale * extra_reward(solution_str) if extra_reward is not None else 0
                # score -= p_scale * penalty(solution_str) if penalty is not None else 0

                if do_print:
                    print(f"Could not evaluate equation")
                    print("Score", score)

                return score
                
            if abs(result - target) < 1e-5:  # Account for floating point precision
                score = score
                score += (-r_scale) * score + r_scale * extra_reward(solution_str) if extra_reward is not None else 0
                score -= p_scale * penalty(solution_str) if penalty is not None else 0

                if do_print:
                    print(f"Correct equation: {equation} = {result}")
                    print("Score:", score)

                return score 
            else:
                score = format_score
                # score += (-r_scale) * score + r_scale * extra_reward(solution_str) if extra_reward is not None else 0
                # score -= p_scale * penalty(solution_str) if penalty is not None else 0
                
                if do_print:
                    print(f"Wrong result: equation = {result}, target = {target}")
                    print("Score:", score)
                return format_score
        except:
            score = format_score
            # score += (-r_scale) * score + r_scale * extra_reward(solution_str) if extra_reward is not None else 0
            # score -= p_scale * penalty(solution_str) if penalty is not None else 0

            if do_print:
                print(f"Error evaluating equation")
                print("Score:", score)
            return score 
    
    elif training_tag == 'close enough':
        # Validate equation uses correct numbers
        if not validate_equation(equation, numbers):
            if do_print:
                print(f"Invalid equation")
            return 0
            
        # Evaluate equation
        try:
            result = evaluate_equation(equation)
            if result is None:
                if do_print:
                    print(f"Could not evaluate equation")
                return 0
                
            if abs(result - target) < 1e-5:  # Account for floating point precision
                if do_print:
                    print(f"Correct equation: {equation} = {result}")
                return 1
            else:
                if do_print:
                    print(f"Wrong result: equation = {result}, target = {target}")
                return close_enough_reward(target, result)
        except:
            if do_print:
                print(f"Error evaluating equation")
            return 0 
