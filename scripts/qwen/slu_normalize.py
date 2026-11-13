"""Parsing + normalisation + scoring for the Qwen structured SLU output (C1 semantic pilot).

Normalisation (applied identically to reference and hypothesis transcripts and entity values):
  1. NFKC, lowercase, curly quotes -> straight; 'a.m.'/'p.m.' -> am/pm
  2. "%" -> " percent", "&" -> " and ", hyphen/slash -> space
  3. times H:MM -> words ("5:30" -> "five thirty", "5:00" -> "five", "5:05" -> "five oh five")
  4. ordinals 1st/2nd/3rd/Nth -> words ("21st" -> "twenty first"); integers 0-9999 -> cardinal words
     ("5" -> "five", "1990" -> "one thousand nine hundred ninety"); decimals "2.5" -> "two point five"
  5. join apostrophe tokens ("derek 's" -> "derek's"); drop remaining punctuation except apostrophes
  6. collapse whitespace
Entity match = same type and same normalised value (multiset); a secondary score ignores spaces inside
values ("grass market" == "grassmarket"). Intent = exact scenario_action label.
"""
import json
import re
import unicodedata

ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen ' \
       'seventeen eighteen nineteen'.split()
TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
ORD = dict(one='first', two='second', three='third', five='fifth', eight='eighth', nine='ninth', twelve='twelfth')


def cardinal(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ('' if n % 10 == 0 else ' ' + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + cardinal(n % 100))
    return cardinal(n // 1000) + ' thousand' + ('' if n % 1000 == 0 else ' ' + cardinal(n % 1000))


def ordinal(n):
    words = cardinal(n).split()
    last = words[-1]
    if last in ORD:
        words[-1] = ORD[last]
    elif last.endswith('y'):
        words[-1] = last[:-1] + 'ieth'
    else:
        words[-1] = last + 'th'
    return ' '.join(words)


def _time(m):
    h, mm = int(m.group(1)), int(m.group(2))
    if mm == 0:
        return cardinal(h)
    return cardinal(h) + (' oh ' + ONES[mm] if mm < 10 else ' ' + cardinal(mm))


def normalize(s):
    s = unicodedata.normalize('NFKC', str(s)).lower().replace('’', "'").replace('‘', "'")
    s = re.sub(r'\b([ap])\.\s?m\.?', r'\1m', s)
    s = s.replace('%', ' percent').replace('&', ' and ')
    s = re.sub(r'[-/]', ' ', s)
    s = re.sub(r'\b(\d{1,2}):(\d{2})\b', _time, s)
    s = re.sub(r'\b(\d{1,4})(st|nd|rd|th)\b', lambda m: ordinal(int(m.group(1))), s)
    s = re.sub(r'\b(\d{1,4})\.(\d+)\b', lambda m: cardinal(int(m.group(1))) + ' point ' +
               ' '.join(ONES[int(c)] for c in m.group(2)), s)
    s = re.sub(r'\d{1,4}', lambda m: ' ' + cardinal(int(m.group(0))) + ' ', s)
    s = re.sub(r"\s+'", "'", s)
    s = re.sub(r"[^\w\s']", ' ', s)
    return ' '.join(s.split())


def parse(text):
    """Return (obj, status). status in ok / no_json / bad_json / bad_schema."""
    t = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M).strip()
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        return None, 'no_json'
    try:
        obj = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None, 'bad_json'
    if not isinstance(obj, dict):
        return None, 'bad_schema'
    ents = obj.get('entities', [])
    good = [e for e in ents if isinstance(e, dict) and isinstance(e.get('type'), str) and isinstance(e.get('value'), str)] \
        if isinstance(ents, list) else []
    out = dict(transcript=obj.get('transcript') if isinstance(obj.get('transcript'), str) else '',
               intent=obj.get('intent') if isinstance(obj.get('intent'), str) else '',
               entities=[(e['type'].strip().lower(), e['value']) for e in good],
               dropped_entities=(len(ents) - len(good)) if isinstance(ents, list) else 1)
    return out, 'ok' if isinstance(ents, list) else 'bad_schema'


def wer_counts(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def entity_counts(ref, hyp, nospace=False):
    f = (lambda v: normalize(v).replace(' ', '')) if nospace else normalize
    R = [(t, f(v)) for t, v in ref]; H = [(t, f(v)) for t, v in hyp]
    tp = 0; pool = R[:]
    for x in H:
        if x in pool:
            pool.remove(x); tp += 1
    return tp, len(H), len(R)


if __name__ == '__main__':   # self-test of the rules on fixed strings (no data)
    cases = [('Wake me up at 5 am this week.', 'wake me up at five am this week'), ("derek 's", "derek's"),
             ('5:30 p.m.', 'five thirty pm'), ('the 21st', 'the twenty first'), ('5:05', 'five oh five'),
             ('R&B', 'r and b'), ('50%', 'fifty percent'), ('2.5', 'two point five'), ('1990', 'one thousand nine hundred ninety'),
             ("Five o'clock", "five o'clock"), ('to-do list', 'to do list')]
    for a, b in cases:
        print(('OK ' if normalize(a) == b else 'FAIL'), repr(a), '->', repr(normalize(a)), '| expected', repr(b))
    print(parse('```json\n{"transcript": "x", "intent": "alarm_set", "entities": [{"type": "time", "value": "5 am"}]}\n```'))
    print(parse('not json'))
    print(entity_counts([('place_name', 'grassmarket')], [('place_name', 'grass market')]),
          entity_counts([('place_name', 'grassmarket')], [('place_name', 'grass market')], nospace=True))
