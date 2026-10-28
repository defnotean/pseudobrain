"""Problems for custom_code_bench.py: self-contained Python tasks with hidden tests and reference solutions.

Each problem: id, tier (easy/medium/hard/fix), prompt (what the model sees), tests (asserts run after the model's
code), ref (a reference solution, never shown to the model; `custom_code_bench.py --check-refs` runs the tests
against it). Written for this repo, so none of it can appear verbatim in public training sets.
"""

PROBLEMS = [
    # ---------------------------------------------------------------- easy
    dict(id="rle_encode", tier="easy",
         prompt="Write `rle_encode(s: str) -> str` that run-length encodes a string as character followed by count, "
                "e.g. 'aaabcc' -> 'a3b1c2'. The empty string encodes to ''.",
         tests="""
assert rle_encode('aaabcc') == 'a3b1c2'
assert rle_encode('') == ''
assert rle_encode('z') == 'z1'
assert rle_encode('aaaaaaaaaaaab') == 'a12b1'
assert rle_encode('abab') == 'a1b1a1b1'
""",
         ref="""
def rle_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)
"""),
    dict(id="rle_decode", tier="easy",
         prompt="Write `rle_decode(s: str) -> str`, the inverse of a run-length encoding where each non-digit "
                "character is followed by a decimal count that may have several digits, e.g. 'a12b1' -> "
                "'aaaaaaaaaaaab'. The empty string decodes to ''.",
         tests="""
assert rle_decode('a3b1c2') == 'aaabcc'
assert rle_decode('') == ''
assert rle_decode('a12b1') == 'a' * 12 + 'b'
assert rle_decode('x100') == 'x' * 100
assert rle_decode('a1b1a1') == 'aba'
""",
         ref="""
import re
def rle_decode(s):
    return "".join(c * int(n) for c, n in re.findall(r"(\\D)(\\d+)", s))
"""),
    dict(id="balanced", tier="easy",
         prompt="Write `is_balanced(s: str) -> bool` that returns True when the brackets (), [] and {} in s are "
                "balanced and properly nested. All other characters are ignored.",
         tests="""
assert is_balanced('')
assert is_balanced('a(b[c]{d}e)f')
assert not is_balanced('(]')
assert not is_balanced('([)]')
assert not is_balanced('((')
assert not is_balanced('))((')
assert is_balanced('{[()()]}x')
""",
         ref="""
def is_balanced(s):
    pairs, st = {')': '(', ']': '[', '}': '{'}, []
    for c in s:
        if c in '([{':
            st.append(c)
        elif c in pairs:
            if not st or st.pop() != pairs[c]:
                return False
    return not st
"""),
    dict(id="roman_to_int", tier="easy",
         prompt="Write `roman_to_int(s: str) -> int` converting a valid Roman numeral (1..3999, subtractive "
                "notation such as IV, XC, CM) to an integer.",
         tests="""
assert roman_to_int('III') == 3
assert roman_to_int('IV') == 4
assert roman_to_int('LVIII') == 58
assert roman_to_int('MCMXCIV') == 1994
assert roman_to_int('MMMCMXCIX') == 3999
""",
         ref="""
def roman_to_int(s):
    v = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
    t = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and v[c] < v[s[i + 1]]:
            t -= v[c]
        else:
            t += v[c]
    return t
"""),
    dict(id="int_to_roman", tier="easy",
         prompt="Write `int_to_roman(n: int) -> str` converting 1 <= n <= 3999 to a Roman numeral using standard "
                "subtractive notation (4 -> 'IV', 900 -> 'CM').",
         tests="""
assert int_to_roman(1) == 'I'
assert int_to_roman(4) == 'IV'
assert int_to_roman(58) == 'LVIII'
assert int_to_roman(1994) == 'MCMXCIV'
assert int_to_roman(3999) == 'MMMCMXCIX'
assert int_to_roman(444) == 'CDXLIV'
""",
         ref="""
def int_to_roman(n):
    t = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = []
    for v, s in t:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)
"""),
    dict(id="merge_intervals", tier="easy",
         prompt="Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping "
                "closed intervals and returns them sorted by start. Intervals that touch ([1,2] and [2,3]) merge. "
                "The input may be unsorted and empty.",
         tests="""
assert merge_intervals([]) == []
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,2],[2,3]]) == [[1,3]]
assert merge_intervals([[5,7],[1,2]]) == [[1,2],[5,7]]
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]
""",
         ref="""
def merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
"""),
    dict(id="top_k_words", tier="easy",
         prompt="Write `top_k_frequent(words: list[str], k: int) -> list[str]` returning the k most frequent "
                "words, ordered by frequency descending and, for equal frequency, alphabetically ascending.",
         tests="""
assert top_k_frequent(['i','love','leetcode','i','love','coding'], 2) == ['i','love']
assert top_k_frequent(['the','day','is','sunny','the','the','the','sunny','is','is'], 4) == ['the','is','sunny','day']
assert top_k_frequent(['b','a','c'], 2) == ['a','b']
""",
         ref="""
from collections import Counter
def top_k_frequent(words, k):
    c = Counter(words)
    return sorted(c, key=lambda w: (-c[w], w))[:k]
"""),
    dict(id="flatten", tier="easy",
         prompt="Write `flatten(x) -> list` that flattens arbitrarily nested lists and tuples into a flat list, "
                "preserving order. Strings, dicts and other values are kept as single elements.",
         tests="""
assert flatten([]) == []
assert flatten([1,[2,[3,(4,5)]],6]) == [1,2,3,4,5,6]
assert flatten(['ab',['cd']]) == ['ab','cd']
assert flatten([[[[]]]]) == []
assert flatten([{'a':1},(None,)]) == [{'a':1},None]
""",
         ref="""
def flatten(x):
    out = []
    for e in x:
        if isinstance(e, (list, tuple)):
            out.extend(flatten(e))
        else:
            out.append(e)
    return out
"""),
    # ---------------------------------------------------------------- medium
    dict(id="lru_cache", tier="medium",
         prompt="Write a class `LRUCache` with `__init__(self, capacity: int)`, `get(self, key) -> int` (returns "
                "-1 when missing) and `put(self, key, value) -> None`. When full, put evicts the least recently "
                "used key; both get and put count as a use. All operations O(1).",
         tests="""
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
d = LRUCache(1)
d.put(5, 5); d.put(5, 6)
assert d.get(5) == 6
d.put(7, 7)
assert d.get(5) == -1
""",
         ref="""
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.c, self.d = capacity, OrderedDict()
    def get(self, k):
        if k not in self.d:
            return -1
        self.d.move_to_end(k)
        return self.d[k]
    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.c:
            self.d.popitem(last=False)
"""),
    dict(id="dijkstra", tier="medium",
         prompt="Write `shortest_paths(n: int, edges: list[tuple[int,int,int]], src: int) -> list[int]` for a "
                "directed graph with nodes 0..n-1 and edges (u, v, w), w >= 0. Return the shortest distance from "
                "src to every node, -1 for unreachable nodes.",
         tests="""
assert shortest_paths(1, [], 0) == [0]
assert shortest_paths(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0) == [0,3,1,4]
assert shortest_paths(3, [(1,2,5)], 0) == [0,-1,-1]
assert shortest_paths(3, [(0,1,0),(1,2,0)], 0) == [0,0,0]
assert shortest_paths(3, [(0,1,7),(0,1,2),(1,0,1)], 1) == [1,0,-1]
""",
         ref="""
import heapq
def shortest_paths(n, edges, src):
    g = [[] for _ in range(n)]
    for u, v, w in edges:
        g[u].append((v, w))
    d = [-1] * n
    pq = [(0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if d[u] != -1:
            continue
        d[u] = du
        for v, w in g[u]:
            if d[v] == -1:
                heapq.heappush(pq, (du + w, v))
    return d
"""),
    dict(id="topo_sort", tier="medium",
         prompt="Write `topo_order(n: int, edges: list[tuple[int,int]]) -> list[int] | None`. Nodes are 0..n-1, an "
                "edge (a, b) means a must come before b. Return the lexicographically smallest topological order, "
                "or None if the graph has a cycle.",
         tests="""
assert topo_order(3, []) == [0,1,2]
assert topo_order(4, [(3,0),(2,0),(1,3)]) == [1,2,3,0]
assert topo_order(2, [(0,1),(1,0)]) is None
assert topo_order(5, [(4,1),(3,1),(1,0)]) == [2,3,4,1,0]
assert topo_order(1, [(0,0)]) is None
""",
         ref="""
import heapq
def topo_order(n, edges):
    g = [[] for _ in range(n)]; deg = [0] * n
    for a, b in edges:
        g[a].append(b); deg[b] += 1
    h = [i for i in range(n) if deg[i] == 0]
    heapq.heapify(h)
    out = []
    while h:
        u = heapq.heappop(h); out.append(u)
        for v in g[u]:
            deg[v] -= 1
            if deg[v] == 0:
                heapq.heappush(h, v)
    return out if len(out) == n else None
"""),
    dict(id="csv_line", tier="medium",
         prompt="Write `parse_csv_line(line: str) -> list[str]` that splits one CSV line on commas. A field may be "
                "wrapped in double quotes, in which case it can contain commas, and a doubled quote \"\" inside a "
                "quoted field is a literal quote. Do not use the csv module. An empty line is one empty field.",
         tests='''
assert parse_csv_line('a,b,c') == ['a','b','c']
assert parse_csv_line('') == ['']
assert parse_csv_line('a,,c,') == ['a','','c','']
assert parse_csv_line('"x,y",z') == ['x,y','z']
assert parse_csv_line('"he said ""hi""",2') == ['he said "hi"','2']
assert parse_csv_line('""') == ['']
''',
         ref="""
def parse_csv_line(line):
    out, cur, i, q = [], [], 0, False
    while i < len(line):
        c = line[i]
        if q:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"'); i += 1
                else:
                    q = False
            else:
                cur.append(c)
        elif c == '"':
            q = True
        elif c == ',':
            out.append("".join(cur)); cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out
"""),
    dict(id="eval_expr", tier="medium",
         prompt="Write `evaluate(expr: str) -> int` for integer expressions with + - * /, parentheses, unary minus "
                "and arbitrary spaces. Usual precedence, left associative; / is integer division truncating toward "
                "zero (so -7/2 == -3). Do not use eval or exec.",
         tests="""
assert evaluate('1 + 2 * 3') == 7
assert evaluate('(1+2)*3') == 9
assert evaluate('-7/2') == -3
assert evaluate('7/-2') == -3
assert evaluate('2*-(3+4)') == -14
assert evaluate(' 10 - 4 - 3 ') == 3
assert evaluate('100/10/5') == 2
assert evaluate('--5') == 5
""",
         ref="""
def evaluate(expr):
    s = expr.replace(' ', ''); pos = 0
    def peek():
        return s[pos] if pos < len(s) else ''
    def eat():
        nonlocal pos; c = s[pos]; pos += 1; return c
    def factor():
        if peek() == '-':
            eat(); return -factor()
        if peek() == '(':
            eat(); v = add(); eat(); return v
        st = pos
        while peek().isdigit():
            eat()
        return int(s[st:pos])
    def mul():
        v = factor()
        while peek() in ('*', '/') and peek():
            op = eat(); r = factor()
            if op == '*':
                v = v * r
            else:
                q = abs(v) // abs(r); v = q if (v >= 0) == (r >= 0) else -q
        return v
    def add():
        v = mul()
        while peek() in ('+', '-') and peek():
            op = eat(); r = mul(); v = v + r if op == '+' else v - r
        return v
    return add()
"""),
    dict(id="lcs", tier="medium",
         prompt="Write `lcs(a: str, b: str) -> str` returning one longest common subsequence of a and b (any one "
                "if several exist).",
         tests="""
def _is_sub(x, y):
    it = iter(y); return all(c in it for c in x)
for a, b, n in [('abcde','ace',3), ('abc','def',0), ('AGGTAB','GXTXAYB',4), ('', 'x', 0), ('aaaa','aa',2)]:
    r = lcs(a, b)
    assert len(r) == n and _is_sub(r, a) and _is_sub(r, b), (a, b, r)
""",
         ref="""
def lcs(a, b):
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            d[i][j] = d[i+1][j+1] + 1 if a[i] == b[j] else max(d[i+1][j], d[i][j+1])
    i = j = 0; out = []
    while i < m and j < n:
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif d[i+1][j] >= d[i][j+1]:
            i += 1
        else:
            j += 1
    return "".join(out)
"""),
    dict(id="edit_distance", tier="medium",
         prompt="Write `edit_distance(a: str, b: str) -> int`, the Levenshtein distance (insert, delete, substitute "
                "each cost 1).",
         tests="""
assert edit_distance('', '') == 0
assert edit_distance('kitten', 'sitting') == 3
assert edit_distance('horse', 'ros') == 3
assert edit_distance('abc', '') == 3
assert edit_distance('intention', 'execution') == 5
""",
         ref="""
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]
"""),
    dict(id="word_wrap", tier="medium",
         prompt="Write `wrap(text: str, width: int) -> list[str]` that greedily packs whitespace-separated words "
                "into lines of at most width characters, words separated by single spaces. A word longer than "
                "width goes on its own line unbroken. No empty lines; empty or all-space text gives [].",
         tests="""
assert wrap('', 10) == []
assert wrap('   ', 3) == []
assert wrap('the quick brown fox', 10) == ['the quick', 'brown fox']
assert wrap('a bb ccc dddd', 4) == ['a bb', 'ccc', 'dddd']
assert wrap('supercalifragilistic is long', 6) == ['supercalifragilistic', 'is', 'long']
assert wrap('one  two\\nthree', 20) == ['one two three']
""",
         ref="""
def wrap(text, width):
    lines, cur = [], ''
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += ' ' + w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines
"""),
    dict(id="next_perm", tier="medium",
         prompt="Write `next_permutation(nums: list[int]) -> list[int]` returning (as a new list) the next "
                "lexicographically greater permutation of nums; if nums is the greatest, return it sorted "
                "ascending. Duplicates may occur.",
         tests="""
assert next_permutation([1,2,3]) == [1,3,2]
assert next_permutation([3,2,1]) == [1,2,3]
assert next_permutation([1,1,5]) == [1,5,1]
assert next_permutation([1,3,2]) == [2,1,3]
assert next_permutation([2,2,1,1]) == [1,1,2,2]
assert next_permutation([7]) == [7]
""",
         ref="""
def next_permutation(nums):
    a = list(nums); i = len(a) - 2
    while i >= 0 and a[i] >= a[i+1]:
        i -= 1
    if i >= 0:
        j = len(a) - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
    a[i+1:] = reversed(a[i+1:])
    return a
"""),
    dict(id="spiral", tier="medium",
         prompt="Write `spiral(matrix: list[list[int]]) -> list[int]` returning the elements of a (possibly "
                "non-square, possibly empty) matrix in clockwise spiral order starting at the top-left.",
         tests="""
assert spiral([]) == []
assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
assert spiral([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]
assert spiral([[1],[2],[3]]) == [1,2,3]
assert spiral([[1,2]]) == [1,2]
""",
         ref="""
def spiral(m):
    out = []
    m = [list(r) for r in m]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out
"""),
    dict(id="islands", tier="medium",
         prompt="Write `count_islands(grid: list[str]) -> int` where each string is a row of '1' (land) and '0' "
                "(water); islands are 4-connected land cells. The grid may be empty and up to 300x300.",
         tests="""
assert count_islands([]) == 0
assert count_islands(['11000','11000','00100','00011']) == 3
assert count_islands(['111','010','111']) == 1
assert count_islands(['101','010','101']) == 5
assert count_islands(['1' * 300] * 300) == 1
""",
         ref="""
def count_islands(grid):
    if not grid:
        return 0
    h, w = len(grid), len(grid[0]); seen = set(); n = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '1' and (i, j) not in seen:
                n += 1; st = [(i, j)]; seen.add((i, j))
                while st:
                    a, b = st.pop()
                    for x, y in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                        if 0 <= x < h and 0 <= y < w and grid[x][y] == '1' and (x, y) not in seen:
                            seen.add((x, y)); st.append((x, y))
    return n
"""),
    dict(id="base_convert", tier="medium",
         prompt="Write `convert(num: str, src: int, dst: int) -> str` converting an integer written in base src to "
                "base dst (2 <= base <= 36, digits 0-9 then A-Z, input digits case-insensitive, output uppercase, "
                "optional leading '-', no leading zeros in output, zero is '0'). Do not use int(num, base).",
         tests="""
assert convert('0', 10, 2) == '0'
assert convert('255', 10, 16) == 'FF'
assert convert('ff', 16, 10) == '255'
assert convert('-101', 2, 10) == '-5'
assert convert('ZZ', 36, 10) == '1295'
assert convert('-0', 10, 7) == '0'
assert convert('123456789', 10, 36) == '21I3V9'
""",
         ref="""
def convert(num, src, dst):
    neg = num.startswith('-'); s = num.lstrip('-'); v = 0
    for c in s:
        v = v * src + int(c, 36)
    if v == 0:
        return '0'
    d = []
    while v:
        v, r = divmod(v, dst); d.append('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[r])
    return ('-' if neg else '') + ''.join(reversed(d))
"""),
    # ---------------------------------------------------------------- hard
    dict(id="median_stream", tier="hard",
         prompt="Write a class `MedianFinder` with `add(self, x: int) -> None` and `median(self) -> float` (mean of "
                "the two middle values for an even count). add and median must be O(log n) and O(1).",
         tests="""
m = MedianFinder()
m.add(5); assert m.median() == 5.0
m.add(1); assert m.median() == 3.0
m.add(9); assert m.median() == 5.0
m.add(-2); assert m.median() == 3.0
import random
r = random.Random(1); xs = []
m2 = MedianFinder()
for _ in range(2000):
    x = r.randint(-1000, 1000); xs.append(x); m2.add(x)
s = sorted(xs)
assert m2.median() == (s[999] + s[1000]) / 2
""",
         ref="""
import heapq
class MedianFinder:
    def __init__(self):
        self.lo, self.hi = [], []
    def add(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def median(self):
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2
"""),
    dict(id="regex_match", tier="hard",
         prompt="Write `match(s: str, p: str) -> bool` implementing full-string regular expression matching where "
                "'.' matches any single character and 'x*' matches zero or more of the preceding element (which "
                "may be '.'). No other metacharacters. Do not use the re module.",
         tests="""
assert not match('aa', 'a')
assert match('aa', 'a*')
assert match('ab', '.*')
assert match('aab', 'c*a*b')
assert not match('mississippi', 'mis*is*p*.')
assert match('', 'a*b*')
assert not match('ab', '.*c')
assert match('a' * 30, 'a*a*a*a*a*a*a*a*a*a*')
""",
         ref="""
from functools import lru_cache
def match(s, p):
    @lru_cache(None)
    def dp(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], '.')
        if j + 1 < len(p) and p[j+1] == '*':
            return dp(i, j + 2) or (first and dp(i + 1, j))
        return first and dp(i + 1, j + 1)
    return dp(0, 0)
"""),
    dict(id="min_window", tier="hard",
         prompt="Write `min_window(s: str, t: str) -> str` returning the shortest substring of s that contains "
                "every character of t with multiplicity; the leftmost one if several have the same length; '' if "
                "none exists or t is empty. Must run in O(len(s) + len(t)).",
         tests="""
assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'
assert min_window('a', 'a') == 'a'
assert min_window('a', 'aa') == ''
assert min_window('abc', '') == ''
assert min_window('aaflslflsldkalskaaa', 'aaa') == 'aaa'
assert min_window('abab', 'ab') == 'ab'
s = 'x' * 200000 + 'abc' + 'y' * 200000
assert min_window(s, 'cba') == 'abc'
""",
         ref="""
from collections import Counter
def min_window(s, t):
    if not t:
        return ''
    need = Counter(t); missing = len(t); best = (0, 0); lo = 0
    for hi, c in enumerate(s, 1):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        if missing == 0:
            while need[s[lo]] < 0:
                need[s[lo]] += 1; lo += 1
            if not best[1] or hi - lo < best[1] - best[0]:
                best = (lo, hi)
            need[s[lo]] += 1; missing += 1; lo += 1
    return s[best[0]:best[1]]
"""),
    dict(id="weighted_jobs", tier="hard",
         prompt="Write `max_weight(jobs: list[tuple[int,int,int]]) -> int`. Each job is (start, end, weight) and "
                "occupies [start, end) (a job ending at 5 is compatible with one starting at 5). Return the maximum "
                "total weight of mutually non-overlapping jobs. Up to 100000 jobs; must be O(n log n).",
         tests="""
assert max_weight([]) == 0
assert max_weight([(1,3,5),(2,5,6),(4,6,5),(6,7,4),(5,8,11),(7,9,2)]) == 17
assert max_weight([(1,2,50),(3,5,20),(6,19,100),(2,100,200)]) == 250
assert max_weight([(0,5,3),(5,10,3)]) == 6
import random
r = random.Random(7)
big = [(s, s + r.randint(1, 50), r.randint(1, 100)) for s in (r.randint(0, 10**6) for _ in range(100000))]
assert max_weight(big) > 0
""",
         ref="""
import bisect
def max_weight(jobs):
    js = sorted(jobs, key=lambda j: j[1]); ends = [j[1] for j in js]; dp = [0]
    for i, (s, e, w) in enumerate(js):
        k = bisect.bisect_right(ends, s, 0, i)
        dp.append(max(dp[-1], dp[k] + w))
    return dp[-1]
"""),
    dict(id="json_path", tier="hard",
         prompt="Write `get_path(obj, path: str)` that walks nested dicts/lists with a path like 'a.b[2].c' or "
                "'[0].name' or 'items[-1]' (negative list indices allowed; dict keys are [A-Za-z0-9_]+). Return the "
                "value, or the sentinel string 'MISSING' when any step does not exist. An empty path returns obj.",
         tests="""
d = {'a': {'b': [0, 1, {'c': 42}]}, 'items': [1, 2, 3], 'x_1': None}
assert get_path(d, 'a.b[2].c') == 42
assert get_path(d, 'items[-1]') == 3
assert get_path(d, 'items[5]') == 'MISSING'
assert get_path(d, 'a.z') == 'MISSING'
assert get_path(d, 'x_1') is None
assert get_path([{'name': 'n'}], '[0].name') == 'n'
assert get_path(d, '') is d
assert get_path(d, 'a.b.c') == 'MISSING'
assert get_path(d, 'items.0') == 'MISSING'
""",
         ref="""
import re
def get_path(obj, path):
    cur = obj
    for key, idx in re.findall(r"([A-Za-z0-9_]+)|\\[(-?\\d+)\\]", path):
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return 'MISSING'
            cur = cur[key]
        else:
            i = int(idx)
            if not isinstance(cur, list) or not -len(cur) <= i < len(cur):
                return 'MISSING'
            cur = cur[i]
    return cur
"""),
    dict(id="knapsack", tier="hard",
         prompt="Write `knapsack(items: list[tuple[int,int]], cap: int) -> int`: items are (weight, value), each "
                "usable at most once; return the maximum value with total weight <= cap. cap <= 10000, up to 200 "
                "items.",
         tests="""
assert knapsack([], 10) == 0
assert knapsack([(1,1),(3,4),(4,5),(5,7)], 7) == 9
assert knapsack([(5,10)], 4) == 0
assert knapsack([(2,3),(2,3),(2,3)], 4) == 6
import random
r = random.Random(3)
it = [(r.randint(1, 100), r.randint(1, 100)) for _ in range(200)]
def _ref(items, cap):
    best = {0: 0}
    for w, v in items:
        for c, val in list(best.items()):
            if c + w <= cap and best.get(c + w, -1) < val + v:
                best[c + w] = val + v
    return max(best.values())
assert knapsack(it, 3000) == _ref(it, 3000)
""",
         ref="""
def knapsack(items, cap):
    dp = [0] * (cap + 1)
    for w, v in items:
        for c in range(cap, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]
"""),
    dict(id="sudoku", tier="hard",
         prompt="Write `solve_sudoku(board: list[list[int]]) -> list[list[int]]` that returns a solved copy of a "
                "9x9 sudoku (0 = blank). The puzzle has a unique solution. Must finish within a few seconds.",
         tests="""
p = [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],
     [7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]
q = [r[:] for r in p]
s = solve_sudoku(p)
assert p == q, 'input must not be modified'
assert all(sorted(r) == list(range(1, 10)) for r in s)
assert all(sorted(c) == list(range(1, 10)) for c in zip(*s))
assert all(sorted(s[i+a][j+b] for a in range(3) for b in range(3)) == list(range(1, 10)) for i in (0,3,6) for j in (0,3,6))
assert all(q[i][j] in (0, s[i][j]) for i in range(9) for j in range(9))
h = [[8,0,0,0,0,0,0,0,0],[0,0,3,6,0,0,0,0,0],[0,7,0,0,9,0,2,0,0],[0,5,0,0,0,7,0,0,0],[0,0,0,0,4,5,7,0,0],
     [0,0,0,1,0,0,0,3,0],[0,0,1,0,0,0,0,6,8],[0,0,8,5,0,0,0,1,0],[0,9,0,0,0,0,4,0,0]]
t = solve_sudoku(h)
assert all(sorted(r) == list(range(1, 10)) for r in t) and t[0][0] == 8 and t[8][1] == 9
assert all(sorted(c) == list(range(1, 10)) for c in zip(*t))
""",
         ref="""
def solve_sudoku(board):
    b = [r[:] for r in board]
    rows = [set() for _ in range(9)]; cols = [set() for _ in range(9)]; boxes = [set() for _ in range(9)]
    empty = []
    for i in range(9):
        for j in range(9):
            v = b[i][j]
            if v:
                rows[i].add(v); cols[j].add(v); boxes[i//3*3+j//3].add(v)
            else:
                empty.append((i, j))
    def go():
        if not empty:
            return True
        best = min(empty, key=lambda c: 9 - len(rows[c[0]] | cols[c[1]] | boxes[c[0]//3*3+c[1]//3]))
        i, j = best; k = i//3*3+j//3
        empty.remove(best)
        for v in range(1, 10):
            if v not in rows[i] and v not in cols[j] and v not in boxes[k]:
                b[i][j] = v; rows[i].add(v); cols[j].add(v); boxes[k].add(v)
                if go():
                    return True
                rows[i].discard(v); cols[j].discard(v); boxes[k].discard(v)
        b[i][j] = 0; empty.append(best)
        return False
    go()
    return b
"""),
    dict(id="time_map", tier="hard",
         prompt="Write a class `TimeMap` with `set(self, key: str, value: str, ts: int) -> None` and "
                "`get(self, key: str, ts: int) -> str` returning the value set for key with the largest timestamp "
                "<= ts, or '' if none. Timestamps for a key may arrive in any order; get must be O(log n) when "
                "they arrive in increasing order.",
         tests="""
m = TimeMap()
m.set('foo', 'bar', 1)
assert m.get('foo', 1) == 'bar' and m.get('foo', 3) == 'bar' and m.get('foo', 0) == ''
m.set('foo', 'bar2', 4)
assert m.get('foo', 4) == 'bar2' and m.get('foo', 5) == 'bar2' and m.get('foo', 3) == 'bar'
m.set('foo', 'mid', 2)
assert m.get('foo', 3) == 'mid'
assert m.get('nope', 9) == ''
""",
         ref="""
import bisect
class TimeMap:
    def __init__(self):
        self.d = {}
    def set(self, key, value, ts):
        ts_list, vals = self.d.setdefault(key, ([], []))
        i = bisect.bisect_right(ts_list, ts)
        ts_list.insert(i, ts); vals.insert(i, value)
    def get(self, key, ts):
        if key not in self.d:
            return ''
        ts_list, vals = self.d[key]
        i = bisect.bisect_right(ts_list, ts)
        return vals[i - 1] if i else ''
"""),
    dict(id="simplify_path", tier="hard",
         prompt="Write `simplify(path: str) -> str` that canonicalises an absolute Unix path: collapse repeated "
                "slashes, drop '.', resolve '..' (which stays at '/' at the root), no trailing slash except for "
                "the root itself. Names such as '...' are ordinary names.",
         tests="""
assert simplify('/home/') == '/home'
assert simplify('/../') == '/'
assert simplify('/home//foo/') == '/home/foo'
assert simplify('/a/./b/../../c/') == '/c'
assert simplify('/.../a/../b') == '/.../b'
assert simplify('/') == '/'
""",
         ref="""
def simplify(path):
    st = []
    for p in path.split('/'):
        if p == '..':
            if st:
                st.pop()
        elif p and p != '.':
            st.append(p)
    return '/' + '/'.join(st)
"""),
    dict(id="ledger", tier="hard",
         prompt="Write `apply_transfers(balances: dict[str,int], transfers: list[tuple[str,str,int]]) -> "
                "tuple[dict[str,int], list[int]]`. Apply transfers (src, dst, amount) in order to a copy of "
                "balances. A transfer is rejected (and skipped) when amount <= 0, src == dst, either account is "
                "unknown, or src has insufficient funds at that moment. Return the final balances and the indices "
                "of the rejected transfers. Do not mutate the input.",
         tests="""
b = {'a': 10, 'b': 0, 'c': 5}
out, rej = apply_transfers(b, [('a','b',7),('b','c',8),('a','c',3),('c','a',0),('a','a',1),('a','z',1),('b','c',7)])
assert out == {'a': 0, 'b': 0, 'c': 15}, out
assert rej == [1, 3, 4, 5], rej
assert b == {'a': 10, 'b': 0, 'c': 5}
assert apply_transfers({}, []) == ({}, [])
""",
         ref="""
def apply_transfers(balances, transfers):
    b = dict(balances); rej = []
    for i, (s, d, a) in enumerate(transfers):
        if a <= 0 or s == d or s not in b or d not in b or b[s] < a:
            rej.append(i); continue
        b[s] -= a; b[d] += a
    return b, rej
"""),
    # ---------------------------------------------------------------- fix (read code, find and fix the bug)
    dict(id="fix_bisect", tier="fix",
         prompt="This function should return the leftmost index at which x could be inserted into the sorted list "
                "a while keeping it sorted (like bisect.bisect_left), but it has bugs. Return the corrected "
                "function, same name and signature, without using the bisect module.\n\n```python\n"
                "def bisect_left(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
                "        if a[mid] <= x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n```",
         tests="""
import bisect, random
r = random.Random(5)
for _ in range(500):
    a = sorted(r.randint(0, 20) for _ in range(r.randint(0, 15)))
    x = r.randint(-1, 21)
    assert bisect_left(a, x) == bisect.bisect_left(a, x), (a, x)
""",
         ref="""
def bisect_left(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
"""),
    dict(id="fix_moving_avg", tier="fix",
         prompt="This function should return the list of averages of every contiguous window of k values (length "
                "len(xs) - k + 1, or [] when k > len(xs)), in O(n), but it has bugs. Return the corrected function, "
                "same name and signature.\n\n```python\n"
                "def moving_average(xs, k):\n    out = []\n    s = sum(xs[:k])\n    for i in range(k, len(xs)):\n"
                "        out.append(s / k)\n        s += xs[i] - xs[i - k + 1]\n    return out\n```",
         tests="""
assert moving_average([1,2,3,4,5], 2) == [1.5, 2.5, 3.5, 4.5]
assert moving_average([1,2,3], 3) == [2.0]
assert moving_average([1,2], 3) == []
assert moving_average([5], 1) == [5.0]
import random
r = random.Random(2); xs = [r.randint(0, 9) for _ in range(300)]
assert all(abs(a - b) < 1e-9 for a, b in zip(moving_average(xs, 7), [sum(xs[i:i+7]) / 7 for i in range(294)]))
assert len(moving_average(xs, 7)) == 294
""",
         ref="""
def moving_average(xs, k):
    if k > len(xs):
        return []
    out = []
    s = sum(xs[:k])
    out.append(s / k)
    for i in range(k, len(xs)):
        s += xs[i] - xs[i - k]
        out.append(s / k)
    return out
"""),
    dict(id="fix_dedupe", tier="fix",
         prompt="This function should remove duplicate records, keeping the first occurrence of each id and the "
                "original order, and must not modify its input, but it has bugs. Return the corrected function, "
                "same name and signature.\n\n```python\n"
                "def dedupe(records, seen=set()):\n    out = records\n    for r in records:\n"
                "        if r['id'] in seen:\n            out.remove(r)\n        seen.add(r['id'])\n    return out\n```",
         tests="""
a = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}, {'id': 1, 'v': 'c'}, {'id': 1, 'v': 'd'}, {'id': 3, 'v': 'e'}]
orig = [dict(r) for r in a]
assert dedupe(a) == [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}, {'id': 3, 'v': 'e'}]
assert a == orig
assert dedupe([{'id': 1, 'v': 'x'}]) == [{'id': 1, 'v': 'x'}], 'state must not leak between calls'
assert dedupe([]) == []
""",
         ref="""
def dedupe(records, seen=None):
    seen = set() if seen is None else set(seen)
    out = []
    for r in records:
        if r['id'] not in seen:
            out.append(r)
        seen.add(r['id'])
    return out
"""),
]
