
def capitalize_except(s, skip={"and", "of", "the"}, caps={"nw", "llc"}):
    words = s.split()
    def convert(i, s):
        lc = s.lower()
        if lc in caps: return s
        if i==0: return s.capitalize()
        if lc in skip: return lc
        return s.capitalize()
    return " ".join( convert(i, w) for i, w in enumerate(words) )
assert capitalize_except("THE SCHOOL OF the NW, LLC") == 'The School of the Nw, LLC'


def fixCaps(series, map={}):
    map = map.copy()
    for c in set(series):
        c_ = c.strip()
        if c_[-1].isupper():
            c_ = capitalize_except(c_)
        if c != c_:
            map[c] = c_
    for (a, b) in sorted(map.items()):
        print(f"{a} -> {b}")
    series = series.replace(map)
    return series
