from itertools import izip
from itertools import tee
from numpy import log2
from collections import defaultdict

def entropy(ps, base=2):
    """Calculate Shannon entropy given a probability distribution """
    probs = [p * log2(p)/log2(base) 
             if p > 0 else 0 for p in ps]
    return -sum(probs)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def slide(seq, window_size):
    """Sliding window over given sequence, yielding an generator """ 
    chunks = len(seq) - window_size
    for i in xrange(0, chunks):
        yield seq[i:i+window_size]

def h_n_cond(seq, ngram, l):
    """Calculates local conditional uncertainty of the next state
    after the measured trajectory ngram. The last parameter is ideally
    equal to len(ngram) in order to result in a measure between 0 and 1 """
    counts = defaultdict(int)
    for e,nxte in pairwise(seq):
        if e == ngram:
            counts[str(nxte)] += 1.
    probdist = counts.values() / sum(counts.values()) 
    return entropy(probdist, l)

def local_uncertainty(seq, length, l=2):
    """ lambda should be equal to the dictionary size of seq """
    out = []
    for s in slide(seq, length):
        out.append(h_n_cond(slide(seq, length), s, l))
    return out
        
