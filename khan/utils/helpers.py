
# invert a permutation
def inv(p):
    inverse = [0] * len(p)
    for i, p in enumerate(p):
        inverse[p] = i
    return inverse

def compute_offsets(all_Xs):
    offsets = []
    last_idx = 0
    for x in all_Xs:
        new_idx = last_idx + x.shape[0]
        offsets.append((last_idx, new_idx))
        last_idx = new_idx
    return offsets