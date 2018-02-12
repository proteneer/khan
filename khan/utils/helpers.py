import math
import numpy as np

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


def ed_harder_rmse(y_groups_true, ys_pred):

    assert sum([len(a) for a in y_groups_true]) == len(ys_pred)
    all_y_idx = 0

    group_rmses = []

    for yg in y_groups_true:
        min_grp_idx = np.argmin(yg)
        min_yts_idx = min_grp_idx + all_y_idx
        correction = yg[min_grp_idx] - ys_pred[min_yts_idx]
        group_l2s = []
        for y_idx, yt in enumerate(yg):
            yp = ys_pred[all_y_idx] + correction
            group_l2s.append((yp-yt)*(yp-yt))
            all_y_idx += 1  

        group_rmse = np.sqrt(sum(group_l2s) / len(group_l2s))
        group_rmses.append(group_rmse)

    ed_rmse = sum(group_rmses) / len(group_rmses)
    return ed_rmse


if __name__ == "__main__":


    #                             true values          pred values
    res = ed_harder_rmse([[2, 1, 3], [8, 9, 2, 3]], [3,0,2, 7,8,4,1])

    # expected                                              5,6,2,-1

    expected = (math.sqrt((2*2+0*0+0*0)/3) + math.sqrt(((8-5)**2+(9-6)**2+(2-2)**2+(3-(-1))**2)/4))/2
    print(expected, res)
