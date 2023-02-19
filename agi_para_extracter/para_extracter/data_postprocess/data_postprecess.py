import numpy as np
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    pass


def find_geara(seg_motor):
    gearaup = []
    for point in seg_motor:
        if point[3] == 7: gearaup.append(point[0:4])
    gearadown = []
    for point in seg_motor:
        if point[3] == 8: gearadown.append(point[0:4])

    positionaup = np.squeeze(np.mean(gearaup[0:3], axis=1))
    positionadown = np.squeeze(np.mean(gearadown[0:3], axis=1))

    return np.vstack((gearaup, gearadown)), positionaup, positionadown


def find_gearb(seg_motor):
    gearb = []
    for point in seg_motor:
        if point[3] == 9:
            gearb.append(point[0:4])

    positionb = np.squeeze(np.mean(gearb[0:3], axis=1))

    return np.array(gearb), positionb


def find_bolts(seg_motor, eps, min_points):
    bolts = []
    for point in seg_motor:
        if point[3] == 6: bolts.append(point[0:3])
    bolts = np.asarray(bolts)
    model = DBSCAN(eps=eps, min_samples=min_points)
    yhat = model.fit_predict(bolts)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    positions = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 50 or i == -1 else clusters_new.append(i)
    flag = 0
    bolts__ = 1
    for clu in clusters_new:
        row_ix = np.where(yhat == clu)
        if flag == 0:
            bolts__ = np.squeeze(np.array(bolts[row_ix, :3]))
            flag = 1
        else:
            inter = np.squeeze(np.array(bolts[row_ix, :3]))
            bolts__ = np.concatenate((bolts__, inter), axis=0)
        np.set_printoptions(precision=2)
        position = np.squeeze(np.mean(bolts[row_ix, :3], axis=1))
        positions.append(position)
    positions = np.array(positions)
    indexs = np.argsort(positions[:, 1])
    positions = positions[indexs, :]

    return positions, len(clusters_new), bolts__
