import numpy as np


def load_next_frame(fp):
    pcount, _ = _read_frame_header(fp)
    return _read_frame_body(fp, pcount)


def _read_frame_header(fp):
    particle_count = int(fp.readline())
    comment = fp.readline()
    return particle_count, comment


def _read_frame_body(fp, pcount):
    positions = np.empty((pcount, 3))
    labels = np.empty((pcount,), dtype=str)
    for i in range(pcount):
        labels[i], *positions[i] = fp.readline().split()
    return positions, labels