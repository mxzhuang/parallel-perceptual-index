#!/usr/bin/env python3
"""Convert the official model files into plain binary files for the C++ port.

Inputs (from the official releases):
  sr-metric/model.mat                     Ma et al. random forests + linear weights
  niqe_release/modelparameters.mat        NIQE pristine MVG model

Outputs:
  models/ma_model.bin
  models/niqe_params.bin
  (optional) model_octave.mat             same Ma model with the node status decoded
                                          to int8, so the official MEX predictor runs in Octave

Note on `nodestatus`: the official training MEX stores node status as 1-byte values
inside a MATLAB char array (2 bytes per element), so the bytes are packed into the
first half of the char buffer. We decode the raw bytes and check the result against
the tree structure (a node is terminal exactly when its left daughter is 0).
"""
import argparse
import struct

import numpy as np
import scipy.io as sio


def decode_nodestatus(rf):
    ns = np.ascontiguousarray(np.array(rf.nodestatus))
    codes = ns.view(np.uint32).astype(np.uint16)  # MATLAB char code units
    nr, nt = int(rf.nrnodes), int(rf.ntree)
    raw = codes.flatten(order="F").astype("<u2").tobytes()
    status = np.frombuffer(raw, dtype=np.int8)[: nr * nt].reshape((nr, nt), order="F")
    # sanity check against tree structure
    for t in range(nt):
        n = int(rf.ndtree[t])
        if not np.array_equal(status[:n, t] == -1, rf.lDau[:n, t] == 0):
            raise RuntimeError(f"node status decoding mismatch in tree {t}")
    return status


def write_ma(model, path):
    rfs = model.rf
    with open(path, "wb") as f:
        f.write(b"PIMA")
        f.write(struct.pack("<I", 1))
        f.write(np.asarray(model.linear, dtype="<f8").reshape(4).tobytes())
        f.write(struct.pack("<I", len(rfs)))
        for rf in rfs:
            if np.any(np.asarray(rf.coef) != 0):
                raise RuntimeError("bias correction is not supported (coef != 0)")
            status = decode_nodestatus(rf)
            nt = int(rf.ntree)
            f.write(struct.pack("<I", nt))
            for t in range(nt):
                n = int(rf.ndtree[t])
                left = rf.lDau[:n, t].astype("<i2")
                right = rf.rDau[:n, t].astype("<i2")
                st = status[:n, t].astype("i1")
                var = rf.mbest[:n, t]
                if var.max() > 255 or rf.lDau[:n, t].max() > 32767:
                    raise RuntimeError("index out of range for compact format")
                var = var.astype("u1")
                term = st == -1
                value = np.where(term, rf.avnode[:n, t], rf.upper[:n, t]).astype("<f8")
                f.write(struct.pack("<I", n))
                for arr in (left, right, st, var, value):
                    f.write(arr.tobytes())


def write_niqe(params, path):
    mu = np.asarray(params["mu_prisparam"], dtype="<f8").reshape(-1)
    cov = np.asarray(params["cov_prisparam"], dtype="<f8")
    d = mu.size
    assert cov.shape == (d, d)
    with open(path, "wb") as f:
        f.write(b"PINQ")
        f.write(struct.pack("<I", d))
        f.write(mu.tobytes())
        f.write(np.ascontiguousarray(cov).tobytes())  # row-major


def write_octave_model(model, path):
    cells = np.empty((1, len(model.rf)), dtype=object)
    for i, rf in enumerate(model.rf):
        cells[0, i] = {
            "lDau": rf.lDau.astype(np.int32),
            "rDau": rf.rDau.astype(np.int32),
            "nodestatus": decode_nodestatus(rf),
            "nrnodes": float(rf.nrnodes),
            "upper": rf.upper,
            "avnode": rf.avnode,
            "mbest": rf.mbest.astype(np.int32),
            "ndtree": rf.ndtree.astype(np.int32),
            "ntree": float(rf.ntree),
            "coef": np.asarray(rf.coef, dtype=float),
        }
    sio.savemat(path, {"model": {"rf": cells, "linear": np.asarray(model.linear, dtype=float).reshape(4, 1)}},
                do_compression=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma", required=True, help="path to sr-metric/model.mat")
    ap.add_argument("--niqe", required=True, help="path to niqe_release/modelparameters.mat")
    ap.add_argument("--out", default="models")
    ap.add_argument("--octave-model", help="also write an Octave-loadable model.mat here")
    a = ap.parse_args()

    ma = sio.loadmat(a.ma, squeeze_me=True, struct_as_record=False, chars_as_strings=False)["model"]
    write_ma(ma, f"{a.out}/ma_model.bin")
    write_niqe(sio.loadmat(a.niqe), f"{a.out}/niqe_params.bin")
    if a.octave_model:
        write_octave_model(ma, a.octave_model)
    print("done")


if __name__ == "__main__":
    main()
