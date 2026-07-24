"""
Smoke test for Algotom + Astra GPU stack.
Does not require a Bruker scan — only verifies imports and a tiny CUDA FBP.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("Importing numpy...")
    import numpy as np

    print("Importing algotom...")
    import algotom.io.loadersaver  # noqa: F401
    import algotom.prep.removal as remo  # noqa: F401
    import algotom.rec.reconstruction as rec

    print("Importing astra...")
    import astra

    print(f"  astra version: {getattr(astra, '__version__', 'unknown')}")

    print("Running tiny FBP_CUDA test...")
    n_angles, n_det = 64, 64
    sino = np.ones((n_angles, n_det), dtype=np.float32)
    center = (n_det - 1) / 2.0
    try:
        img = rec.astra_reconstruction(
            sino,
            center,
            method="FBP_CUDA",
            apply_log=False,
        )
    except Exception as exc:
        print("FBP_CUDA failed:", exc)
        print("CPU fallback check (FBP)...")
        try:
            img = rec.astra_reconstruction(
                sino,
                center,
                method="FBP",
                apply_log=False,
            )
        except Exception as exc2:
            print("CPU FBP also failed:", exc2)
            print("SMOKE TEST FAILED")
            return 1
        print(f"  recon shape: {np.asarray(img).shape}")
        print("CPU FBP works, but CUDA path failed — check NVIDIA drivers / astra CUDA build.")
        print("SMOKE TEST PARTIAL (CPU only)")
        return 2

    print(f"  recon shape: {np.asarray(img).shape}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
