"""Checks for the random-face filters in vcam_random.py.

The jaw measures below were taken from hand-labelled faces from
thispersondoesnotexist.com (texture ratio, skin-colour ratio of chin/jaw vs cheeks).
Run: venv/bin/python tests/random_face_check.py           (rules only, offline)
     venv/bin/python tests/random_face_check.py --live    (also one real search per filter)
"""
import os
import sys
import threading

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import vcam_random as vr  # noqa: E402

BEARDS = {  # Full or clear beards, including a dense dark one (low texture).
    "f23": (2.85, 0.71), "f24": (2.60, 0.90), "f26": (2.75, 1.13), "f30": (2.69, 1.43),
    "m7": (2.78, 0.56), "m8": (2.04, 0.61), "m17": (1.21, 0.64), "m24": (2.29, 0.74),
}
LIGHT_STUBBLE = {"f21": (1.02, 0.92), "m22": (1.37, 0.89), "m18": (1.62, 1.16), "f5": (1.68, 0.88)}
CLEAN = {
    "f2": (1.06, 0.91), "f4": (0.81, 1.00), "f8": (1.52, 0.82), "f9": (0.98, 1.38),
    "f10": (1.06, 1.33), "f12": (1.04, 0.96), "m1": (0.99, 1.01), "m2": (1.08, 0.98),
    "m3": (1.30, 2.27), "m6": (0.71, 1.06), "m9": (1.10, 0.78), "m13": (1.01, 1.53),
    "m16": (1.06, 1.22), "m23": (1.27, 0.85),
}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


hair, shaven = vr.FILTERS["Men with facial hair"], vr.FILTERS["Men clean-shaven"]
missed = [k for k, jaw in BEARDS.items() if not hair("M", jaw)]
check("facial-hair filter finds every beard", not missed, f"missed {missed}")
wrong = [k for k, jaw in {**CLEAN, **LIGHT_STUBBLE}.items() if hair("M", jaw)]
check("facial-hair filter skips clean and light stubble", not wrong, f"picked {wrong}")
wrong = [k for k, jaw in BEARDS.items() if shaven("M", jaw)]
check("clean-shaven filter never picks a beard", not wrong, f"picked {wrong}")
kept = sum(shaven("M", jaw) for jaw in CLEAN.values())
check("clean-shaven filter keeps most clean faces", kept >= len(CLEAN) - 2, f"{kept}/{len(CLEAN)}")
check("women filter rejects men and vice versa",
      vr.FILTERS["Women"]("F", None) and not vr.FILTERS["Women"]("M", None)
      and vr.FILTERS["Men"]("M", None) and not vr.FILTERS["Men"]("F", None))
check("no jaw measure means no beard decision", not hair("M", None) and not shaven("M", None))

if "--live" in sys.argv:
    import modules.globals

    modules.globals.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    for choice in vr.FILTERS:
        path, message = vr.find_face(choice, threading.Event())
        check(f"live search: {choice}", path is not None, message)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
