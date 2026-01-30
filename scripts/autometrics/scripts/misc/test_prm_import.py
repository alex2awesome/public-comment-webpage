#!/usr/bin/env python3
import sys

try:
    from autometrics.metrics.reference_free.PRMRewardModel import MathProcessRewardModel
except Exception as e:
    print(f"IMPORT_FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

def main():
    try:
        m = MathProcessRewardModel(persistent=False)
        print("OK: imported and initialized MathProcessRewardModel")
    except Exception as e:
        print(f"INIT_FAIL: {type(e).__name__}: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()


