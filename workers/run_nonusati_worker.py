#!/usr/bin/env python3
"""
Backward-compatibility wrapper for run_nonusati_worker.py, forwarding directly
to run_algo_worker.py.
"""
from workers.run_algo_worker import main

if __name__ == "__main__":
    import seed_util; seed_util.apply_env_seed()
    main()
