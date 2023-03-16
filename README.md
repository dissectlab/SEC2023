# edge_3d_journal

1. Difference Detector: task_schedule_{dataset name}.py
   - Change window size or use arbitrary window size
2. Edge Server (task execution, profiling, compute F-score): server_{dataset name}.py
   - Change root path and IP address
3. Create golden results: create_best_results_{dataset name}.py
   - With Original resolution
4. Profiling, compute difference: diff_test.py
   - With a given threshold
5. Optimization, solutions: common/opt.py
   - Use GEKKO tool
