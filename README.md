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
   
# Dataset 
1. Walk

       walk/walking_all_frame_cam1

            /walking_all_frame_cam2
       
            /walking_all_frame_cam3
       
            /walking_all_frame_cam4
       
            /walking_all_frame_cam5
            
2. Book Pickup

       book/0

            /1
       
            /2
       
            /3
       
            /4
       
3. Handshake
   
      handshake/0

            /1
       
            /2
       
            /3
       
            /4
   
# Your project path:
  book
  handshake
  walk
  task_schedule_{dataset name}.py
  server_{dataset name}.py
  ......
