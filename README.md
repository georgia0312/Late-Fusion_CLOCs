## Late Fusion - CLOCs

### Design a Python package for model-agnostic Late fusion with CLOCs 

![image](https://github.com/user-attachments/assets/4f9aef23-b2ed-4394-8d40-3e17c2fe5e0e)

1. Save 3D detection and 2D detection candidates before NMS as text files
2. Generate Input Tensor T = (IoU, 3D score, 2D score, distance_to_lidar)
3. Train Fusion Network that receives tensor T as input
4. Update 3D confident score with the output of Fusion Network
5. Proceed NMS and get the final 3D detection results
