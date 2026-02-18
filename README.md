# Workpiece Coordinate System Measurement for a Robotic Timber Joinery Workflow
## Description

This repository presents an integrated design-to-fabrication workflow combining parametric modeling, industrial robotics, and 3D vision sensing for the robotic fabrication of timber joinery.  
The included diagram illustrates the complete workflow implemented in this research.

### Main Scripts

- **A_camera_robot_calibration.py**  
  Performs eye-in-hand calibration of a Zivid 3D camera mounted on a KUKA industrial robot.

- **B_pointcloud_capture.py**  
  Moves the robot through predefined poses and captures the point cloud of the timber workpiece.

- **C_compute_wcs_pointcloud.py**  
  Computes the KUKA robot BASE coordinate system and estimates the timber workpiece cross-section for tenon machining.

### Additional Files

The remaining files correspond to implementation outputs and datasets associated with the experimental workflow described in the publication:

https://www.mdpi.com/2075-5309/15/15/2712

![605](https://github.com/user-attachments/assets/c3bea1e0-1d92-43a2-a118-53234f8e92f0)

## Citation
If you use this repository, please cite:
Quitral-Zapata, F., García-Alvarado, R., Martínez-Rocamora, A., & González-Böhme, L. F. (2025). Workpiece Coordinate System Measurement for a Robotic Timber Joinery Workflow. Buildings, 15(15), 2712. https://doi.org/10.3390/buildings15152712
