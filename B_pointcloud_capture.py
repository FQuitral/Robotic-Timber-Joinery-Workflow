import yaml
import numpy as np
import zivid
# Use standard imports
import robolink 
import robodk  
import datetime
import time 
import os 
import traceback # Import traceback for detailed error printing

# --- Configuration ---
# Path to the calibration file containing the Flange-to-Camera transform
CALIBRATION_FILE = "calibration_results.yaml" 
POSE_INFO_YAML_FILE = "pose_capture_info.yaml" 

# Output file names
ORIGINAL_CLOUD_ZDF_FILE = "original_cloud.zdf"
TRANSFORMED_CLOUD_PLY_FILE = "transformed_cloud.ply" # For the transformed data

# RoboDK Connection (Set to False to run purely in simulation for testing)
CONNECT_TO_ROBODK = True 

# Zivid Camera Settings (Optional: customize as needed)
def get_capture_settings():
    """Returns a Zivid Settings object for capture."""
    settings = zivid.Settings()
    settings.acquisitions.append(zivid.Settings.Acquisition())
    # Example settings - Adjust based on your needs
    settings.acquisitions[0].aperture = 3.00 
    settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=10000) # 10ms
    settings.acquisitions[0].brightness = 1.80
    return settings

# Helper function to write PLY file (simple version)
def save_ply(filename, xyz, rgba=None):
    """Saves point cloud data (XYZ and optional RGBA) to an ASCII PLY file."""
    num_points = xyz.shape[0]
    if num_points == 0:
        print(f"Warning: Attempted to save an empty point cloud to {filename}. Skipping.")
        return
        
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    except OSError as e:
        print(f"Error creating directory for {filename}: {e}")
        return # Cannot save if directory cannot be created

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]
    has_color = rgba is not None and rgba.shape[0] == num_points
    if has_color:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar alpha",
        ])
    header.append("end_header")
    
    try:
        with open(filename, 'w') as f:
            for line in header:
                f.write(line + '\n')
            
            for i in range(num_points):
                point_data_str = [f"{coord:.6f}" for coord in xyz[i]] # Format coordinates
                if has_color:
                    # Format color as integers
                    point_data_str.extend(map(str, map(int, rgba[i]))) 
                f.write(" ".join(point_data_str) + '\n')
        print(f"Successfully saved point cloud data to {filename}")
    except IOError as e:
        print(f"Error writing PLY file {filename}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred saving PLY {filename}: {e}")
         traceback.print_exc() # Print details for unexpected errors

# --- Main Script ---
def main():
    # Initialize variables
    rdk = None
    robot = None
    app = None
    camera = None
    T_Base_Flange_np = None 
    T_Flange_Camera_np = None
    current_joints_list = None 

    try:
        # --- 1. Connect to RoboDK, get Robot, and CONFIGURE FOR REAL ROBOT ---
        if CONNECT_TO_ROBODK: 
            print("Connecting to RoboDK...")
            rdk = robolink.Robolink() 

            robot = rdk.ItemUserPick('Select the robot', robolink.ITEM_TYPE_ROBOT) 
            if not robot.Valid():
                raise Exception("Robot selection failed, user cancelled, or RoboDK connection issue.")
            print(f"Using robot: {robot.Name()}")

            print("Attempting to connect and configure robot for run mode...")

            if rdk.RunMode() != robolink.RUNMODE_SIMULATE:
                 print("WARNING: RoboDK RunMode is not SIMULATE. Skipping real robot connection setup.")
                 raise Exception("RoboDK is not in Simulation mode. Cannot guarantee online run mode setup.")
            else:
                robot_ip = '192.168.100.101' 
                robot_port = 7000           
                print(f"Setting connection parameters - IP: {robot_ip}, Port: {robot_port}")
                robot.setConnectionParams(robot_ip, robot_port, '/', 'anonymous', '') 

                print("Connecting to robot driver...")
                success = robot.Connect() 
                status, status_msg = robot.ConnectedState() 

                if status != robolink.ROBOTCOM_READY: 
                    print(f"Connection Status: {status_msg} (Code: {status})")
                    raise Exception("Failed to connect to robot driver: " + status_msg)
                print("Successfully connected to robot driver.")

                print("Setting RunMode to RUN_ROBOT...")
                rdk.setRunMode(robolink.RUNMODE_RUN_ROBOT) 
                print("Run mode successfully set to RUNMODE_RUN_ROBOT.")

            # --- Define Target Capture Pose (Joint Angles) ---
            target_joints_capture = [1.02, -116.47, 102.67, -13.34, -58.27, 21.33, 500]
            print(f"Defined target capture joints: {target_joints_capture}")

            # --- Move Robot to Capture Pose ---
            print("Moving robot to the defined capture pose...")
            robot.MoveJ(target_joints_capture, blocking=True)
            print("Robot reached capture pose.")

            # --- Wait Before Capture ---
            wait_time_seconds = 3
            print(f"Waiting for {wait_time_seconds} seconds before capture...")
            time.sleep(wait_time_seconds)
            print("Wait finished.")
            
            # --- Set Reference Frame to WORLD and Tool to Flange --- 
            print("Setting Reference Frame to 'World' and Tool to Flange Origin")
            world_frame_item = None
            target_frame_name = "World" 

            try:
                # --- Get the frame named "World" directly ---
                print(f"Attempting to get reference frame named '{target_frame_name}'...")
                world_frame_item = rdk.Item(target_frame_name, robolink.ITEM_TYPE_FRAME)

                if not world_frame_item.Valid():
                    raise Exception(f"Could not find a FRAME item named '{target_frame_name}' in the RoboDK station. Please ensure it exists and is of type Frame.")

                robot.setFrame(world_frame_item)
                world_frame_name = world_frame_item.Name() 

                identity_pose_for_flange = robodk.Pose(0,0,0,0,0,0) 
                robot.setPoseTool(identity_pose_for_flange)

                print(f"Active Tool Frame set to: Flange (via identity TCP)")
                print(f"Active Reference Frame successfully set to: '{world_frame_name}'")

            except robolink.ERRobolink as comm_error:
                print(f"Error: Robolink communication error during Item lookup, setTool, or setFrame: {comm_error}")
                raise 
            except Exception as setup_error:
                print(f"ERROR: Could not set World Frame or Flange Tool. Error: {setup_error}")
                if f"Could not find a FRAME item named '{target_frame_name}'" in str(setup_error):
                     print("----> Please check your RoboDK station setup and ensure the frame exists.")
                raise Exception("Failed to set World/Flange frames.") from setup_error 

            T_World_Flange_rdk = robot.Pose() 
            
            T_World_Flange_np_raw = np.array(T_World_Flange_rdk) 
            T_World_Flange_np = T_World_Flange_np_raw.T 
            
            print("World -> Flange Matrix (RAW from RoboDK conversion):") 
            print(T_World_Flange_np_raw)
            print("World -> Flange Matrix (Transposed - Correct Format):")
            print(T_World_Flange_np) 
            
            if not np.allclose(T_World_Flange_np[3, :], [0, 0, 0, 1]):
                print("\n! ERROR: The retrieved World->Flange matrix has an incorrect last row! !\n")
                raise Exception("Retrieved an invalid transformation matrix for World->Flange.") 
            else:
                print("World->Flange matrix appears valid.")
                
            # --- Print Joint Angles ---
            current_joints = robot.Joints()
            current_joints_list = current_joints.tolist() 
            print(f"Robot Joint Angles (at capture pose): {[f'{j:.6f}' for j in current_joints_list]}") 

        else: 
            print("Skipping RoboDK connection...")
            return  

        # --- 3. Load Flange-to-Camera Calibration Matrix ---
        if not os.path.exists(CALIBRATION_FILE):
            raise FileNotFoundError(f"Calibration file not found at {CALIBRATION_FILE}")
            
        print(f"Loading Flange -> Camera transformation from {CALIBRATION_FILE}...")
        with open(CALIBRATION_FILE, 'r') as f: calib_data = yaml.safe_load(f)
        
        # Extract matrix
        if "eye_in_hand" in calib_data and "transform_matrix" in calib_data["eye_in_hand"]: 
            matrix_list = calib_data["eye_in_hand"]["transform_matrix"] 
            T_Flange_Camera_np = np.array(matrix_list)
            if T_Flange_Camera_np.shape != (4, 4): raise ValueError("Loaded Flange->Camera matrix is not 4x4.") 
            print("Flange -> Camera Matrix loaded successfully.")
        else: raise KeyError("Could not find 'eye_in_hand'/'transform_matrix' keys in JSON.") 
            
        # --- 4. Connect to Zivid Camera ---
        print("Connecting to Zivid camera...")
        app = zivid.Application()
        camera = app.connect_camera()
        print(f"Connected to camera: {camera.info.serial_number}")

        # --- 5. Capture Frame ---
        print("Capturing frame...")
        settings = get_capture_settings()
        frame = camera.capture(settings)
        point_cloud = frame.point_cloud() 
        print("Capture successful.")
            
        # --- 6. Save Original Point Cloud as ZDF ---
        try:
            print(f"Saving original point cloud to {ORIGINAL_CLOUD_ZDF_FILE}...")
            os.makedirs(os.path.dirname(ORIGINAL_CLOUD_ZDF_FILE) or '.', exist_ok=True) 
        
            frame.save(ORIGINAL_CLOUD_ZDF_FILE) 

            print(f"Successfully saved original cloud to {ORIGINAL_CLOUD_ZDF_FILE}")
        except Exception as e:
            print(f"Error saving original point cloud as ZDF: {e}")

        # --- 7. Calculate World-to-Camera Transformation --- 
        print("Calculating World -> Camera transformation matrix...")
        if T_World_Flange_np is None or T_Flange_Camera_np is None: 
             raise ValueError("Cannot calculate World->Camera transform due to missing input matrices.")
        
        # Calculate World->Camera = World->Flange * Flange->Camera
        T_World_Camera_np = T_World_Flange_np @ T_Flange_Camera_np 
        
        print("World -> Camera Matrix calculated:")
        print(T_World_Camera_np)

        # --- 7.5 Save Pose Information to YAML ---
        print("Preparing data for YAML output...")
        data_to_save = {}

        # Add World->Camera matrix if available
        if T_World_Camera_np is not None:
            try:
                data_to_save['world_to_camera_matrix'] = T_World_Camera_np.tolist()
                print(" -> Included World->Camera matrix.")
            except Exception as e:
                print(f"Warning: Could not convert World->Camera matrix to list: {e}")
        else:
            print(" -> World->Camera matrix not available.")

        # Add Robot Joints if available 
        if current_joints_list is not None:
            data_to_save['robot_joints_at_capture'] = current_joints_list
            print(" -> Included Robot Joints at capture.")
        else:
            print(" -> Robot Joints not available (maybe didn't connect to RoboDK?).")

        if data_to_save:
            print(f"Saving pose information to {POSE_INFO_YAML_FILE}...")
            try:
                os.makedirs(os.path.dirname(POSE_INFO_YAML_FILE) or '.', exist_ok=True)
                with open(POSE_INFO_YAML_FILE, 'w') as f:
                    yaml.dump(data_to_save, f, default_flow_style=False, indent=4)
                print(f"Successfully saved pose info to {POSE_INFO_YAML_FILE}")
            except TypeError as e:
                print(f"ERROR: Could not serialize pose data to YAML. Check data types (especially numpy arrays needing .tolist()). Error: {e}")
            except IOError as e:
                print(f"ERROR: Could not write pose info file {POSE_INFO_YAML_FILE}. Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while saving pose info YAML: {e}")
        else:
            print("No pose information available to save to YAML.")

        # --- 8. Transform Point Cloud (Using World->Camera) --- 
        print("Transforming point cloud coordinates to World Frame...")
        xyz_transformed = None; rgba_valid = None 
        
        xyz_original = point_cloud.copy_data("xyz") 
        rgba_original = point_cloud.copy_data("rgba") 
        mask_valid = ~np.isnan(xyz_original[:, :, 0]) 
        xyz_valid = xyz_original[mask_valid]; rgba_valid = rgba_original[mask_valid] 
        num_valid_points = xyz_valid.shape[0]

        if num_valid_points == 0: 
             print("Warning: No valid points found. Skipping transformation.")
        else:
            print(f"Found {num_valid_points} valid points to transform.")
            ones_column = np.ones((num_valid_points, 1))
            xyz_homogeneous = np.hstack((xyz_valid, ones_column)) 

            xyz_transformed_homogeneous = (T_World_Camera_np @ xyz_homogeneous.T).T 
            
            w_coords = xyz_transformed_homogeneous[:, 3]
            if np.any(np.abs(w_coords) < 1e-9): 
                 print("Warning: Clamping near-zero w coords."); w_coords[np.abs(w_coords) < 1e-9] = 1e-9 
            xyz_transformed = xyz_transformed_homogeneous[:, :3] / w_coords[:, None] 
            print("Point cloud transformation to World Frame complete.")

            # --- 9. Save Transformed Point Cloud as PLY ---
            if xyz_transformed is not None and rgba_valid is not None:
                print(f"Saving transformed point cloud (World Coords) to {TRANSFORMED_CLOUD_PLY_FILE}...")
                save_ply(TRANSFORMED_CLOUD_PLY_FILE, xyz_transformed, rgba_valid)
            else: 
                print("Skipping saving of transformed cloud.")

    # --- General Exception Handling ---
    except FileNotFoundError as e:
        print(f"\n--- ERROR: Required file not found ---")
        print(e)
    except KeyError as e:
        print(f"\n--- ERROR: Missing key in YAML data ---")
        print(f"Could not find key: {e}")
    except ValueError as e:
         print(f"\n--- ERROR: Invalid value or data format ---")
         print(e)
    except yaml.YAMLError as e:
        print(f"\n--- ERROR: Could not parse YAML file ---")
        print(f"YAML Error: {e}")     
    except Exception as e: 
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        traceback.print_exc() 

    # --- Cleanup ---
    finally:
        print("\n--- Cleaning up resources ---")
        if camera: 
            print("Disconnecting camera...")
            del camera
        else:
            print("Camera was not connected or already cleaned up.")
            
        if app: 
            print("Releasing Zivid application...")
            del app
        else:
             print("Zivid application was not created or already cleaned up.")
             
        print("Script finished.")

# --- Run Main ---
if __name__ == "__main__":
    main()