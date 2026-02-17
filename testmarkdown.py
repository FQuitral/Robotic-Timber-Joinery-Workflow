"""
This example shows how to perform Hand-Eye calibration with ROBODK,
automatically iterating through predefined poses and performing both
Eye-to-Hand and Eye-in-Hand calibrations.

MODIFIED: Saves Eye-in-Hand calibration results (transform matrix and residuals)
          to a JSON file, WITHOUT modifying the original Poses or pose handling logic.
"""

from robolink import * # API to communicate with RoboDK
from robodk import * # robodk robotics toolbox
import time                 # For pauses
import datetime             # For Zivid exposure time
import numpy as np          # For array manipulation
import zivid                # For Zivid camera and calibration API
#import json                 # For saving results to JSON <--- ADDED IMPORT
import yaml 

# Any interaction with RoboDK must be done through RDK:
RDK = Robolink()

# Select a robot (popup is displayed if more than one robot is available)
robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')

# --- Robot Connection Configuration ---
RUN_ON_ROBOT = True # Set to False to run purely in simulation

# Important: by default, the run mode is RUNMODE_SIMULATE
# If the program is generated offline manually the runmode will be RUNMODE_MAKE_ROBOTPROG,
# Therefore, we should not run the program on the robot
if RDK.RunMode() != RUNMODE_SIMULATE:
    RUN_ON_ROBOT = False

if RUN_ON_ROBOT:
    # Update connection parameters if required:
    # Format: robot.setConnectionParams('IP_ADDRESS', PORT, '/REMOTE_PATH', 'FTP_USER', 'FTP_PASS')
    robot.setConnectionParams('192.168.100.101', 7000, '/', 'anonymous', '') # MODIFY IP IF NEEDED

    # Connect to the robot using default IP
    print("Connecting to robot...")
    success = robot.Connect() # Try to connect once
    # success = robot.ConnectSafe() # Try to connect multiple times
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        # Stop if the connection did not succeed
        print(status_msg)
        raise Exception("Failed to connect to robot: " + status_msg)
    print("Connected to robot.")

    # This will set to run the API programs on the robot and the simulator (online programming)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)
    # Note: This is often set automatically when we Connect() to the robot through the API
    print("Run mode set to RUNMODE_RUN_ROBOT.")

# Get the current joint position of the robot (updates simulator)
joints_ref = robot.Joints()

# Optional: Set robot speed/acceleration/rounding if needed
# robot.setSpeedJoints(20) # Sets the joint speed of a robot in deg/s
# robot.setAccelerationJoints(50) # Sets the joint acceleration in deg/s2
# robot.setRounding(1) # Rounding accuracy in mm for Cartesian moves (if used)
# robot.setSpeed(250) # Set linear speed in mm/s for Cartesian moves (if used)
# robot.setAcceleration(1000) # Linear acceleration in mm/s2 (if used)

print("Moving robot to starting position...")
# Use a known safe starting joint configuration if desired
try:
    # Asegurar que el TCP es el flange y la referencia es la base
    robot.setTool(Pose(0,0,0,0,0,0))
    robot.setFrame(robot.Parent()) # robot.Parent() suele ser la base
    print("Tool set to flange, Frame set to robot base.")
    robot.MoveJ([-90,-120,120,0,0,0]) # Example home position
except RobolinkError as e:
    print(f"Could not move robot to home position: {e}")
    # Decide if you want to continue or raise an exception here
    # raise Exception("Failed to move robot to home")

# --- Zivid Camera Functions ---

def _acquire_checkerboard_frame(camera):
    """Acquire checkerboard frame with specific settings."""
    print("Configuring Zivid settings for checkerboard capture...")
    settings = zivid.Settings()
    settings.acquisitions.append(zivid.Settings.Acquisition())
    # Adjust these settings based on your lighting and checkerboard
    settings.acquisitions[0].aperture = 6   # Example aperture
    settings.acquisitions[0].brightness = 1.8 # Example brightness
    settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=20000) # 20ms
    settings.processing.filters.smoothing.gaussian.enabled = True
    # settings.processing.filters.smoothing.gaussian.sigma = 1.5
    # Add other filters if needed (e.g., outlier removal)
    # settings.processing.filters.outlier.removal.enabled = True
    # settings.processing.filters.outlier.removal.threshold = 5.0

    print("Capturing checkerboard image with Zivid camera...")
    try:
        frame = camera.capture(settings)
        print("Capture successful.")
        return frame
    except Exception as e:
        print(f"Error capturing frame: {e}")
        raise # Re-raise the exception to be handled in _main

# --- Robot Pose Function ---
# THIS FUNCTION IS UNCHANGED FROM ORIGINAL
def _enter_robot_pose(robot, pose_joints):
    """Moves the robot to the given joint pose and returns the Cartesian pose for Zivid."""
    print(f"Moving robot to joints: {list(pose_joints)}")
    try:
        robot.MoveJ(list(pose_joints)) # Use the argument directly
    except RobolinkError as e:
        print(f"RoboDK Error during MoveJ: {e}")
        raise Exception(f"Failed to move robot to joints {list(pose_joints)}")

    print("Waiting for robot to stabilize...")
    time.sleep(3) # Adjust sleep time if necessary

    print("Obtaining Cartesian pose from robot...")
    try:
        pose_rdk = robot.Pose() # Gets the tool flange pose relative to the robot base
    except RobolinkError as e:
        print(f"RoboDK Error getting pose: {e}")
        raise Exception("Failed to get robot pose after moving.")

    # Convert the RoboDK pose (Mat object) to a 4x4 NumPy array
    pose_matrix_np = np.array(pose_rdk)

    # Important: Verify the matrix format expected by zivid.calibration.Pose.
    # RoboDK's pose_rdk is typically [[R, T], [0, 0, 0, 1]], where R is 3x3 rotation, T is 3x1 translation.
    # np.array(pose_rdk) should produce the correct 4x4 matrix.
    # If Zivid expects a different format (e.g., transposed), adjust accordingly.
    data = np.transpose(pose_matrix_np) # Uncomment ONLY if Zivid requires transposed matrix

    #data = pose_matrix_np # Assuming direct conversion is correct

    try:
        robot_pose_zivid = zivid.calibration.Pose(data)
        print(f"Robot Cartesian pose obtained (Zivid format):\n{robot_pose_zivid.to_matrix()}")
        return robot_pose_zivid
    except Exception as e:
        print(f"Error creating Zivid Pose object: {e}")
        raise # Re-raise exception

# --- Main Execution ---

def _main():
    app = zivid.Application()

    print("Connecting to Zivid camera...")
    try:
        camera = app.connect_camera()
        print(f"Connected to camera: {camera.info.serial_number}")
    except Exception as e:
        print(f"Error connecting to Zivid camera: {e}")
        return # Cannot continue without camera

    # Define the sequence of robot joint poses for calibration
    # IMPORTANT: These poses must be valid for YOUR robot and setup.
    # They should provide diverse views of the calibration target.
    # Ensure these are 6-joint values for a typical 6-axis robot.
    # POSES ARRAY UNCHANGED FROM ORIGINAL
    Poses = np.array([
        [-53.99, -83.89, 84.5, 120.67, -62.57, 86.14, 500], [-55.12, -78.65, 88.59, 98.56, -56.21, 112.19, 500],
        [-57.9, -70.55, 88.2, 73.22, -57.52, 139.16, 500], [-61.51, -61.53, 83.22, 53.4, -64.35, 157.16, 500],
        [-66.83, -65.73, 94.11, 36.82, -62.18, 161.18, 500], [-63.19, -77.38, 101.45, 55.28, -49.41, 144.8, 500],
        [-60.09, -88.06, 102.25, 89.12, -41.42, 110.35, 500], [-58.78, -93.92, 96.33, 123.29, -47.98, 73.99, 500],
        [-68.29, -102.1, 104.22, 139.73, -31.18, 45.57, 500], [-69.93, -96.11, 111.64, 73.74, -20.68, 110.56, 500],
        [-72.42, -83.28, 110.7, 27.57, -40.14, 155.43, 500], [-74.73, -69.44, 101.75, 16.07, -59.37, 165.01, 500],
        [-83.65, -71.92, 103.8, -6.46, -55.8, 169.67, 500], [-83.58, -85.89, 112.24, -10.23, -34.25, 175.24, 500],
        [-82.66, -98.62, 112.93, -48.96, -6.47, 217.7, 500], [-80.91, -104.56, 105.72, -172.88, -24.86, 346.19, 500],
        [-91.51, -101.29, 101.83, -140.51, -35.68, 303.5, 500], [-92.81, -95.58, 107.59, -94.06, -24.7, 250.04, 500],
        [-92.51, -84.89, 107.31, -45.2, -35.61, 195.81, 500], [-91.1, -72.79, 100.88, -27.37, -52.81, 175.9, 500]
        # Add more poses if needed for better calibration results
    ])
    num_poses = len(Poses)
    print(f"Defined {num_poses} poses for calibration.")

    hand_eye_input = [] # List to store valid (robot_pose, detection_result) pairs

    # Automatically iterate through all defined poses
    for index, current_pose_joints in enumerate(Poses):
        print(f"\n--- Processing Pose {index + 1} de {num_poses} ---")

        try:
            # 1. Move robot and get its Cartesian pose
            robot_pose = _enter_robot_pose(robot, current_pose_joints)

            # 2. Acquire frame from Zivid camera
            frame = _acquire_checkerboard_frame(camera)

            # 3. Detect checkerboard (with retries)
            max_attempts = 3
            attempt = 0
            detection_result = None
            print("Detecting checkerboard in point cloud...")
            while attempt < max_attempts and not detection_result:
                attempt += 1
                print(f"Attempting detection {attempt}/{max_attempts}...")
                # Perform detection using the point cloud from the frame
                current_detection = zivid.calibration.detect_feature_points(frame.point_cloud())

                if current_detection and current_detection.valid():
                    print("Checkerboard detection SUCCESSFUL.")
                    detection_result = current_detection # Store the valid result
                    break # Exit retry loop
                else:
                    # Detection failed or result is invalid
                    if attempt < max_attempts:
                        print(f"Detection FAILED (attempt {attempt}). Retrying...")
                        time.sleep(0.5) # Small pause before next attempt
                    else:
                        print(f"Detection FAILED after {max_attempts} attempts for pose {index + 1}.")

            # 4. Add valid pair to input list
            if detection_result:
                hand_eye_input.append(zivid.calibration.HandEyeInput(robot_pose, detection_result))
                print(f"Pose {index + 1} data added successfully.")
            else:
                 print(f"Skipping pose {index + 1} due to failed checkerboard detection.")

        except KeyboardInterrupt:
             print("\n*** User interrupted the process. Exiting loop. ***")
             break # Allow user to stop the process
        except Exception as ex:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR processing pose {index + 1}: {ex}")
            print(f"Attempting to continue with the next pose...")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Continue to the next iteration of the loop


    # --- Perform Calibrations ---
    num_valid_poses = len(hand_eye_input)
    print(f"\nFinished collecting data. Acquired {num_valid_poses} valid pose(s).")

    # --->>> ADDED: Dictionary to store all calibration results before saving to JSON
    calibration_data_to_save = {}

    # Check if enough data was collected (Zivid typically needs at least 5-10)
    min_required_poses = 5
    if num_valid_poses < min_required_poses:
           print(f"\nInsufficient valid data ({num_valid_poses}) collected. Need at least {min_required_poses} for reliable calibration.")
           print("Calibration aborted. No results will be saved.")
           return # Exit _main

    print("\nProceeding with Hand-Eye Calibration...")

    # --- Performing Eye-to-Hand (eth) Calibration ---
    # SECTION UNCHANGED FROM ORIGINAL (Remains Commented)
    # print("\n--- Performing Eye-to-Hand (eth) Calibration ---")
    # try:
    #     calibration_result_eth = zivid.calibration.calibrate_eye_to_hand(hand_eye_input)
    #     # ... (original printing logic here) ...
    # except Exception as e:
    #     # ... (original error handling here) ...


    # --- Performing Eye-in-Hand (eih) Calibration ---
    print("\n--- Performing Eye-in-Hand (eih) Calibration ---")
    # --->>> ADDED: Dictionary to hold EIH results for JSON saving
    eih_results = {}
    try:
        calibration_result_eih = zivid.calibration.calibrate_eye_in_hand(hand_eye_input)
        # --->>> ADDED: Mark that the calibration function was called
        eih_results['performed'] = True

        if calibration_result_eih and calibration_result_eih.valid():
            print("Eye-in-Hand Calibration SUCCESSFUL")
            # --->>> ADDED: Mark result as valid
            eih_results['valid'] = True
            print("EIH Result (Transform: Robot Flange -> Camera Optical Frame):")
            try:
                # --->>> MODIFIED: Get transform and STORE it as list for JSON
                transform_result = calibration_result_eih.transform()
                pose_matrix_eih_np = np.array(transform_result)
                pose_matrix_eih_list = pose_matrix_eih_np.tolist() # Convert to list
                eih_results['transform_matrix'] = pose_matrix_eih_list # Store in dict

                # Keep original print statements for console feedback
                print("--- Matriz 4x4 para Grasshopper (formato CSV, 8 decimales) ---")
                matrix_str_gh = "\n".join([",".join(map(lambda x: f"{x:.8f}", row)) for row in pose_matrix_eih_np]) # Use numpy array here for formatting
                print(matrix_str_gh)
                print("--- FIN NUEVO CÓDIGO ---") # Original comment, kept for reference
            except AttributeError as e:
                print(f"      ERROR accessing transform attribute: {e}")
                # --->>> ADDED: Store None if transform cannot be accessed
                eih_results['transform_matrix'] = None
                eih_results['valid'] = False # Consider invalid if transform fails

            print("EIH Residuals (average error per point in mm):")
            # --->>> MODIFIED: Store residuals in a list for JSON
            residuals_list = []
            try:
                for idx, res in enumerate(calibration_result_eih.residuals()):
                    # Keep original print statement
                    print(f"  Pose Input Index {idx}: Rotational Residual {res.rotation():.4f} deg, Translational Residual {res.translation():.4f} mm")
                    # --->>> ADDED: Append residual data to list
                    residuals_list.append({
                        'pose_index': idx,
                        'rotation_deg': res.rotation(),
                        'translation_mm': res.translation()
                    })
                # --->>> ADDED: Store the complete list in the dictionary
                eih_results['residuals'] = residuals_list
            except Exception as res_e:
                print(f"      ERROR accessing/printing residuals: {res_e}")
                # --->>> ADDED: Store None if residuals cannot be accessed
                eih_results['residuals'] = None

        else:
            # Calibration result object exists but is not valid
            print("Eye-in-Hand Calibration FAILED (result invalid).")
            # --->>> ADDED: Store status in dictionary
            eih_results['valid'] = False
            eih_results['transform_matrix'] = None
            try:
                # --->>> MODIFIED: Try to get and store residuals even if invalid
                residuals_list = []
                print("Available Residuals (if any):") # Keep print
                for idx, res in enumerate(calibration_result_eih.residuals()):
                    # Keep print
                    print(f"  Pose Input Index {idx}: Rotational Residual {res.rotation():.4f} deg, Translational Residual {res.translation():.4f} mm")
                    # --->>> ADDED: Append residual data to list
                    residuals_list.append({
                        'pose_index': idx,
                        'rotation_deg': res.rotation(),
                        'translation_mm': res.translation()
                    })
                eih_results['residuals'] = residuals_list # Store the list
            except Exception:
                print("      Could not retrieve residuals.") # Keep print
                # --->>> ADDED: Store None if residuals cannot be accessed
                eih_results['residuals'] = None

    except Exception as e:
        # Error during the calibrate_eye_in_hand call itself or subsequent processing
        print(f"ERROR during Eye-in-Hand calibration section: {e}")
        # --->>> ADDED: Store failure status and error message
        eih_results['performed'] = False
        eih_results['valid'] = False
        eih_results['error'] = str(e)
        import traceback
        traceback.print_exc() # Keep traceback for console debugging

    # --->>> ADDED: Add EIH results (success or failure details) to the main dictionary
    calibration_data_to_save['eye_in_hand'] = eih_results


    # --- Save Results to YAML File --- # <--- (Comentario opcional actualizado)
    # output_filename = "calibration_results.json" # <--- LÍNEA ANTIGUA
    output_filename = "calibration_results.yaml" # <--- LÍNEA NUEVA (o .yml)
    print(f"\nSaving EIH calibration results to {output_filename}...") # <--- Mensaje actualizado
    try:
        with open(output_filename, 'w') as f:
            # Use indent for pretty printing the JSON file
            # json.dump(calibration_data_to_save, f, indent=4) # <--- LÍNEA ANTIGUA

            # Usa yaml.dump para guardar. default_flow_style=False es para un formato más legible (tipo bloque)
            yaml.dump(calibration_data_to_save, f, default_flow_style=False, indent=4) # <--- LÍNEA NUEVA
        print(f"Successfully saved results to {output_filename}") # <--- Mensaje actualizado
    except TypeError as e:
        # Esta excepción aún puede ocurrir si hay tipos de datos no serializables para YAML
        print(f"ERROR: Could not serialize results to YAML. Check data types. Error: {e}")
    except IOError as e:
        print(f"ERROR: Could not write to file {output_filename}. Error: {e}")
    except Exception as e:
        # Puedes capturar yaml.YAMLError si quieres ser más específico, pero Exception es más general
        print(f"An unexpected error occurred while saving YAML: {e}")

    print("\nCalibration process finished.") # Original final message


# --- Script Entry Point ---
# UNCHANGED FROM ORIGINAL
if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        print(f"\n--- An unhandled error occurred: ---")
        print(e)
        import traceback
        traceback.print_exc()
    finally:
        # Optional: Add any cleanup code here, e.g., disconnecting camera/robot if needed
        print("\nScript execution finished.")