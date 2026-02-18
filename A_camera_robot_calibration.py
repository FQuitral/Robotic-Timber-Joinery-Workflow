"""
This example shows how to perform Hand-Eye calibration with ROBODK,
automatically iterating through predefined poses and performing both
Eye-to-Hand and Eye-in-Hand calibrations.

Saves Eye-in-Hand calibration results (transform matrix and residuals)
to a YAML file.
"""

from robolink import * # API to communicate with RoboDK
from robodk import * # robodk robotics toolbox
import time                # For pauses
import datetime            # For Zivid exposure time
import numpy as np         # For array manipulation
import zivid               # For Zivid camera and calibration API
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
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        # Stop if the connection did not succeed
        print(status_msg)
        raise Exception("Failed to connect to robot: " + status_msg)
    print("Connected to robot.")

    # This will set to run the API programs on the robot and the simulator (online programming)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)
    print("Run mode set to RUNMODE_RUN_ROBOT.")

# Get the current joint position of the robot (updates simulator)
joints_ref = robot.Joints()

print("Moving robot to starting position...")
try:
    robot.setTool(Pose(0,0,0,0,0,0))
    robot.setFrame(robot.Parent())
    print("Tool set to flange, Frame set to robot base.")
    robot.MoveJ([-90,-120,120,0,0,0]) # Example home position
except RobolinkError as e:
    print(f"Could not move robot to home position: {e}")

# --- Zivid Camera Functions ---

def _acquire_checkerboard_frame(camera):
    """Acquire checkerboard frame with specific settings."""
    print("Configuring Zivid settings for checkerboard capture...")
    settings = zivid.Settings()
    settings.acquisitions.append(zivid.Settings.Acquisition())
    settings.acquisitions[0].aperture = 6   # Example aperture
    settings.acquisitions[0].brightness = 1.8 # Example brightness
    settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=20000) # 20ms
    settings.processing.filters.smoothing.gaussian.enabled = True

    print("Capturing checkerboard image with Zivid camera...")
    try:
        frame = camera.capture(settings)
        print("Capture successful.")
        return frame
    except Exception as e:
        print(f"Error capturing frame: {e}")
        raise

# --- Robot Pose Function ---
def _enter_robot_pose(robot, pose_joints):
    """Moves the robot to the given joint pose and returns the Cartesian pose for Zivid."""
    print(f"Moving robot to joints: {list(pose_joints)}")
    try:
        robot.MoveJ(list(pose_joints))
    except RobolinkError as e:
        print(f"RoboDK Error during MoveJ: {e}")
        raise Exception(f"Failed to move robot to joints {list(pose_joints)}")

    print("Waiting for robot to stabilize...")
    time.sleep(3)

    print("Obtaining Cartesian pose from robot...")
    try:
        pose_rdk = robot.Pose()
    except RobolinkError as e:
        print(f"RoboDK Error getting pose: {e}")
        raise Exception("Failed to get robot pose after moving.")

    pose_matrix_np = np.array(pose_rdk)
    data = np.transpose(pose_matrix_np) 

    try:
        robot_pose_zivid = zivid.calibration.Pose(data)
        print(f"Robot Cartesian pose obtained (Zivid format):\n{robot_pose_zivid.to_matrix()}")
        return robot_pose_zivid
    except Exception as e:
        print(f"Error creating Zivid Pose object: {e}")
        raise

# --- Main Execution ---

def _main():
    app = zivid.Application()

    print("Connecting to Zivid camera...")
    try:
        camera = app.connect_camera()
        print(f"Connected to camera: {camera.info.serial_number}")
    except Exception as e:
        print(f"Error connecting to Zivid camera: {e}")
        return

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
    ])
    num_poses = len(Poses)
    print(f"Defined {num_poses} poses for calibration.")

    hand_eye_input = [] 

    for index, current_pose_joints in enumerate(Poses):
        print(f"\n--- Processing Pose {index + 1} de {num_poses} ---")

        try:
            robot_pose = _enter_robot_pose(robot, current_pose_joints)
            frame = _acquire_checkerboard_frame(camera)

            max_attempts = 3
            attempt = 0
            detection_result = None
            print("Detecting checkerboard in point cloud...")
            while attempt < max_attempts and not detection_result:
                attempt += 1
                print(f"Attempting detection {attempt}/{max_attempts}...")
                current_detection = zivid.calibration.detect_feature_points(frame.point_cloud())

                if current_detection and current_detection.valid():
                    print("Checkerboard detection SUCCESSFUL.")
                    detection_result = current_detection 
                    break 
                else:
                    if attempt < max_attempts:
                        print(f"Detection FAILED (attempt {attempt}). Retrying...")
                        time.sleep(0.5) 
                    else:
                        print(f"Detection FAILED after {max_attempts} attempts for pose {index + 1}.")

            if detection_result:
                hand_eye_input.append(zivid.calibration.HandEyeInput(robot_pose, detection_result))
                print(f"Pose {index + 1} data added successfully.")
            else:
                 print(f"Skipping pose {index + 1} due to failed checkerboard detection.")

        except KeyboardInterrupt:
             print("\n*** User interrupted the process. Exiting loop. ***")
             break 
        except Exception as ex:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR processing pose {index + 1}: {ex}")
            print(f"Attempting to continue with the next pose...")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    num_valid_poses = len(hand_eye_input)
    print(f"\nFinished collecting data. Acquired {num_valid_poses} valid pose(s).")

    calibration_data_to_save = {}

    min_required_poses = 5
    if num_valid_poses < min_required_poses:
           print(f"\nInsufficient valid data ({num_valid_poses}) collected. Need at least {min_required_poses} for reliable calibration.")
           print("Calibration aborted. No results will be saved.")
           return 

    print("\nProceeding with Hand-Eye Calibration...")

    print("\n--- Performing Eye-in-Hand (eih) Calibration ---")
    eih_results = {}
    try:
        calibration_result_eih = zivid.calibration.calibrate_eye_in_hand(hand_eye_input)
        eih_results['performed'] = True

        if calibration_result_eih and calibration_result_eih.valid():
            print("Eye-in-Hand Calibration SUCCESSFUL")
            eih_results['valid'] = True
            print("EIH Result (Transform: Robot Flange -> Camera Optical Frame):")
            try:
                transform_result = calibration_result_eih.transform()
                pose_matrix_eih_np = np.array(transform_result)
                pose_matrix_eih_list = pose_matrix_eih_np.tolist() 
                eih_results['transform_matrix'] = pose_matrix_eih_list 

                print("--- Matriz 4x4 para Grasshopper (formato CSV, 8 decimales) ---")
                matrix_str_gh = "\n".join([",".join(map(lambda x: f"{x:.8f}", row)) for row in pose_matrix_eih_np]) 
                print(matrix_str_gh)
            except AttributeError as e:
                print(f"      ERROR accessing transform attribute: {e}")
                eih_results['transform_matrix'] = None
                eih_results['valid'] = False 

            print("EIH Residuals (average error per point in mm):")
            residuals_list = []
            try:
                for idx, res in enumerate(calibration_result_eih.residuals()):
                    print(f"  Pose Input Index {idx}: Rotational Residual {res.rotation():.4f} deg, Translational Residual {res.translation():.4f} mm")
                    residuals_list.append({
                        'pose_index': idx,
                        'rotation_deg': res.rotation(),
                        'translation_mm': res.translation()
                    })
                eih_results['residuals'] = residuals_list
            except Exception as res_e:
                print(f"      ERROR accessing/printing residuals: {res_e}")
                eih_results['residuals'] = None

        else:
            print("Eye-in-Hand Calibration FAILED (result invalid).")
            eih_results['valid'] = False
            eih_results['transform_matrix'] = None
            try:
                residuals_list = []
                print("Available Residuals (if any):") 
                for idx, res in enumerate(calibration_result_eih.residuals()):
                    print(f"  Pose Input Index {idx}: Rotational Residual {res.rotation():.4f} deg, Translational Residual {res.translation():.4f} mm")
                    residuals_list.append({
                        'pose_index': idx,
                        'rotation_deg': res.rotation(),
                        'translation_mm': res.translation()
                    })
                eih_results['residuals'] = residuals_list 
            except Exception:
                print("      Could not retrieve residuals.") 
                eih_results['residuals'] = None

    except Exception as e:
        print(f"ERROR during Eye-in-Hand calibration section: {e}")
        eih_results['performed'] = False
        eih_results['valid'] = False
        eih_results['error'] = str(e)
        import traceback
        traceback.print_exc() 

    calibration_data_to_save['eye_in_hand'] = eih_results

    # --- Save Results to YAML File ---
    output_filename = "calibration_results.yaml" 
    print(f"\nSaving EIH calibration results to {output_filename}...") 
    try:
        with open(output_filename, 'w') as f:
            yaml.dump(calibration_data_to_save, f, default_flow_style=False, indent=4) 
        print(f"Successfully saved results to {output_filename}") 
    except TypeError as e:
        print(f"ERROR: Could not serialize results to YAML. Check data types. Error: {e}")
    except IOError as e:
        print(f"ERROR: Could not write to file {output_filename}. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving YAML: {e}")

    print("\nCalibration process finished.") 

# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        print(f"\n--- An unhandled error occurred: ---")
        print(e)
        import traceback
        traceback.print_exc()
    finally:
        print("\nScript execution finished.")