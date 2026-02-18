import zivid
import numpy as np
import os
import traceback
import yaml
import math
import copy 

# --- Configuration ---
# Input ZDF file
ZDF_FILE_PATH = "original_cloud.zdf"
# YAML file containing the World -> Camera matrix
POSE_INFO_YAML_FILE = "pose_capture_info.yaml"
# Output file base name (.ply extension will be appended)
CROPPED_OUTPUT_BASE = "cropped_cloud_relative"
# Cropping threshold for the world X coordinate, relative to the calculated origin X
CROP_X_THRESHOLD_RELATIVE = 150.0 

OUTPUT_YAML_FILE = "output_parameters.yaml" 

# --- Main Script ---
def main():
    app = None 
    frame = None
    point_cloud = None
    T_World_Camera_np = None
    origin_point_world_xyz = None 

    try:
        # --- 1. Verify Input Files ---
        if not os.path.exists(ZDF_FILE_PATH):
            raise FileNotFoundError(f"El archivo ZDF de entrada no se encontró en: {ZDF_FILE_PATH}")
        print(f"Archivo ZDF de entrada encontrado en: {ZDF_FILE_PATH}")

        if not os.path.exists(POSE_INFO_YAML_FILE):
            raise FileNotFoundError(f"El archivo YAML de pose no se encontró en: {POSE_INFO_YAML_FILE}")
        print(f"Archivo YAML de pose encontrado en: {POSE_INFO_YAML_FILE}")

        # --- 2. Initialize Zivid and Load Frame ---
        print("Inicializando Zivid Application...")
        app = zivid.Application()

        print(f"Cargando Frame desde '{ZDF_FILE_PATH}'...")
        frame = zivid.Frame(ZDF_FILE_PATH)
        print("Frame cargado exitosamente.")

        # Extract point cloud data (XYZ and RGBA)
        point_cloud = frame.point_cloud()
        xyz_camera_frame = point_cloud.copy_data("xyz") 
        rgba_data = point_cloud.copy_data("rgba")      
        height, width, _ = xyz_camera_frame.shape
        print(f"Dimensiones de la nube de puntos: {height} x {width}")

        # --- 3. Load Transformation Matrix (World -> Camera) ---
        print(f"\nLeyendo matriz de transformación desde: {POSE_INFO_YAML_FILE}")
        with open(POSE_INFO_YAML_FILE, 'r') as f:
            pose_data = yaml.safe_load(f)

        # Validate and load matrix from YAML
        if isinstance(pose_data, dict) and 'world_to_camera_matrix' in pose_data:
            matrix_list = pose_data['world_to_camera_matrix']
            if (isinstance(matrix_list, list) and len(matrix_list) == 4 and
                    all(isinstance(row, list) and len(row) == 4 for row in matrix_list)):
                T_World_Camera_np = np.array(matrix_list)
                if T_World_Camera_np.shape != (4, 4):
                     raise ValueError("La matriz cargada desde YAML no tiene dimensiones 4x4.")
                print("Matriz World -> Camera cargada exitosamente desde YAML.")
                print(T_World_Camera_np)
            else:
                raise ValueError("La estructura de 'world_to_camera_matrix' en YAML no es una lista 4x4 válida.")
        else:
            raise KeyError("No se encontró la clave 'world_to_camera_matrix' en el archivo YAML o el archivo no es un diccionario válido.")

        # --- 4 & 5. Find Closest Point (Camera Origin) and Transform to World ---
        print("\nBuscando el punto más cercano a la cámara (menor valor Z)...")
        # Extract only Z coordinates
        z_values = xyz_camera_frame[:, :, 2]
        # Create copy and replace NaN with Infinity to ignore them during search
        z_values_no_nan = z_values.copy()
        z_values_no_nan[np.isnan(z_values_no_nan)] = np.inf

        # Verify if valid points exist
        if np.all(z_values_no_nan == np.inf):
            print("Advertencia: No se encontraron puntos válidos (todos son NaN) en la nube de puntos original.")
            print("No se puede calcular el punto origen para el recorte.")
            return 

        # Find index of minimum Z (ignoring NaNs)
        min_z_index_flat = np.argmin(z_values_no_nan)
        min_z_row, min_z_col = np.unravel_index(min_z_index_flat, z_values_no_nan.shape)

        # Get full XYZ coordinates of that point in the camera frame
        closest_point_camera_xyz = xyz_camera_frame[min_z_row, min_z_col, :]
        print(f"Punto más cercano encontrado en índice ({min_z_row}, {min_z_col})")
        print(f"  -> Coordenadas en Frame Cámara (X, Y, Z): {closest_point_camera_xyz}")

        # Transform this point (camera origin) to World coordinates
        # Convert to homogeneous coordinates
        origin_point_camera_homogeneous = np.append(closest_point_camera_xyz, 1.0)
        # Apply transformation T_World_Camera * P_camera
        origin_point_world_homogeneous = T_World_Camera_np @ origin_point_camera_homogeneous
        # Convert back to cartesian coordinates
        w = origin_point_world_homogeneous[3]
        if abs(w) < 1e-9: 
             w = 1.0
        origin_point_world_xyz = origin_point_world_homogeneous[:3] / w 

        # --- 6. Print World Coordinates of the Origin Point ---
        if origin_point_world_xyz is not None:
            print("\n--- Coordenadas del Punto Origen (más cercano) en el Sistema del Mundo ---")
            print(f"  -> Coordenada X (Mundo): {origin_point_world_xyz[0]:.4f}")
            print(f"  -> Coordenada Y (Mundo): {origin_point_world_xyz[1]:.4f}")
            print(f"  -> Coordenada Z (Mundo): {origin_point_world_xyz[2]:.4f}")
        else:
            print("Error: No se pudo calcular el punto origen del mundo. Terminando.")
            return 

        # --- 7. Transform ALL points to World and Crop relative to Origin ---
        print(f"\nTransformando todos los puntos válidos al Frame del Mundo para recortar...")

        # Prepare camera points for batch transformation
        xyz_camera_flat = xyz_camera_frame.reshape(-1, 3)
        # Mask for valid points (not NaN) in the original cloud
        valid_points_mask_flat = ~np.isnan(xyz_camera_flat[:, 2])
        xyz_camera_valid_flat = xyz_camera_flat[valid_points_mask_flat] 
        num_valid_points = xyz_camera_valid_flat.shape[0]

        if num_valid_points == 0:
            print("Advertencia: No hay puntos válidos en la nube original para transformar o recortar.")
            xyz_world_valid_flat = np.empty((0,3)) 
        else:
            # Convert valid points to homogeneous coordinates (N_valid, 4)
            homogeneous_camera_coords = np.hstack(
                (xyz_camera_valid_flat, np.ones((num_valid_points, 1), dtype=xyz_camera_valid_flat.dtype))
            )
            # Apply transformation: P_world_homo_T = T_World_Camera * P_camera_homo_T
            homogeneous_world_coords_T = T_World_Camera_np @ homogeneous_camera_coords.T
            # Transpose result: P_world_homo = P_world_homo_T.T
            homogeneous_world_coords = homogeneous_world_coords_T.T
            # Convert back to cartesian coordinates (N_valid, 3)
            w_world = homogeneous_world_coords[:, 3].reshape(-1, 1)
            w_world[np.abs(w_world) < 1e-9] = 1.0 
            xyz_world_valid_flat = homogeneous_world_coords[:, :3] / w_world
            print(f"Transformación completada para {num_valid_points} puntos válidos.")

        # Define the absolute cropping threshold in world coordinates
        crop_threshold_world_x = origin_point_world_xyz[0] + CROP_X_THRESHOLD_RELATIVE
        print(f"\nRecortando puntos con World X > {crop_threshold_world_x:.4f}")
        print(f"  (Basado en Origen Mundial X {origin_point_world_xyz[0]:.4f} + Umbral Relativo {CROP_X_THRESHOLD_RELATIVE:.4f})")

        # Create cropping mask based on transformed world coordinates
        if num_valid_points > 0:
            crop_mask_flat = (xyz_world_valid_flat[:, 0] <= crop_threshold_world_x)
        else:
            crop_mask_flat = np.array([], dtype=bool)

        # Map this cropping mask back to the original (Height, Width) structure
        final_keep_mask = np.zeros((height, width), dtype=bool) 
        if num_valid_points > 0:
            valid_indices_row, valid_indices_col = np.where(valid_points_mask_flat.reshape(height, width))
            indices_to_keep_in_valid_subset = np.where(crop_mask_flat)[0]
            final_rows = valid_indices_row[indices_to_keep_in_valid_subset]
            final_cols = valid_indices_col[indices_to_keep_in_valid_subset]
            final_keep_mask[final_rows, final_cols] = True

        # Create modified NumPy arrays (assigning NaN/zero for cropped points)
        xyz_cropped = xyz_camera_frame.copy()  
        rgba_cropped = rgba_data.copy()        
        # Mask for points to remove is the inverse of the mask to keep
        remove_mask = ~final_keep_mask
        # Assign NaN to XYZ coordinates of points to remove
        xyz_cropped[remove_mask] = np.nan
        # Assign transparent color to removed points
        rgba_cropped[remove_mask] = [0, 0, 0, 0]

        num_kept_points = np.sum(final_keep_mask)
        print(f"Número de puntos mantenidos después del recorte: {num_kept_points}")
        print(f"Número de puntos inválidos (NaN) en array recortado: {np.sum(remove_mask)}")

        # --- 8. Prepare Data and Visualize (Original Color + Weighted Arrows) ---

        print("\n--- Preparando datos para Visualización ---")

        points_to_visualize = np.empty((0,3))
        colors_for_o3d = np.empty((0,3)) 
        title_suffix = " (Color Original)"

        if num_kept_points > 0:
            # Use geometric crop mask 'final_keep_mask'
            points_to_visualize = xyz_world_valid_flat[crop_mask_flat] 
            colors_original_kept = rgba_data[final_keep_mask]
            if colors_original_kept.shape[0] == points_to_visualize.shape[0]:
                 colors_for_o3d = colors_original_kept[:, :3].astype(np.float64) / 255.0
            else:
                 print("Advertencia: Discrepancia puntos/colores originales. No se asignarán colores.")
                 colors_for_o3d = np.empty((0,3)) 
            print(f"Preparando {points_to_visualize.shape[0]} puntos con colores originales.")
        else:
            print("No hay puntos para visualizar después del recorte.")

        # --- 9. Find Closest Points to Virtual Points ---
        print("\nBuscando puntos más cercanos a puntos virtuales relativos al origen...")
        point_max_y = None
        point_min_z = None

        if origin_point_world_xyz is not None and points_to_visualize.shape[0] > 0:
            # --- Define Virtual Points ---
            try:
                virtual_point_1 = origin_point_world_xyz + np.array([-1000.0, 1000.0, 1000.0])
                virtual_point_2 = origin_point_world_xyz + np.array([-1000.0, -1000.0, -1000.0])
                print(f" -> Punto Virtual 1 (para Y+): {virtual_point_1}")
                print(f" -> Punto Virtual 2 (para Z-): {virtual_point_2}")

                # --- Find Closest Point to Virtual Point 1 (for Y+) ---
                diffs_to_vp1 = points_to_visualize - virtual_point_1
                dist_sq_to_vp1 = np.sum(diffs_to_vp1**2, axis=1)
                idx_closest_to_vp1 = np.argmin(dist_sq_to_vp1)
                point_max_y = points_to_visualize[idx_closest_to_vp1]
                print(f" -> Punto Y+ (más cercano a PV1) encontrado: {point_max_y}")

                # --- Find Closest Point to Virtual Point 2 (for Z-) ---
                diffs_to_vp2 = points_to_visualize - virtual_point_2
                dist_sq_to_vp2 = np.sum(diffs_to_vp2**2, axis=1)
                idx_closest_to_vp2 = np.argmin(dist_sq_to_vp2)
                point_min_z = points_to_visualize[idx_closest_to_vp2]
                print(f" -> Punto Z- (más cercano a PV2) encontrado: {point_min_z}")

            except Exception as e:
                print(f" -> Error calculando puntos virtuales o buscando más cercanos: {e}")
                point_max_y = None 
                point_min_z = None

        else:
            if origin_point_world_xyz is None:
                 print("No se pudo calcular el origen del mundo. No se pueden definir puntos virtuales.")
            if points_to_visualize.shape[0] == 0:
                 print("No hay puntos válidos después del recorte inicial. No se pueden buscar puntos cercanos.")

        # --- 10. Calculate KUKA Frame (X,Y,Z,A,B,C) ---
        print("\nCalculando parámetros del frame KUKA...")
        kuka_params = None
        vector_y_length_for_yaml = None 
        vector_z_length_for_yaml = None 

        if origin_point_world_xyz is not None and point_max_y is not None and point_min_z is not None:
            try:
                # --- 1. Define Vectors from Origin ---
                origin = origin_point_world_xyz
                vec_y_raw = point_max_y - origin
                vec_z_raw = point_min_z - origin 

                # --- Normalize Vectors and STORE LENGTHS ---
                norm_y = np.linalg.norm(vec_y_raw)
                norm_z = np.linalg.norm(vec_z_raw)
                vector_y_length_for_yaml = norm_y 
                vector_z_length_for_yaml = norm_z 

                if norm_y < 1e-6 or norm_z < 1e-6:
                    print(" -> Error: Vector Y+ o Z- tiene longitud cero. No se puede definir el frame.")
                else:
                    unit_vec_y = vec_y_raw / norm_y
                    unit_vec_z_approx = vec_z_raw / norm_z

                    # --- 2. Calculate Derived X Vector (Z- x Y+) ---
                    axis_x = np.cross(unit_vec_z_approx, unit_vec_y)
                    norm_x = np.linalg.norm(axis_x)

                    if norm_x < 1e-6:
                         print(" -> Error: Producto cruz Z- x Y+ es cero (vectores paralelos?). No se puede definir el frame.")
                    else:
                        axis_x = axis_x / norm_x

                        # --- 3. Define Frame Y Vector ---
                        axis_y = unit_vec_y

                        # --- 4. Calculate Frame Z Vector (Orthogonal) ---
                        axis_z = np.cross(axis_x, axis_y)
                        axis_z = axis_z / np.linalg.norm(axis_z)

                        # --- 5. Construct Rotation Matrix ---
                        rotation_matrix = np.column_stack((axis_x, axis_y, axis_z))

                        # --- 6. Extract A, B, C Angles (ZYX Euler - KUKA) ---
                        sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
                        singular = sy < 1e-6
                        if not singular:
                            b_rad = np.arctan2(-rotation_matrix[2,0], sy)
                            a_rad = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                            c_rad = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                        else:
                            b_rad = np.arctan2(-rotation_matrix[2,0], sy)
                            a_rad = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                            c_rad = 0
                        A_deg = np.degrees(a_rad)
                        B_deg = np.degrees(b_rad)
                        C_deg = np.degrees(c_rad)

                        # --- 7. Final Result (KUKA Parameters) ---
                        kuka_params = {
                            'X': origin[0],
                            'Y': origin[1],
                            'Z': origin[2],
                            'A': A_deg,
                            'B': B_deg,
                            'C': C_deg
                        }
                        print(" -> Parámetros KUKA calculados:")
                        print(f"    X: {kuka_params['X']:.4f}")
                        print(f"    Y: {kuka_params['Y']:.4f}")
                        print(f"    Z: {kuka_params['Z']:.4f}")
                        print(f"    A: {kuka_params['A']:.4f}")
                        print(f"    B: {kuka_params['B']:.4f}")
                        print(f"    C: {kuka_params['C']:.4f}")

                        # --- 8. SAVE TO YAML FILE ---
                        if kuka_params is not None and vector_y_length_for_yaml is not None and vector_z_length_for_yaml is not None:
                            data_to_save = {
                                'Width': float(vector_y_length_for_yaml), 
                                'Height': float(vector_z_length_for_yaml), 
                                'KUKA_Parameters': {k: float(v) for k, v in kuka_params.items()} 
                            }
                            try:
                                print(f"\nGuardando parámetros en '{OUTPUT_YAML_FILE}'...")
                                with open(OUTPUT_YAML_FILE, 'w') as f:
                                    yaml.dump(data_to_save, f, sort_keys=False, default_flow_style=False, width=1000)
                                print(" -> Guardado exitoso.")
                            except Exception as save_e:
                                print(f" -> ERROR al guardar archivo YAML: {save_e}")

            except Exception as e:
                print(f" -> Error durante el cálculo del frame KUKA: {e}")
                import traceback
                traceback.print_exc()
            else: 
                pass 

        else: 
            print(" -> No se encontraron todos los puntos necesarios (Origen, Y+, Z-) para calcular el frame y guardar.")

        # --- Open3D Visualization ---
        if points_to_visualize.shape[0] > 0:
            try:
                import open3d as o3d
                import copy 
                print("Librería Open3D encontrada.")

                # --- Helper Function to Rotate Arrows ---
                def get_rotation_matrix(source_vec_norm, target_vec_norm):
                    source_vec = source_vec_norm / np.linalg.norm(source_vec_norm)
                    target_vec = target_vec_norm / np.linalg.norm(target_vec_norm)
                    dot = np.dot(source_vec, target_vec)
                    if np.isclose(dot, 1.0): return np.identity(3)
                    if np.isclose(dot, -1.0):
                        axis_try = np.array([1.0, 0.0, 0.0])
                        if np.linalg.norm(np.cross(source_vec, axis_try)) < 1e-8:
                            axis_try = np.array([0.0, 1.0, 0.0]) 
                        angle = np.pi
                        K = np.array([[0, -axis_try[2], axis_try[1]],
                                      [axis_try[2], 0, -axis_try[0]],
                                      [-axis_try[1], axis_try[0], 0]])
                        return np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

                    axis = np.cross(source_vec, target_vec)
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm < 1e-8: return np.identity(3) 
                    axis = axis / axis_norm
                    angle = np.arccos(np.clip(dot, -1.0, 1.0))
                    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    R = R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    return R

                output_ply_file = f"{CROPPED_OUTPUT_BASE}.ply" 

                if points_to_visualize.shape[0] > 0:
                    try:
                        import open3d as o3d

                        print(f"\nIntentando guardar la nube recortada en '{output_ply_file}'...")

                        o3d_cloud_save = o3d.geometry.PointCloud()
                        o3d_cloud_save.points = o3d.utility.Vector3dVector(points_to_visualize)
                        print(f" -> Se guardarán {points_to_visualize.shape[0]} puntos.")

                        if colors_for_o3d.shape[0] == points_to_visualize.shape[0]:
                            o3d_cloud_save.colors = o3d.utility.Vector3dVector(colors_for_o3d)
                            print(f" -> Se guardarán {colors_for_o3d.shape[0]} colores correspondientes.")
                        else:
                            print(" -> Advertencia: No se guardarán colores (no coinciden con los puntos).")

                        if o3d.io.write_point_cloud(output_ply_file, o3d_cloud_save, write_ascii=False):
                            print(f" -> ¡Éxito! Nube recortada guardada en: {output_ply_file}")
                        else:
                            print(f" -> ERROR: Open3D reportó un fallo al guardar en {output_ply_file}")

                    except Exception as save_e:
                        print(f"\n--- ERROR durante el guardado del archivo PLY '{output_ply_file}' ---")
                        print(f"Detalles: {save_e}")

                else:
                    print(f"\nNo hay puntos válidos en 'points_to_visualize' para guardar en {output_ply_file}.")

                print(f"\nPreparando visualización para {points_to_visualize.shape[0]} puntos...")
                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(points_to_visualize)

                if colors_for_o3d.shape[0] == points_to_visualize.shape[0]:
                    print(f"Asignando {colors_for_o3d.shape[0]} colores originales...")
                    o3d_cloud.colors = o3d.utility.Vector3dVector(colors_for_o3d)
                else:
                    print("Advertencia: No se asignarán colores a la nube principal.")

                # --- Create Markers for Key Points (Spheres) ---
                markers = []
                marker_radius = 2.0 
                key_points_map = {} 

                if origin_point_world_xyz is not None:
                    key_points_map['origin'] = {'point': origin_point_world_xyz, 'color': [0.0, 0.0, 0.0]} 
                if point_max_y is not None:
                    key_points_map['y_plus'] = {'point': point_max_y, 'color': [0.0, 0.0, 0.0]} 
                if point_min_z is not None:
                    key_points_map['z_minus'] = {'point': point_min_z, 'color': [0.0, 0.0, 0.0]} 

                for name, data in key_points_map.items():
                    try:
                        marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
                        marker.translate(data['point'])
                        marker.paint_uniform_color(data['color'])
                        markers.append(marker)
                        print(f"Añadiendo marcador esfera {name} en {data['point']}")
                    except Exception as e:
                         print(f"Error creando marcador esfera {name}: {e}")

                # --- Create Arrows for Origin->Y+ and Origin->Z- Vectors ---
                vector_arrows = []
                arrow_base_radius_factor = 0.01 
                arrow_cone_radius_factor = 2.0  
                arrow_cone_height_factor = 0.15 

                # Origin -> Y+ Vector
                if 'origin' in key_points_map and 'y_plus' in key_points_map:
                    origin = key_points_map['origin']['point']
                    target_y = key_points_map['y_plus']['point']
                    vector_y = target_y - origin
                    vector_y_length = np.linalg.norm(vector_y)

                    if vector_y_length > 1e-6: 
                        print(f"Creando flecha para Vector Y+ (longitud: {vector_y_length:.2f})")
                        cyl_radius = vector_y_length * arrow_base_radius_factor
                        cone_radius = cyl_radius * arrow_cone_radius_factor
                        cone_height = vector_y_length * arrow_cone_height_factor
                        cyl_height = vector_y_length - cone_height

                        try:
                            # Create base arrow pointing to +Z
                            mesh_arrow_y = o3d.geometry.TriangleMesh.create_arrow(
                                cylinder_radius=cyl_radius, cone_radius=cone_radius,
                                cylinder_height=cyl_height, cone_height=cone_height
                            )
                            rotation_y = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), vector_y)
                            mesh_arrow_y.rotate(rotation_y, center=[0, 0, 0])
                            mesh_arrow_y.translate(origin) 
                            mesh_arrow_y.paint_uniform_color([0.0, 1.0, 0.0]) 
                            vector_arrows.append(mesh_arrow_y)
                        except Exception as e:
                            print(f"Error creando flecha para Vector Y+: {e}")
                    else:
                         print("Vector Y+ tiene longitud cero, no se crea flecha.")

                # Origin -> Z- Vector
                if 'origin' in key_points_map and 'z_minus' in key_points_map:
                    origin = key_points_map['origin']['point']
                    target_z = key_points_map['z_minus']['point']
                    vector_z = target_z - origin
                    vector_z_length = np.linalg.norm(vector_z)

                    if vector_z_length > 1e-6:
                        print(f"Creando flecha para Vector Z- (longitud: {vector_z_length:.2f})")
                        cyl_radius = vector_z_length * arrow_base_radius_factor
                        cone_radius = cyl_radius * arrow_cone_radius_factor
                        cone_height = vector_z_length * arrow_cone_height_factor
                        cyl_height = vector_z_length - cone_height

                        try:
                            mesh_arrow_z = o3d.geometry.TriangleMesh.create_arrow(
                                cylinder_radius=cyl_radius, cone_radius=cone_radius,
                                cylinder_height=cyl_height, cone_height=cone_height
                            )
                            rotation_z = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), vector_z)
                            mesh_arrow_z.rotate(rotation_z, center=[0, 0, 0])
                            mesh_arrow_z.translate(origin)
                            mesh_arrow_z.paint_uniform_color([0.0, 0.0, 1.0]) 
                            vector_arrows.append(mesh_arrow_z)
                        except Exception as e:
                             print(f"Error creando flecha para Vector Z-: {e}")
                    else:
                         print("Vector Z- tiene longitud cero, no se crea flecha.")

                print(f"Se crearon {len(vector_arrows)} flechas de vectores.")

                # --- Final Visualization ---
                geometries_to_draw = [o3d_cloud] + markers + vector_arrows
                window_title = "Nube Recortada + Puntos Clave + Vectores (Origen->Y+, Origen->Z-)"

                print("\nAbriendo ventana de visualización...")
                print("NOTA: Cierra la ventana de visualización para continuar/finalizar el script.")
                o3d.visualization.draw_geometries(
                    geometries_to_draw,
                    window_name=window_title,
                    width=1024,
                    height=768,
                    point_show_normal=False
                )
                print("Ventana de visualización cerrada.")

            except ImportError:
                print("\n--- ERROR: Open3D no está instalado. ---")
                print("Para visualizar el resultado, instálalo: pip install open3d")
            except Exception as e:
                print("\n--- ERROR durante la preparación o visualización con Open3D ---")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        else:
            print("Finalmente, no hay puntos válidos para visualizar.")
    
    # --- Exception Handling ---
    except FileNotFoundError as e:
        print(f"\n--- ERROR: Archivo no encontrado ---")
        print(e)
    except (ValueError, KeyError, yaml.YAMLError) as e:
        print(f"\n--- ERROR: Problema con archivo YAML, matriz o datos inválidos ---")
        print(e)
    except RuntimeError as e:
         print(f"\n--- ERROR de Zivid (Runtime) ---")
         print(e)
    except ImportError as e:
         print(f"\n--- ERROR: Falta una librería necesaria ---")
         print(e)
    except Exception as e:
        print(f"\n--- Ocurrió un error inesperado ---")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Detalles: {e}")
        traceback.print_exc() 

    # --- Cleanup ---
    finally:
        if 'point_cloud' in locals() and point_cloud:
             del point_cloud
        if 'frame' in locals() and frame:
             del frame
        if 'app' in locals() and app:
             del app
        print("\nScript finalizado.")

# --- Execute Script ---
if __name__ == "__main__":
    main()
