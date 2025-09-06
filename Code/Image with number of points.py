import os
import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d
import json

# === Helper Functions ===
def load_velodyne_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def load_projection_matrix(calib_file):
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_00:'):
                values = [float(x) for x in line.strip().split()[1:]]
                return np.array(values).reshape(3, 4)
    raise ValueError("P_rect_00 not found in calibration file.")

def load_velo_to_cam_transform(cam_to_velo_file):
    with open(cam_to_velo_file, 'r') as f:
        values = [float(x) for x in f.readline().strip().split()]
    if len(values) != 12:
        raise ValueError("Expected 12 values in cam-to-velo file.")
    T_cam_to_velo = np.eye(4)
    T_cam_to_velo[:3, :] = np.array(values).reshape(3, 4)
    return np.linalg.inv(T_cam_to_velo)  # velo-to-cam

# === Paths ===
base_dir = "../KITTI-360_sample"
input_folder = os.path.join(base_dir, "data_2d_raw", "2013_05_28_drive_0000_sync", "image_00", "data_rect")
lidar_folder = os.path.join(base_dir, "data_3d_raw", "2013_05_28_drive_0000_sync", "velodyne_points", "data")
calib_file = os.path.join(base_dir, "calibration", "perspective.txt")
cam_to_velo_file = os.path.join(base_dir, "calibration", "calib_cam_to_velo.txt")

output_folder = os.path.join( "../output_segmented_AB")
os.makedirs(output_folder, exist_ok=True)

# === Load Calibration ===
P = load_projection_matrix(calib_file)
Tr = load_velo_to_cam_transform(cam_to_velo_file)

# === YOLO Configuration ===
model = YOLO("yolov8x-seg.pt")
conf_threshold = 0.5
car_class_id = 2

# === Process Each Image ===
for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith(".png"):
        continue

    print(f"Processing: {filename}")
    img_path = os.path.join(input_folder, filename)
    pcd_path = os.path.join(lidar_folder, filename.replace(".png", ".bin"))

    # Load data
    image = cv2.imread(img_path)
    points = load_velodyne_bin(pcd_path)[:, :3]

    # YOLO inference
    results = model(image, conf=conf_threshold, classes=[car_class_id])
    result = results[0]

    if result.masks is None or result.boxes is None:
        print(f"No masks found in: {filename}")
        continue

    masks_tensor = result.masks.data
    h, w = masks_tensor.shape[1], masks_tensor.shape[2]

    # Project LiDAR points to camera and then to image plane
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  # Nx4
    points_cam = (Tr @ points_hom.T).T
    points_cam = points_cam[points_cam[:, 2] > 0]  # Keep points in front of camera
    points_img = (P @ points_cam[:, :4].T).T
    points_img[:, 0] /= points_img[:, 2]
    points_img[:, 1] /= points_img[:, 2]

    # Prepare mask overlay
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(masks_tensor.shape[0])]
    scores = result.boxes.conf
    classes = result.boxes.cls.int()
    valid_indices = ((scores > conf_threshold) & (classes == car_class_id)).nonzero(as_tuple=True)[0]

    image_lidar = image.copy()
    point_colors = np.zeros_like(points_cam[:, :3])
    point_counts_with_colors = []  # Store (count, color) for each car

    for i, idx in enumerate(valid_indices):
        mask_np = masks_tensor[idx].cpu().numpy()
        binary_mask = (mask_np > 0.5).astype(np.uint8)

        kernel_size = max(3, int(min(h, w) * 0.02))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        color = colors[i]
        dark_color = tuple(int(c) for c in color)

        for c in range(3):
            mask_overlay[:, :, c][eroded_mask == 1] = color[c]

        eroded_resized = cv2.resize(eroded_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        count = 0
        for j, img_pt in enumerate(points_img):
            u, v = int(img_pt[0]), int(img_pt[1])
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                if eroded_resized[v, u] == 1:
                    cv2.circle(image_lidar, (u, v), 0, dark_color, -1)
                    point_colors[j] = np.array(dark_color) / 255.0
                    count += 1

        point_counts_with_colors.append((count, dark_color))

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    pcd.estimate_normals()
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Save point cloud
    ply_filename = filename.replace(".png", ".ply")
    ply_output_path = os.path.join(output_folder, f"{ply_filename}")
    o3d.io.write_point_cloud(ply_output_path, pcd, write_ascii=True)
    print(f"Saved 3D point cloud: {ply_output_path}")

    # Final blended image
    mask_overlay_resized = cv2.resize(mask_overlay, (image.shape[1], image.shape[0]))
    blended = cv2.addWeighted(image_lidar, 0.7, mask_overlay_resized, 0.5, 0)

    # === Draw top-left corner table with car-colored text ===
    table_width = 170
    line_height = 25
    table_height = line_height * (len(point_counts_with_colors) + 1)
    x, y = 10, 30

    #overlay = blended.copy()
    cv2.rectangle(blended, (x - 25, y - 25), (x + table_width, y - 10 + table_height), (0, 0, 0), -1)
    #blended = cv2.addWeighted(overlay, 0.7, blended, 0.3, 0)

    cv2.putText(blended, "Detected Cars:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (count, color) in enumerate(point_counts_with_colors):
        label = f"Car {i+1}: {count} pts"
        cv2.putText(blended, label, (x, y + (i+1) * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Save final image
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, blended)
    print(f"Saved: {out_path}")
