import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from collections import Counter
import gc
import math
import cv2
import pandas as pd

# === Paths ===
base_dir = "../KITTI-360_sample"
ply_folder = "output_DM"
json_folder = os.path.join(base_dir, "bboxes_3D_cam0")
output_image_dir = "lidar_points_in_3D"
os.makedirs(output_image_dir, exist_ok=True)

excel_data = []

def duplicate_with_jitter(points, colors, times=5, scale=0.02):
    jittered_points = [points]
    jittered_colors = [colors]
    for _ in range(times):
        noise = np.random.normal(0, scale, points.shape)
        jittered_points.append(points + noise)
        jittered_colors.append(colors)
    return np.vstack(jittered_points), np.vstack(jittered_colors)

def create_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_3d_box(corners, color):
    lines = [
        [1, 4], [4, 6], [6, 3], [3, 1],
        [5, 4], [7, 6], [2, 3], [0, 1],
        [5, 7], [7, 2], [2, 0], [0, 5]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.reshape(-1, 3))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set

def process_frame(filename, ply_path, json_path, output_dir):
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_points():
            print(f"[{filename}] Empty point cloud. Skipping.")
            return

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        is_black = np.all(colors == 0, axis=1)
        black_points = points[is_black]
        black_colors = colors[is_black]
        colored_points = points[~is_black]
        colored_colors = colors[~is_black]

        if len(colored_points) == 0:
            print(f"[{filename}] No colored points found. Skipping.")
            return

        enlarged_points, enlarged_colors = duplicate_with_jitter(colored_points, colored_colors)
        enlarged_colors_rounded = np.round(enlarged_colors, decimals=3)
        total_color_counts = Counter(tuple(c) for c in enlarged_colors_rounded)

        with open(json_path, 'r') as f:
            data = json.load(f)
        boxes = [np.array(box['corners_cam0'], dtype=np.float32) for box in data]

        geometries = [
            create_point_cloud(black_points, black_colors),
            create_point_cloud(enlarged_points, enlarged_colors),
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        ]

        filtered_box_index = 1
        frame_results = []

        precision_sum = 0.0
        recall_sum = 0.0
        pr_count = 0

        box_masks = []
        for box in boxes:
            hull = Delaunay(box)
            mask = hull.find_simplex(enlarged_points) >= 0
            box_masks.append(mask)

        for i, mask in enumerate(box_masks):
            points_in_box = enlarged_points[mask]
            colors_in_box = enlarged_colors_rounded[mask]

            if len(colors_in_box) == 0:
                continue

            color_counts = Counter(tuple(clr) for clr in colors_in_box)
            dominant_color = max(color_counts.items(), key=lambda x: x[1])[0]

            TP = color_counts[dominant_color]
            total_pts = total_color_counts[dominant_color]
            FP = total_pts - TP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

            # Recall calculation
            FN = 0
            for j, other_mask in enumerate(box_masks):
                if j != i:
                    other_colors = enlarged_colors_rounded[other_mask]
                    FN += np.sum(np.all(other_colors == dominant_color, axis=1))
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            frame_results.append((filtered_box_index, total_pts, TP, precision, recall, dominant_color))

            excel_data.append({
                "Filename": filename,
                "Box No": filtered_box_index,
                "Dominant Color (R,G,B)": str(dominant_color),
                "Total Points": total_pts,
                "True Positives": TP,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3)
            })

            if total_pts >= 10:
                geometries.append(create_3d_box(boxes[i], dominant_color))
                precision_sum += precision
                recall_sum += recall
                pr_count += 1
            else:
                print(f"[{filename}] Box {filtered_box_index} filtered (pts: {total_pts} < 10)")

            filtered_box_index += 1

        avg_precision = precision_sum / pr_count if pr_count > 0 else 0.0
        avg_recall = recall_sum / pr_count if pr_count > 0 else 0.0

        # === Visualization ===
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geo in geometries:
            vis.add_geometry(geo)

        max_bound = pcd.get_max_bound()
        min_bound = pcd.get_min_bound()
        center_z = ((0.7 * max_bound[2])) / 2.0
        center = np.array([0.0, 0, center_z - 0.25 * center_z])
        extent_xz = np.array([max_bound[0] - min_bound[0], max_bound[2] - min_bound[2]])
        diag_length = np.linalg.norm(extent_xz)
        zoom = 1000 / (diag_length * 45)
        zoom = min(zoom, 0.18)

        front = np.array([0, -1, 0])
        up = np.array([0, 0, 1])
        theta = math.radians(45)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)]
        ])
        rotated_front = Rx @ front
        rotated_up = Rx @ up

        ctr = vis.get_view_control()
        ctr.set_front(rotated_front.tolist())
        ctr.set_up(rotated_up.tolist())
        ctr.set_lookat(center)
        ctr.set_zoom(zoom)

        render_option = vis.get_render_option()
        render_option.point_size = 0.5

        vis.poll_events()
        vis.update_renderer()

        out_path = os.path.join(output_dir, f"{int(filename)}.png")
        vis.capture_screen_image(out_path)
        vis.destroy_window()

        # === Overlay annotation ===
        try:
            img = cv2.imread(out_path)
            if img is not None:
                overlay_height = 40 + 40 * len(frame_results)
                overlay_width = 740
                overlay_color = (230, 230, 230)

                cv2.rectangle(img, (10, 10), (10 + overlay_width, 10 + overlay_height), overlay_color, -1)

                y_offset = 40
                cv2.putText(img, f"{filename} Precision & Recall", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                y_offset += 40

                for box_num, total_pts, TP, precision, recall, color in frame_results:
                    line = f"Box {box_num}: Pts={total_pts}, TP={TP}, Prec={precision:.3f}, Rec={recall:.3f}"
                    r, g, b = (np.array(color) * 255).astype(int)
                    color_bgr = (int(b), int(g), int(r))
                    cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
                    y_offset += 35

                avg_line = f"Avg. Precision: {avg_precision:.3f} | Avg. Recall: {avg_recall:.3f}"
                text_size = cv2.getTextSize(avg_line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = img.shape[0] - 30
                cv2.putText(img, avg_line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 200), 3)

                cv2.imwrite(out_path, img)
                print(f"[{filename}] Annotated screenshot saved with overlay.")
            else:
                print(f"[{filename}] Could not load saved image for overlay.")
        except Exception as e:
            print(f"[{filename}] Error during overlay annotation: {e}")

        del geometries
        gc.collect()

    except Exception as e:
        print(f"[{filename}] Error: {str(e)}")

# === Batch Processing ===
all_files = sorted([f for f in os.listdir(ply_folder) if f.endswith(".ply")])

for fname in all_files:
    filename = os.path.splitext(fname)[0]
    ply_path = os.path.join(ply_folder, fname)
    json_path = os.path.join(json_folder, f"BBoxes_{int(filename)}.json")

    if not os.path.exists(ply_path):
        print(f"[{filename}] Missing PLY file. Skipping.")
        continue
    if not os.path.exists(json_path):
        print(f"[{filename}] Missing JSON. Skipping.")
        continue

    process_frame(filename, ply_path, json_path, output_image_dir)

# === Save Results to Excel ===
df = pd.DataFrame(excel_data)
df.to_excel("bbox_point_counts.xlsx", index=False)
print("Results saved to bbox_point_counts.xlsx")
