import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from collections import Counter
import gc
import cv2

# === Paths ===
base_dir = "../KITTI-360_sample"
ply_folder = "D:\RWU\Sem 2\Lidar Radar\Lidar - Copy\Lidar - Copy\KITTI-360_sample\output_DM"
json_folder = os.path.join(base_dir, "bboxes_3D_cam0")
output_image_dir = "top_view_outputs"
os.makedirs(output_image_dir, exist_ok=True)

results = []  # For text file summary

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

        if not results:
            results.append("Photo name & Box no. & Color & Total points of this color & Points inside bbox & Precision \\\\\n\\hline")

        filtered_box_index = 1
        frame_results = []

        dominant_colors = []  # Store dominant color for each box

        for box in boxes:
            hull = Delaunay(box)
            mask = hull.find_simplex(enlarged_points) >= 0
            colors_in_box = enlarged_colors_rounded[mask]

            if len(colors_in_box) == 0:
                continue

            color_counts = Counter(tuple(clr) for clr in colors_in_box)
            dominant_color = max(color_counts.items(), key=lambda x: x[1])[0]
            dominant_colors.append(dominant_color)

            total_pts = total_color_counts[dominant_color]
            inside_pts = color_counts[dominant_color]

            TP = inside_pts
            FP = total_pts - TP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

            color_str = "".join(f"{{{c:.3f}}}" for c in dominant_color)

            result_line = (
                f"{filename} & Box {filtered_box_index} & \\colordotrgb{color_str} & {total_pts} & {inside_pts} & {precision:.3f} \\\\")

            results.append(result_line)
            frame_results.append((filtered_box_index, total_pts, inside_pts, precision, dominant_color))

            if total_pts >= 10:
                geometries.append(create_3d_box(box, dominant_color))

            filtered_box_index += 1

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geo in geometries:
            vis.add_geometry(geo)

        max_bound = pcd.get_max_bound()
        center_z = ((0.7 * max_bound[2])) / 2.0
        center = np.array([0.0, 0, center_z - 0.25 * center_z])

        min_bound = pcd.get_min_bound()
        extent_xz = np.array([max_bound[0] - min_bound[0], max_bound[2] - min_bound[2]])
        diag_length = np.linalg.norm(extent_xz)
        zoom = 1000 / (diag_length * 45)
        zoom = min(zoom, 0.18)

        ctr = vis.get_view_control()
        ctr.set_front([0, -1, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_lookat(center)
        ctr.set_zoom(zoom)

        render_option = vis.get_render_option()
        render_option.point_size = 0.5

        vis.poll_events()
        vis.update_renderer()

        out_path = os.path.join(output_dir, f"{int(filename)}.png")
        vis.capture_screen_image(out_path)
        vis.destroy_window()

        # === Overlay precision table on screenshot ===
        # === Overlay precision table on screenshot with BLACK background ===
        try:
            img = cv2.imread(out_path)
            overlay_height = 25 + 22 * len(frame_results)
            overlay_width = 360  # Reduced width for better fit
            overlay_color = (255, 255, 255)  # Light black (dark gray)

            cv2.rectangle(img, (10, 10), (10 + overlay_width, 10 + overlay_height), overlay_color, -1)

            y_offset = 30
            cv2.putText(img, f"{filename} Precision Table", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Title in white
            y_offset += 20

            for box_num, total_pts, inside_pts, precision, dominant_color in frame_results:
                line = f"Box {box_num}: Pts={total_pts}, In={inside_pts}, Prec={precision:.3f}"
                r, g, b = (np.array(dominant_color) * 255).astype(int)
                color_bgr = (int(b), int(g), int(r))  # OpenCV uses BGR
                cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color_bgr, 2)
                y_offset += 18

            cv2.imwrite(out_path, img)
            print(f"[{filename}] 🖼️ Annotated screenshot saved with light black background overlay.")

        except Exception as e:
            print(f"[{filename}] ⚠️ Error adding precision overlay: {str(e)}")


        del geometries
        gc.collect()

    except Exception as e:
        print(f"[{filename}] ❌ Error: {str(e)}")

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

# === Save LaTeX Table to Notepad ===
with open("bbox_point_counts.txt", "w") as f:
    f.write("\n".join(results))

print("Results saved to bbox_point_counts.txt")
