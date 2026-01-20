import os
import datetime
import random
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from root import ROOT


# ---------- Paths ----------
PROJECT_ROOT = ROOT

# ---------- Albumentations transforms ----------
BACKGROUND_AUGMENT = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=1),
        A.Downscale((0.9, 1.0), p=1),
    ], p=1)
])

ICON_AUGMENT = A.Compose([
    A.OneOf([
        A.MotionBlur(direction_range=[-0.5, 0.5], p=1),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=1),
        A.Downscale((0.9, 1.0), p=1),
    ], p=0.15),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=1),
        A.OpticalDistortion(distort_limit=0.03, shift_limit=0.03, p=1),
    ], p=0.01),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.05)
])

class SyntheticDataGenerator:
    # Cosmetic / geometry constants
    RED_TEAM_CIRCLE_BGR = (61, 61, 232)
    BLUE_TEAM_CIRCLE_BGR = (220, 150, 0)

    OBSERVER_RECTANGLE_WH = [75, 40]
    CV_THICKNESS_WORK_SIZE = 256 * 3

    ENABLE_RANDOM_ICON_ERASING = True
    ICON_ERASE_PROB_THRESHOLD = 0.7

    def __init__(self):
        self.resource_dir = os.path.join(PROJECT_ROOT, "assets")
        self.split_names = ["train", "val", "test"]

        # load all resources
        self._load_resources()

        # fixed options
        self.output_image_size = 256
        self.apply_fog_of_war = True
        self.draw_map_objects = True
        self.circle_detection_type = 0 # 0: multiclass, 1: all 0, 2: binary
        self.thickness_correction = True

        self.ping_attach_prob = 0.1
        self.ping_overlay_colors = [(177, 145, 53), (39, 39, 148)]
        self.ping_overlays = self._load_ping_overlays()

        self.champion_names = list(self.all_champion_names)
        self.num_champions = len(self.champion_names)
        assert self.num_champions >= 10, "Selected Champion is less than 10"

    # ---------------- Public API ----------------
    def generate_data(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        use_hsv_augmentation: bool = False,
        use_noise: bool = False,
        yaml: bool = True,
        allow_icon_overlap: bool = False,
    ):
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.use_hsv_augmentation = bool(use_hsv_augmentation)
        self.use_noise = bool(use_noise)

        self._make_output_directories()

        # Make icon templates (augmented + circular masked) once
        self.champion_icon_templates = self._make_circular_icon_templates(self.champion_names)

        circle_radius_pixels = int(np.round(self.icon_size_pixels * 0.5 + 0.5 + 1.0))
        yolo_box_size_norm = 2 * (circle_radius_pixels + 2) / self.minimap_canvas_size

        split_sizes = [self.n_train, self.n_val, self.n_test]
        for split_size, split_name in zip(split_sizes, self.split_names):
            for index in range(split_size):
                # refresh base canvas every 10 images
                if index % 10 == 0:
                    base_minimap = self._prepare_base_minimap()

                composed_canvas = base_minimap.copy()

                # Choose 10 champions and random augmentations
                chosen_champ_indices = np.random.permutation(self.num_champions)[:10]
                chosen_icons = np.array(self.champion_icon_templates)[chosen_champ_indices]
                chosen_icons = np.array([self._augment_icon_bgra(icon) for icon in chosen_icons])

                # Class labeling policy
                if self.circle_detection_type == 1:
                    class_ids = [0] * 10
                elif self.circle_detection_type == 2:
                    class_ids = [0] * 5 + [1] * 5
                else:
                    class_ids = list(chosen_champ_indices)

                # Optional HSV augmentation
                if self.use_hsv_augmentation:
                    hsv_augmented_icons = []
                    for icon_rgba in chosen_icons:
                        bgr = icon_rgba[:, :, :3]
                        alpha = icon_rgba[:, :, 3:]
                        bgr_hsv_aug = self._augment_hsv(bgr)
                        hsv_augmented_icons.append(np.concatenate((bgr_hsv_aug, alpha), axis=2))
                    chosen_icons = np.array(hsv_augmented_icons)

                # Sample positions for up to 10 icons
                if allow_icon_overlap:
                    top_left_positions, center_norm_positions, center_round_positions = self._sample_overlapping_icon_positions(
                        n_icons=10,
                        radius= int(np.round(self.icon_size_pixels * 0.5)),
                        min_offset_ratio=0.5,
                    )
                else:
                    top_left_positions, center_norm_positions, center_round_positions = self._sample_nonoverlapping_icon_positions(10)

                # With small probability, reduce number of champions shown
                active_indices = list(range(10))
                if np.random.uniform() < 0.05:
                    k = np.random.randint(1, 10)
                    active_indices = sorted(np.random.choice(10, size=k, replace=False).tolist())

                chosen_icons = [chosen_icons[i] for i in active_indices]
                class_ids = [class_ids[i] for i in active_indices]
                top_left_positions = [top_left_positions[i] for i in active_indices]
                center_norm_positions = [center_norm_positions[i] for i in active_indices]
                center_round_positions = [center_round_positions[i] for i in active_indices]

                # Compose icons + labels
                yolo_labels = []
                for icon_slot_index, icon_rgba in enumerate(chosen_icons):
                    icon_top_left = top_left_positions[icon_slot_index]
                    center_norm = center_norm_positions[icon_slot_index]
                    center_px_round = center_round_positions[icon_slot_index]

                    # Optional icon occlusion
                    if self.ENABLE_RANDOM_ICON_ERASING and np.random.rand() > self.ICON_ERASE_PROB_THRESHOLD and allow_icon_overlap == False:
                        icon_rgba = self._erase_random_circle_area(icon_rgba)

                    composed_canvas = self._overlay_rgba(composed_canvas, icon_rgba, icon_top_left[0], icon_top_left[1])

                    composed_canvas = self._draw_team_circle(
                        composed_canvas,
                        center_px_round,
                        circle_radius_pixels,
                        icon_slot_index
                    )

                    yolo_labels.append((
                        class_ids[icon_slot_index],
                        center_norm[0], center_norm[1],
                        yolo_box_size_norm, yolo_box_size_norm
                    ))

                # apply random pings
                composed_canvas = self._apply_random_pings(composed_canvas)

                # Optional noise
                if self.use_noise:
                    composed_canvas = self._add_gaussian_noise(composed_canvas, ratio=0.1, amp=100)

                # Resize to output
                resized_canvas = cv2.resize(
                    composed_canvas,
                    (self.output_image_size, self.output_image_size),
                    interpolation=cv2.INTER_AREA
                )

                # Observer rectangle
                resized_canvas = self._draw_observer_rectangle(resized_canvas, scale=0.4)

                # Background augmentation
                resized_canvas = BACKGROUND_AUGMENT(image=resized_canvas)["image"]

                # Save outputs
                image_out_path = os.path.join(self.save_dir, split_name, "images", f"{index}.png")
                label_out_path = os.path.join(self.save_dir, split_name, "labels", f"{index}.txt")

                cv2.imwrite(image_out_path, resized_canvas)
                with open(label_out_path, "w") as f:
                    for label in yolo_labels:
                        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(*label) + "\n")

                print(f"\rprogress [{index + 1}/{split_size}] ({split_name})", end="")

            print("")
        
        print("Synthetic Data generation completed.")

        if yaml:
            self._write_dataset_yaml()

    # ---------------- Resource loading ----------------
    def _load_resources(self):
        """
        Loads:
          - champion icons from data/raw/champion_image/*.png
          - minimap background data/raw/map11.png
          - map objects overlays from data/raw/nuki/*.png
        """
        boundary_crop = 2
        champion_dir = os.path.join(self.resource_dir, "champs")
        champion_files = os.listdir(champion_dir)

        self.champion_icons_original = {}
        self.all_champion_names = []

        for filename in sorted(champion_files):
            if not filename.lower().endswith(".png"):
                continue
            champion_name = filename[:-4]
            icon_path = os.path.join(champion_dir, f"{champion_name}.png")
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if icon is None:
                continue
            self.champion_icons_original[champion_name] = icon[boundary_crop:-boundary_crop, boundary_crop:-boundary_crop]
            self.all_champion_names.append(champion_name)

        print(f"Load champion images -- {len(self.all_champion_names)} images")

        self.minimap_bgr = self._load_minimap_image()
        self.minimap_canvas_size = self.minimap_bgr.shape[0]

        # same scaling logic you had
        self.icon_size_pixels = int(np.round(self.minimap_canvas_size / 512 * 43))
        self.map_object_overlays = self._load_map_objects()

    def _load_minimap_image(self):
        minimap_path = os.path.join(self.resource_dir, "map", "map11.png")
        minimap = cv2.imread(minimap_path)
        if minimap is None:
            raise FileNotFoundError(f"Minimap not found: {minimap_path}")

        print("Load minimap images -- done")
        return minimap

    def _load_map_objects(self):
        overlays = {}
        object_dir = os.path.join(self.resource_dir, "icons")
        filelist = [f for f in os.listdir(object_dir) if f.lower().endswith(".png")]
        for filename in filelist:
            path = os.path.join(object_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img_scaled = cv2.resize(img, (0, 0), fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
            overlays[filename[:-4]] = img_scaled
        print("Load object images -- done")
        return overlays
    
    def _load_ping_overlays(self):
        ping_dir = os.path.join(self.resource_dir, 'pings')
        overlays = []
        if not os.path.exists(ping_dir):
            print(f'[WARN] ping_dir not found: {ping_dir}')
            return overlays

        filelist = sorted(os.listdir(ping_dir))
        for fname in filelist:
            if not fname.lower().endswith('.png'):
                continue
            img = cv2.imread(os.path.join(ping_dir, fname), cv2.IMREAD_UNCHANGED)  # BGRA
            if img is None:
                continue
            overlays.append(img)
        print('Load ping overlays -- done')
        return overlays

    # ---------------- Output dirs / yaml ----------------
    def _make_output_directories(self):
        folder_name = f"lol_minimap_{self.output_image_size}_{datetime.datetime.now().strftime('%y%m%d_%H%M')}"
        self.save_dir = os.path.join(PROJECT_ROOT, "data", "synthetics", folder_name)

        for split_name in self.split_names:
            os.makedirs(os.path.join(self.save_dir, split_name, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, split_name, "labels"), exist_ok=True)

    def _write_dataset_yaml(self):
        yaml_path = os.path.join(self.save_dir, "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"path: {os.path.basename(self.save_dir)}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n")
            f.write(f"nc: {self.num_champions}\n")
            f.write("names: [{}]\n".format(", ".join([f"'{c}'" for c in self.champion_names])))

    # ---------------- Base canvas preparation ----------------
    def _prepare_base_minimap(self):
        """
        Creates base canvas:
          minimap -> optional fog -> optional map objects
        """
        if self.apply_fog_of_war:
            fog_mask = self._make_fog_of_war_mask()
            fogged = self._blend_fog(self.minimap_bgr, fog_mask)
        else:
            fogged = self.minimap_bgr.copy()

        if self.draw_map_objects:
            return self._draw_map_objects_random(fogged)
        return fogged

    # ---------------- Icon template creation ----------------
    def _make_circular_icon_templates(self, champion_names):
        """
        Creates circular-masked RGBA icons for each champion, with ICON_AUGMENT applied.
        """
        circle_size = self.icon_size_pixels + 2
        circular_alpha_mask = self._create_circular_alpha_mask(circle_size, circle_size, circle_size / 2)

        templates = []
        for name in champion_names:
            original = self.champion_icons_original[name]

            # Ensure RGBA
            if original.shape[2] == 4:
                icon_bgra = original.copy()
            else:
                icon_bgra = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)

            # Resize and apply circular alpha mask
            icon_bgra = cv2.resize(icon_bgra, (circle_size, circle_size), interpolation=cv2.INTER_AREA)
            icon_bgra = (icon_bgra * circular_alpha_mask).astype(icon_bgra.dtype)

            templates.append(icon_bgra)

        return templates

    def _create_circular_alpha_mask(self, height, width, radius):
        """
        Returns (H, W, 4) uint8 mask where alpha is 0 outside circle.
        (Kept same brute-force logic style to avoid subtle differences.)
        """
        center_y = height / 2 - 0.5
        center_x = width / 2 - 0.5
        mask = np.ones((height, width, 4), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                x = center_x - i
                y = center_y - j
                if x**2 + y**2 > radius**2:
                    mask[j, i, 3] = 0
        return mask

    # ---------------- Icon placement sampling ----------------
    def _random_top_left_in_map(self, scale=1.0, n=2):
        min_val = self.icon_size_pixels * 1.5 * scale
        max_val = (self.minimap_canvas_size - self.icon_size_pixels * 2.0) * scale
        return np.random.randint(min_val, max_val, n)

    def _sample_nonoverlapping_icon_positions(self, n_icons=1, radius=None):
        """
        Returns:
          top_left_positions: list[np.array([x,y])]
          center_norm_positions: list[np.array([x_norm,y_norm])]
          center_round_positions: list[tuple(int,int)]
        """
        if n_icons == 1:
            top_left = self._random_top_left_in_map()
            center_xy = top_left + self.icon_size_pixels * 0.5 + 0.5
            center_norm = center_xy / self.minimap_canvas_size
            center_round = tuple(np.round(center_xy).astype(int))
            return top_left, center_norm, center_round

        if radius is None:
            radius = int(np.round(self.icon_size_pixels * 0.5))

        occupancy = np.zeros(self.minimap_bgr.shape, dtype=np.uint8)

        top_left_positions = []
        while len(top_left_positions) < n_icons:
            pos = self._random_top_left_in_map()
            if occupancy[pos[1], pos[0], 0] == 0:
                top_left_positions.append(pos)
                occupancy = cv2.circle(occupancy, (pos[0], pos[1]), radius, (255, 255, 255), -1)

        center_positions = [p + self.icon_size_pixels * 0.5 + 0.5 for p in top_left_positions]
        center_norm_positions = [c / self.minimap_canvas_size for c in center_positions]
        center_round_positions = [tuple(np.round(c).astype(int)) for c in center_positions]

        return top_left_positions, center_norm_positions, center_round_positions
    
    def _sample_overlapping_icon_positions(
        self,
        n_icons: int,
        radius: int | None = None,
        min_offset_ratio: float = 0.5,
        max_tries: int = 5000,
    ):
        if radius is None:
            radius = int(np.round(self.icon_size_pixels * 0.5))

        min_offset = float(min_offset_ratio) * float(radius)
        min_offset_sq = min_offset * min_offset

        top_left_positions: list[np.ndarray] = []
        tries = 0
        while len(top_left_positions) < n_icons:
            tries += 1
            if tries > max_tries:
                min_offset_sq *= 0.8
                tries = 0

            pos = self._random_top_left_in_map()

            ok = True
            for prev in top_left_positions:
                dx = float(pos[0] - prev[0])
                dy = float(pos[1] - prev[1])
                if dx * dx + dy * dy < min_offset_sq:
                    ok = False
                    break

            if ok:
                top_left_positions.append(pos)

        center_positions = [p + self.icon_size_pixels * 0.5 + 0.5 for p in top_left_positions]
        center_norm_positions = [c / self.minimap_canvas_size for c in center_positions]
        center_round_positions = [tuple(np.round(c).astype(int)) for c in center_positions]

        return top_left_positions, center_norm_positions, center_round_positions

    # ---------------- Drawing / compositing ----------------
    def _overlay_rgba(self, background_bgr, overlay_rgba, x, y):
        bg_h, bg_w = background_bgr.shape[:2]
        if x >= bg_w or y >= bg_h:
            return background_bgr

        oh, ow = overlay_rgba.shape[:2]

        if x + ow > bg_w:
            ow = bg_w - x
            overlay_rgba = overlay_rgba[:, :ow]
        if y + oh > bg_h:
            oh = bg_h - y
            overlay_rgba = overlay_rgba[:oh]

        # Ensure alpha channel exists
        if overlay_rgba.shape[2] < 4:
            alpha = np.ones((overlay_rgba.shape[0], overlay_rgba.shape[1], 1), dtype=overlay_rgba.dtype) * 255
            overlay_rgba = np.concatenate([overlay_rgba, alpha], axis=2)

        overlay_rgb = overlay_rgba[..., :3]
        alpha = overlay_rgba[..., 3:] / 255.0

        background_bgr[y:y+oh, x:x+ow, :3] = (
            (1.0 - alpha) * background_bgr[y:y+oh, x:x+ow, :3] + alpha * overlay_rgb
        )
        return background_bgr

    def _draw_team_circle(self, canvas_bgr, center_xy, radius, icon_slot_index):
        """
        icon_slot_index < 5 -> red, else blue (kept same)
        thickness correction: draw at higher res then downsample (kept same)
        """
        if self.thickness_correction:
            enlarged = cv2.resize(
                canvas_bgr,
                (self.CV_THICKNESS_WORK_SIZE, self.CV_THICKNESS_WORK_SIZE),
                interpolation=cv2.INTER_CUBIC
            )
            center_xy = tuple(np.round(1.5 * np.array(center_xy)).astype(int))
            radius = int(1.5 * radius)
        else:
            enlarged = np.copy(canvas_bgr)

        color = self.RED_TEAM_CIRCLE_BGR if icon_slot_index < 5 else self.BLUE_TEAM_CIRCLE_BGR
        enlarged = cv2.circle(enlarged, center_xy, radius, color, 2)

        if self.thickness_correction:
            canvas_bgr = cv2.resize(
                enlarged,
                (self.minimap_canvas_size, self.minimap_canvas_size),
                interpolation=cv2.INTER_AREA
            )
            return canvas_bgr

        return enlarged

    def _draw_observer_rectangle(self, canvas_bgr, scale=1.0):
        cam_pos = self._random_top_left_in_map(scale)
        top_left = tuple(cam_pos)
        bottom_right = tuple(cam_pos + np.array(self.OBSERVER_RECTANGLE_WH))
        return cv2.rectangle(canvas_bgr, top_left, bottom_right, (255, 255, 255), 2)

    # ---------------- Fog of war ----------------
    def _make_fog_of_war_mask(self):
        fog = np.zeros(self.minimap_bgr.shape, dtype=np.uint8)
        n_circles = 40

        radius = np.random.randint(
            int(self.icon_size_pixels * 0.5),
            int(self.icon_size_pixels * 1.5),
            n_circles
        )
        x = np.random.randint(
            int(self.icon_size_pixels * 0.5),
            int(self.minimap_canvas_size - self.icon_size_pixels * 0.5),
            n_circles
        )
        y = np.random.randint(
            int(self.icon_size_pixels * 0.5),
            int(self.minimap_canvas_size - self.icon_size_pixels * 0.5),
            n_circles
        )

        for r, xt, yt in zip(radius, x, y):
            fog = cv2.circle(fog, (xt, yt), r, (255, 255, 255), -1)

        return cv2.GaussianBlur(fog, (21, 21), 0)

    def _blend_fog(self, minimap_bgr, fog_mask_bgr):
        alpha_max = 0.65
        gray = cv2.cvtColor(fog_mask_bgr, cv2.COLOR_BGR2GRAY)
        alpha = (1.0 - gray / 255.0) * alpha_max

        black = np.zeros(fog_mask_bgr.shape, dtype=np.uint8)
        out = np.empty_like(minimap_bgr)

        for i in range(3):
            out[:, :, i] = (1 - alpha) * minimap_bgr[:, :, i] + alpha * black[:, :, i]
        return out.astype(np.uint8)

    # ---------------- Map objects ----------------
    def _draw_map_objects_random(self, canvas_bgr):
        # probabilities (same meaning as your snippet)
        third_tower_skip_prob = 0.2     # if rand > 0.2 => place (80%)
        second_tower_skip_prob = 0.4    # if rand > 0.4 => place (60%) BUT only if stage-3 exists
        first_tower_skip_prob = 0.4     # if rand > 0.4 => place (60%) BUT only if stage-2 and stage-3 exist
        jungle_rate = 0.2               # used as interval thresholds

        ov = self.map_object_overlays
        out = canvas_bgr

        # --- Nexus (always) ---
        out = self._overlay_rgba(out, ov["red_nexus"], 419, 45)
        out = self._overlay_rgba(out, ov["blue_nexus"], 47, 408)

        # --- Inhibitors (always) ---
        out = self._overlay_rgba(out, ov["red_exhibitor"], 376, 28)
        out = self._overlay_rgba(out, ov["red_exhibitor"], 391, 92)
        out = self._overlay_rgba(out, ov["red_exhibitor"], 455, 108)
        out = self._overlay_rgba(out, ov["blue_exhibitor"], 26, 377)
        out = self._overlay_rgba(out, ov["blue_exhibitor"], 97, 388)
        out = self._overlay_rgba(out, ov["blue_exhibitor"], 105, 455)

        # --- Tower flags ---
        rt2 = rt3 = rm2 = rm3 = rb2 = rb3 = False
        bt2 = bt3 = bm2 = bm3 = bb2 = bb3 = False

        # --- Normal towers (stage-3 first) ---
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["red_tower"], 343, 26); rt3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["red_tower"], 374, 110); rm3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["red_tower"], 460, 132); rb3 = True

        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["blue_tower"], 29, 350); bt3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["blue_tower"], 114, 368); bm3 = True
        if np.random.rand() > third_tower_skip_prob:
            out = self._overlay_rgba(out, ov["blue_tower"], 137, 451); bb3 = True

        # --- Normal towers (depends on stage-3) ---
        if rt3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower"], 263, 34); rt2 = True
        if rm3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower"], 325, 148); rm2 = True
        if rb3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower"], 449, 212); rb2 = True

        if bt3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower"], 40, 264); bt2 = True
        if bm3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower"], 163, 329); bm2 = True
        if bb3 and (np.random.rand() > second_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower"], 228, 443); bb2 = True #

        # --- Outer towers (depends on both stage-2 and stage-3) ---
        if rt2 and rt3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower2"], 138, 18)
        if rm2 and rm3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower2"], 296, 203)
        if rb2 and rb3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["red_tower2"], 468, 342)

        if bt2 and bt3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower2"], 22, 136)
        if bm2 and bm3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower2"], 191, 272)
        if bb2 and bb3 and (np.random.rand() > first_tower_skip_prob):
            out = self._overlay_rgba(out, ov["blue_tower2"], 349, 458)

        # --- Buff spots: jungle_buff / hourglass_silver / hourglass_gold / none ---
        out = self._place_random_jungle_object(out, (235, 125), jungle_rate)
        out = self._place_random_jungle_object(out, (374, 264), jungle_rate)
        out = self._place_random_jungle_object(out, (118, 227), jungle_rate)
        out = self._place_random_jungle_object(out, (256, 368), jungle_rate)

        # --- Jungle monsters: if rand > jungle_rate (80%) ---
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 216, 81)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 262, 176)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 374, 217)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 430, 283)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 69, 215)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 123, 280)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 236, 321)
        if np.random.rand() > jungle_rate: out = self._overlay_rgba(out, ov["jungle_monster"], 282, 414)

        # --- Baron & Dragon: same threshold style ---
        out = self._place_random_epic_object(out, (156, 141), jungle_rate, key="baron")
        out = self._place_random_epic_object(out, (334, 344), jungle_rate, key="dragon_fire")

        return out

    def _place_random_jungle_object(self, canvas_bgr, xy, jungle_rate):
        x, y = xy
        ov = self.map_object_overlays
        r = np.random.rand()
        if r < jungle_rate:
            return self._overlay_rgba(canvas_bgr, ov["jungle_buff"], x, y)
        elif r < jungle_rate * 2:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_silver"], x, y)
        elif r < jungle_rate * 2.5:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_gold"], x, y)
        return canvas_bgr

    def _place_random_epic_object(self, canvas_bgr, xy, jungle_rate, key: str):
        x, y = xy
        ov = self.map_object_overlays
        r = np.random.rand()
        if r < jungle_rate:
            return self._overlay_rgba(canvas_bgr, ov[key], x, y)
        elif r < jungle_rate * 2:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_silver"], x, y)
        elif r < jungle_rate * 2.5:
            return self._overlay_rgba(canvas_bgr, ov["hourglass_gold"], x, y)
        return canvas_bgr
    
    # ---------------- Pings ----------------
    def _recolor_overlay(self, overlay, new_bgr):
        """
        Tint recolor:
        - preserves alpha channel
        - preserves overlay's luminance/shape (keeps edges & shading)
        - applies a new color tone (new_bgr) based on original brightness
        """
        if overlay is None:
            return overlay

        out = overlay.copy()

        # Ensure BGRA
        if out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)

        bgr = out[..., :3].astype(np.float32)
        alpha = out[..., 3:4]  # keep as-is (uint8)

        # Luminance (0~1). You can use mean or a perceptual weighting; mean is fine here.
        lum = bgr.mean(axis=2, keepdims=True) / 255.0  # (H, W, 1)

        # Apply tint color with luminance preserved
        tint = np.array(new_bgr, dtype=np.float32).reshape(1, 1, 3)  # (1,1,3)
        tinted_bgr = lum * tint  # (H,W,3)

        out[..., :3] = np.clip(tinted_bgr, 0, 255).astype(np.uint8)
        out[..., 3:4] = alpha  # preserve alpha
        return out
    
    def _draw_ping_circle(self, background_bgr, center, radius, color_bgr, alpha=100):
        """
        Faster ping circle drawing using ROI alpha blending.
        - No BGRA conversion of the full image
        - No full-size overlay allocation
        - Only processes a small ROI around the circle
        """
        h, w = background_bgr.shape[:2]
        cx, cy = center

        # ROI bounds (clipped to image)
        x0 = max(cx - radius, 0)
        y0 = max(cy - radius, 0)
        x1 = min(cx + radius + 1, w)
        y1 = min(cy + radius + 1, h)

        if x0 >= x1 or y0 >= y1:
            return background_bgr

        roi = background_bgr[y0:y1, x0:x1].astype(np.float32)

        # Build circle mask inside ROI
        rr = radius
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (dist2 <= rr * rr).astype(np.float32)  # (roi_h, roi_w)

        # Alpha map: per-pixel alpha (0..1)
        a = (alpha / 255.0) * mask  # (roi_h, roi_w)
        a = a[..., None]  # (roi_h, roi_w, 1)

        color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)

        # Blend: roi = (1-a)*roi + a*color
        blended = (1.0 - a) * roi + a * color

        out = background_bgr.copy()
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out
    
    def _apply_random_pings(self, canvas):
        if len(self.ping_overlays) == 0:
            return canvas

        h, w = canvas.shape[:2]
        out = canvas.copy()

        for overlay in self.ping_overlays:
            if np.random.rand() >= self.ping_attach_prob:
                continue

            idx = np.random.randint(len(self.ping_overlay_colors))
            color = self.ping_overlay_colors[idx]
            recolored = self._recolor_overlay(overlay, color)

            oh, ow = recolored.shape[:2]
            scale = 0.1 * np.random.uniform(0.9, 1.1)
            new_w = int(ow * scale)
            new_h = int(oh * scale)
            if new_w <= 0 or new_h <= 0:
                continue

            resized_overlay = cv2.resize(
                recolored, (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            # 랜덤 위치
            x = np.random.randint(0, max(0, w - new_w))
            y = np.random.randint(0, max(0, h - new_h))

            # 중심/원 radius
            radius = np.random.randint(5, 25)
            center = (x + new_w // 2, y + new_h // 2)

            out = self._draw_ping_circle(out, center, radius, color, alpha= 100)
            out = self._overlay_rgba(out, resized_overlay, x, y)

        return out

    # ---------------- Augmentations ----------------
    def _augment_icon_bgra(self, icon_bgra):
        bgr = icon_bgra[:, :, :3]
        alpha = icon_bgra[:, :, 3:]

        bgr_aug = ICON_AUGMENT(image=bgr)["image"]
        return np.concatenate([bgr_aug, alpha], axis=2)
    
    def _erase_random_circle_area(self, icon_rgba):
        h, w = icon_rgba.shape[:2]
        radius = np.random.randint(min(h, w) // 8, min(h, w) // 2)
        center_x = np.random.randint(radius - w // 2, w // 2 + radius)
        center_y = np.random.randint(radius - h // 2, h // 2 + radius)

        # Ensure alpha channel
        if icon_rgba.shape[2] == 3:
            icon_rgba = cv2.cvtColor(icon_rgba, cv2.COLOR_BGR2BGRA)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        alpha = icon_rgba[:, :, 3]
        alpha[mask == 255] = 0
        icon_rgba[:, :, 3] = alpha
        return icon_rgba

    def _augment_hsv(self, bgr, hgain=0.01, sgain=0.2, vgain=0.1, hoffset=0, soffset=0, voffset=0):
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
            dtype = bgr.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0] + hoffset) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1] + soffset, 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2] + voffset, 0, 255).astype(dtype)

            hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def _add_gaussian_noise(self, image_bgr, ratio=0.1, amp=20, mean=0, var=0.1):
        # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        h, w, c = image_bgr.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (h, w, c)) * amp
        gauss = gauss.astype(int)

        rand_mask = (np.random.rand(h, w, 1) < ratio)
        noisy = np.clip(image_bgr + gauss * rand_mask, 0, 255).astype(image_bgr.dtype)
        return noisy


# ---------------- Example run ----------------
# python -m scripts.data.synthetic_data_generator_v2
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_data(
        n_train= 10,
        n_val= 1,
        n_test= 1,
        use_hsv_augmentation= True,
        use_noise= True,
        yaml= True,
        allow_icon_overlap= True
    )
