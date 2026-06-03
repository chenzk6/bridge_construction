import json
from pathlib import Path
import cv2
import numpy as np

# ===== 可改参数 =====
DICT_NAME = "DICT_4X4_50"   # 常用：4x4_50
IDS = [1, 2, 3, 4, 5, 6, 7,8,9] # 要生成的marker id
MARKER_PX = 600             # 图片中marker像素边长
BORDER_BITS = 1
MARKER_SIZE_MM = 40.0     # 实际打印后边长（给位姿估计用）
OUT_DIR = Path("data/aruco_markers")
# ===================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dict_id = getattr(cv2.aruco, DICT_NAME)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    mapping = {
        "dictionary": DICT_NAME,
        "marker_size_mm": MARKER_SIZE_MM,
        "items": []
    }

    for mid in IDS:
        marker = cv2.aruco.generateImageMarker(aruco_dict, mid, MARKER_PX, borderBits=BORDER_BITS)

        # 加白边，方便打印和检测
        pad = int(MARKER_PX * 0.125)
        canvas = np.full((MARKER_PX + 2 * pad, MARKER_PX + 2 * pad), 255, dtype=np.uint8)
        canvas[pad:pad + MARKER_PX, pad:pad + MARKER_PX] = marker

        # 写id文字（文字不参与识别）
        cv2.putText(canvas, f"id={mid}", (20, canvas.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

        out_png = OUT_DIR / f"aruco_{mid:03d}.png"
        cv2.imwrite(str(out_png), canvas)

        mapping["items"].append({
            "id": mid,
            "length_mm": None,      # 这里填木块长度，例如 180.0
            "type": "wood_block"    # 可自定义
        })

    cfg = OUT_DIR / "aruco_mapping.json"
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"done: {OUT_DIR}")
    print(f"mapping: {cfg}")

if __name__ == "__main__":
    main()