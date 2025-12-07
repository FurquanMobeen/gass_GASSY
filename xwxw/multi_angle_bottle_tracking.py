import os
import re
import cv2
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import easyocr
import timm
import math
from datetime import datetime

"""Multi-angle gas bottle tracking and data fusion.

Processes four videos (back_left, back_right, front, top) sequentially, detects bottles,
classifies them (ok/nok), detects Tarra weight & Recertification year regions, runs OCR,
and fuses detections into global bottle identities using ConvNeXt feature similarity.

Final summary written to: videos/output/multi_camera_bottle_summary.txt

Fusion heuristic:
 - Extract embedding via classifier.forward_features for each detection.
 - Cosine similarity against existing global embeddings; assign if similarity >= THRESH (default 0.90).
 - Otherwise create new global ID.
 - Embedding per global ID maintained as running average.

Update rules:
 - Classification status updated if changed (keep latest).
 - Tarra / Year stored when first valid values parsed.
 - Angles_seen records mapping: angle_name -> local_tracker_id.

Decision rule:
 - push away if status == nok OR (year detected and year < 2025)
 - keep otherwise.

Notes:
 - Sequential processing avoids GPU contention; true simultaneous multi-camera fusion
   would require timestamp synchronization and possibly re-identification fine-tuning.
"""

VIDEO_PATHS = [
    "videos/14_55_back_left_cropped.mp4",
    "videos/14_55_back_right_cropped.mp4",
    "videos/14_55_front_cropped.mp4",
    "videos/14_55_top_cropped.mp4",
]

ANGLE_TAGS = {
    'back_left': 'back_left',
    'back_right': 'back_right',
    'front': 'front',
    'top': 'top',
}

SUMMARY_PATH = 'videos/output/multi_camera_bottle_summary.txt'
os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

CURRENT_YEAR = datetime.now().year
SIMILARITY_THRESHOLD = 0.90
MAX_FRAMES_PER_VIDEO = None  # set to integer to limit frames, or None for all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models once
bottle_yolo = YOLO('models/yolo11n_bottles.pt')
text_yolo = YOLO('models/new_yolo11s_extract_tarra_weights.pt')
reader = easyocr.Reader(['en'])

# Classifier (ConvNeXtV2 base) for ok/nok + embeddings
classifier_ckpt = torch.load('models/convnextv2_base_trained.pth', map_location=device)
classifier = timm.create_model('convnextv2_base', pretrained=False, num_classes=2)
classifier.load_state_dict(classifier_ckpt)
classifier.to(device)
classifier.eval()
CLASS_NAMES = ['nok', 'ok']

clf_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_colour(seed_val):
    random.seed(seed_val)
    return tuple(random.randint(0, 255) for _ in range(3))

def extract_embedding_and_class(roi_bgr):
    if roi_bgr.size == 0:
        return None, None, 0.0
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = clf_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # Embedding before head
        feats = classifier.forward_features(tensor)  # shape [1, C]
        emb = feats.flatten(1)
        logits = classifier.head(emb) if hasattr(classifier, 'head') else classifier.classifier(emb)
        probs = torch.softmax(logits, dim=1)
        cls_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs).item())
    return emb.squeeze(0).cpu(), CLASS_NAMES[cls_idx], conf

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

def preprocess_crop_for_ocr(crop):
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def ocr_read(crop, allow_digits_only=False):
    proc = preprocess_crop_for_ocr(crop)
    kwargs = {}
    if allow_digits_only:
        kwargs['allowlist'] = '0123456789.'
    results = reader.readtext(proc, detail=0, paragraph=False, **kwargs)
    return ' '.join(results).strip() if results else ''

def parse_tarra(text):
    if not text:
        return ''
    norm = text.replace(',', '.').upper()
    m = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*KG\b', norm)
    if m:
        try:
            w = float(m.group(1))
            if 5.0 <= w <= 30.0:
                return f"{w:.1f} KG"
        except ValueError:
            pass
    # fallback numeric w/out KG
    m2 = re.search(r'\b(\d{1,2}\.\d)\b', norm)
    if m2:
        try:
            w = float(m2.group(1))
            if 5.0 <= w <= 30.0:
                return f"{w:.1f} KG"
        except ValueError:
            pass
    return ''

def parse_year(text):
    if not text:
        return ''
    m = re.search(r'\b20\d{2}\b', text)
    if m:
        y = int(m.group(0))
        if 2010 <= y <= 2099:
            return str(y)
    return ''

# Global storage
global_records = []  # list of dicts
next_global_id = 0

def match_global(embedding):
    global next_global_id
    best_id = None
    best_sim = 0.0
    for rec in global_records:
        sim = cosine_similarity(embedding, rec['embedding'])
        if sim > best_sim:
            best_sim = sim
            best_id = rec['id']
    if best_id is not None and best_sim >= SIMILARITY_THRESHOLD:
        return best_id, best_sim
    # create new
    gid = next_global_id
    next_global_id += 1
    global_records.append({
        'id': gid,
        'embedding': embedding.clone(),
        'status': 'N/A',
        'tarra': '',
        'year': '',
        'angles': {},  # angle_name -> local_id
    })
    return gid, 1.0

def update_global_record(gid, angle, local_id, status, tarra, year, embedding):
    rec = next(r for r in global_records if r['id'] == gid)
    # Running average embedding (simple)
    rec['embedding'] = (rec['embedding'] * 0.7) + (embedding * 0.3)
    if status and status != rec['status']:
        rec['status'] = status
    if tarra and not rec['tarra']:
        rec['tarra'] = tarra
    if year and not rec['year']:
        rec['year'] = year
    if angle not in rec['angles']:
        rec['angles'][angle] = local_id

def angle_name_from_path(path):
    base = os.path.basename(path).lower()
    for k in ANGLE_TAGS:
        if k in base:
            return ANGLE_TAGS[k]
    return 'unknown'

def process_video(path):
    cap = cv2.VideoCapture(path)
    angle = angle_name_from_path(path)
    local_id_mapping = {}
    next_local = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MAX_FRAMES_PER_VIDEO is not None and frame_idx >= MAX_FRAMES_PER_VIDEO:
            break
        results = bottle_yolo.track(frame, conf=0.4, tracker='bytetrack_custom.yaml', persist=True, stream=True)
        for result in results:
            for box in result.boxes:
                if box.conf[0] < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                emb, status, cls_conf = extract_embedding_and_class(roi)
                if emb is None:
                    continue
                # Local tracking ID mapping
                if box.id is not None:
                    raw_id = int(box.id[0])
                    if raw_id not in local_id_mapping:
                        local_id_mapping[raw_id] = next_local
                        next_local += 1
                    local_id = local_id_mapping[raw_id]
                else:
                    local_id = -1
                # Text detection inside ROI
                tarra_found = ''
                year_found = ''
                text_results = text_yolo(roi, conf=0.35, verbose=False)
                for tr in text_results:
                    if not hasattr(tr, 'boxes') or len(tr.boxes) == 0:
                        continue
                    for tbox in tr.boxes:
                        if float(tbox.conf[0]) < 0.35:
                            continue
                        tx1, ty1, tx2, ty2 = map(int, tbox.xyxy[0])
                        tx1 = max(0, tx1); ty1 = max(0, ty1)
                        tx2 = min(roi.shape[1]-1, tx2); ty2 = min(roi.shape[0]-1, ty2)
                        crop = roi[ty1:ty2, tx1:tx2]
                        raw_text = ocr_read(crop)
                        if raw_text:
                            if not tarra_found:
                                tarra_found = parse_tarra(raw_text)
                            if not year_found:
                                year_found = parse_year(raw_text)
                # Match / create global ID
                gid, sim = match_global(emb)
                update_global_record(gid, angle, local_id, status, tarra_found, year_found, emb)
        frame_idx += 1
    cap.release()

def main():
    for vp in VIDEO_PATHS:
        if not os.path.exists(vp):
            print(f"Warning: video not found {vp}")
            continue
        print(f"Processing {vp} ...")
        process_video(vp)

    # Write summary
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write('global_id,status,tarra,year,decision,angles,local_ids\n')
        for rec in global_records:
            status = rec['status']
            tarra = rec['tarra'] or 'N/A'
            year = rec['year'] or 'N/A'
            decision = 'keep'
            if status == 'nok':
                decision = 'push away'
            else:
                if rec['year'] and rec['year'].isdigit() and int(rec['year']) < 2025:
                    decision = 'push away'
            angles_list = ';'.join(sorted(rec['angles'].keys())) or 'N/A'
            local_map = ';'.join(f"{a}:{lid}" for a, lid in rec['angles'].items()) or 'N/A'
            line = f"{rec['id']},{status},{tarra},{year},{decision},{angles_list},{local_map}"
            f.write(line + '\n')
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Total global bottles: {len(global_records)}")

if __name__ == '__main__':
    main()