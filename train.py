"""
NEURONEXUS InspectAI - Auto-Detecting Bottle Classifier Trainer
Company: NEURONEXUS | Application: InspectAI
Tagline: INTELLIGENCE • CONVERGENCE • INNOVATION

Automatically detects your existing 'good' and 'defect'/'defects' folders,
prepares the training dataset, and trains YOLOv8 to generate 'best.pt'.
"""
import os, sys, shutil, random

def find_images(folder):
    if not os.path.exists(folder): return []
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def auto_detect_folders():
    good_candidates = ['dataset/train/good', 'dataset/good', 'good', 'goods', 'good_bottles', 'pass']
    defect_candidates = ['dataset/train/defect', 'dataset/defect', 'dataset/defects', 'defects', 'defect', 'bad', 'defect_bottles']
    
    # Check current dir subfolders for any custom name containing 'good' or 'defect'
    try:
        for item in os.listdir('.'):
            if os.path.isdir(item):
                name = item.lower()
                if ('good' in name or 'pass' in name) and item not in good_candidates and item != 'dataset':
                    good_candidates.append(item)
                if ('defect' in name or 'bad' in name) and item not in defect_candidates and item != 'dataset':
                    defect_candidates.append(item)
    except Exception:
        pass

    best_good_src, best_good_imgs = None, []
    for c in good_candidates:
        imgs = find_images(c)
        if len(imgs) > len(best_good_imgs):
            best_good_src, best_good_imgs = c, imgs

    best_defect_src, best_defect_imgs = None, []
    for c in defect_candidates:
        imgs = find_images(c)
        if len(imgs) > len(best_defect_imgs):
            best_defect_src, best_defect_imgs = c, imgs

    return (best_good_src, best_good_imgs), (best_defect_src, best_defect_imgs)

def main():
    print("==================================================")
    print("NEURONEXUS InspectAI - Smart Model Training Station")
    print("==================================================")

    (good_src, good_imgs), (defect_src, defect_imgs) = auto_detect_folders()

    print(f"\n📂 Auto-Detected Image Folders:")
    print(f"  • Good Bottles:   {len(good_imgs)} images found in '{good_src or 'Not found'}'")
    print(f"  • Defect Bottles: {len(defect_imgs)} images found in '{defect_src or 'Not found'}'")

    if len(good_imgs) == 0 or len(defect_imgs) == 0:
        print("\n[!] Could not find bottle photos in standard folders.")
        print("Please check what your folders are named. Recommended names:")
        print("  - Folder for valid bottles:   'good' or 'dataset/good'")
        print("  - Folder for defect bottles:  'defects' (or 'defect' or 'dataset/defect')")
        return

    # Prepare dataset directory structure (train + val split)
    for split in ["train", "val"]:
        for cls in ["good", "defect"]:
            os.makedirs(os.path.join("dataset", split, cls), exist_ok=True)

    # Copy / Sync into dataset/train and dataset/val
    if good_src != "dataset/train/good":
        print("\nSyncing good images into training dataset...")
        random.seed(42)
        random.shuffle(good_imgs)
        val_count_g = max(1, int(len(good_imgs) * 0.15))
        val_g = good_imgs[:val_count_g]
        train_g = good_imgs[val_count_g:]
        for f in train_g:
            shutil.copy(f, os.path.join("dataset", "train", "good", os.path.basename(f)))
        for f in val_g:
            shutil.copy(f, os.path.join("dataset", "val", "good", os.path.basename(f)))

    if defect_src != "dataset/train/defect":
        print("Syncing defect images into training dataset...")
        random.seed(42)
        random.shuffle(defect_imgs)
        val_count_d = max(1, int(len(defect_imgs) * 0.15))
        val_d = defect_imgs[:val_count_d]
        train_d = defect_imgs[val_count_d:]
        for f in train_d:
            shutil.copy(f, os.path.join("dataset", "train", "defect", os.path.basename(f)))
        for f in val_d:
            shutil.copy(f, os.path.join("dataset", "val", "defect", os.path.basename(f)))

    train_good_count = len(os.listdir("dataset/train/good"))
    train_defect_count = len(os.listdir("dataset/train/defect"))
    print(f"✔ Dataset ready: {train_good_count} good, {train_defect_count} defect images staged.")

    # Check Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[!] Ultralytics library not installed.")
        print("Please run in PowerShell: pip install ultralytics")
        return

    print("\n🚀 Starting Training on YOLOv8 Nano Classifier (20 epochs)...")
    model = YOLO("yolov8n-cls.pt")
    model.train(data="dataset", epochs=20, imgsz=224, project="runs", name="inspectai_bottle")

    best_weights = os.path.join("runs", "inspectai_bottle", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "best.pt")
        print("\n==================================================================")
        print("✔ SUCCESS: 'best.pt' trained and placed in project root!")
        print("InspectAI will now automatically use this Deep Learning Neural Network.")
        print("==================================================================")
    else:
        print("\nTraining complete. Check the 'runs/' directory for weights.")

if __name__ == "__main__":
    main()  