import sys
import os

# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.config.settings import TRAIN_CSV, IMG_SIZE, MEAN, STD, DEVICE
    from src.data.annotations import load_csv_records
    from src.data.dataset import CsvSegDataset
    from src.models.unet import UNet

    print("[OK] All imports successful! Dependencies are installed.")
    print(f"[INFO] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")

    # 1. Test Data Loading
    print("\n--- Testing Data Pipeline ---")
    records = load_csv_records(TRAIN_CSV)
    print(f"Loaded {len(records)} records from CSV.")

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    dataset = CsvSegDataset(records, transform=transform)
    image_tensor, mask_tensor = dataset[0]

    print(f"Successfully loaded first item!")
    print(f"Image Tensor Shape : {image_tensor.shape}")   # [3, 256, 256]
    print(f"Mask  Tensor Shape : {mask_tensor.shape}")    # [1, 256, 256]

    # 2. Test Model Architecture
    print("\n--- Testing U-Net Model ---")
    model = UNet(base=32)
    print("Model instantiated successfully.")

    batch_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(batch_tensor)

    print(f"Model Output Shape : {output.shape}")  # [1, 1, 256, 256]
    print("\n[PASS] Pipeline test completed successfully!")
    print("       Data and Model modules are wired up correctly.")

except ImportError as e:
    print(f"[FAIL] Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")
except Exception as e:
    import traceback
    print(f"[FAIL] An error occurred during testing:")
    traceback.print_exc()

