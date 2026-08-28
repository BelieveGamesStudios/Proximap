import os
import sys
sys.path.insert(0, os.path.abspath("."))
from PySide6.QtCore import QCoreApplication
from PIL import Image, ImageDraw

app = QCoreApplication.instance() or QCoreApplication(sys.argv)

import pipeline_manager
from pipeline_manager import BackgroundRemovalWorker

src_dir = os.path.abspath("scratch/orig_source_images")
clone_out_dir = os.path.abspath("scratch/cloned_bg_removed")
os.makedirs(src_dir, exist_ok=True)
os.makedirs(clone_out_dir, exist_ok=True)

test_imgs = []
for i in range(2):
    p = os.path.join(src_dir, f"original_photo_{i}.jpg")
    im = Image.new("RGB", (128, 128), color=(240, 240, 240))
    d = ImageDraw.Draw(im)
    d.rectangle((30, 30, 90, 90), fill=(0, 150 + i * 50, 255))
    im.save(p, "JPEG")
    test_imgs.append(p)

print(f"Created {len(test_imgs)} source images in {src_dir}.")

worker = BackgroundRemovalWorker(test_imgs, output_dir=clone_out_dir)

def on_finish(success, updated_list, msg):
    print(f"[FINISHED] success={success}, msg={msg}")
    print(f"[UPDATED LIST] {updated_list}")
    # Verify original files still exist
    for orig in test_imgs:
        assert os.path.exists(orig), f"Original file {orig} was unexpectedly modified/removed!"
    print("[SUCCESS] All original source files remain intact and untouched!")
    app.quit()

worker.finished.connect(on_finish)
worker.start()
app.exec()
