# Deteksi Objek

**Soal :**
- Buat program pendeteksi objek.
  
  # Jawaban

```python
# Install necessary libraries
!pip install ultralytics

# Import libraries
from ultralytics import YOLO
from google.colab import files
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Upload a video or image for object detection
uploaded = files.upload()

# Load the YOLO model (pre-trained weights)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or other variants based on performance needs

# Perform object detection
for file_name in uploaded.keys():
    # Read the image
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    results = model(img)

    # Visualize results
    for result in results:
        annotated_frame = result.plot()
        plt.imshow(annotated_frame)
        plt.axis('off')
        plt.show()
```

#Hasil Gambar
![jln pt2](https://github.com/user-attachments/assets/81cf46dc-a607-4e8a-882f-9d4de94543fc)
![hsl](https://github.com/user-attachments/assets/261f1736-4a4f-495b-ab64-a109048f8fd8)

