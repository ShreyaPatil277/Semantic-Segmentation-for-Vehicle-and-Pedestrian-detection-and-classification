import json
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Load image and annotation
image_path = r"C:\Users\shrey\Desktop\IDD-RoadSceneSegmentation\Dataset\leftImg8bit\val\47\000897_leftImg8bit.png"
json_path = r"C:\Users\shrey\Desktop\IDD-RoadSceneSegmentation\Dataset\gtFine\val\47\000897_gtFine_polygons.json"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

with open(json_path, 'r') as f:
    annotation = json.load(f)

# Target labels (including 'truck' now)
target_labels = ['car', 'motorcycle', 'autorickshaw', 'vehicle fallback', 'road', 'person', 'truck']

# Blank mask
selected_mask = np.zeros((height, width, 3), dtype=np.uint8)

# Colors
simple_colors = {
    'road': [0, 255, 0],              # Green
    'car': [255, 0, 0],               # Red
    'motorcycle': [0, 0, 255],         # Blue
    'autorickshaw': [255, 255, 0],     # Yellow
    'vehicle fallback': [255, 0, 255], # Pink
    'person': [255, 165, 0],           # Orange
    'truck': [0, 255, 255]             # Cyan (added truck color)
}

# Store objects (optional)
selected_objects = []

# Create mask
for obj in annotation['objects']:
    label = obj['label']
    if label in target_labels:
        if isinstance(obj['polygon'], list):
            if all(isinstance(i, (int, float)) for i in obj['polygon']):
                if len(obj['polygon']) % 2 == 0:
                    polygon_points = [(obj['polygon'][i], obj['polygon'][i+1]) 
                                      for i in range(0, len(obj['polygon']), 2)]
                else:
                    continue  # Skip malformed polygon
            else:
                polygon_points = obj['polygon']

            polygon = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))

            # Fill the mask
            cv2.fillPoly(selected_mask, [polygon], color=simple_colors[label])

            # Bounding box (optional)
            x_min = int(min(polygon_points, key=lambda p: p[0])[0])
            y_min = int(min(polygon_points, key=lambda p: p[1])[1])
            x_max = int(max(polygon_points, key=lambda p: p[0])[0])
            y_max = int(max(polygon_points, key=lambda p: p[1])[1])
            cv2.rectangle(selected_mask, (x_min, y_min), (x_max, y_max), simple_colors[label], 2)

            selected_objects.append({
                "Label": label,
                "Polygon Coordinates": str(polygon_points),
                "Bounding Box": f"({x_min}, {y_min}), ({x_max}, {y_max})"
            })
        else:
            print(f"Warning: Skipping object with invalid polygon data for label: {label}")

# Save to CSV (optional)
df_selected = pd.DataFrame(selected_objects)
df_selected.to_csv('vehicle_road_objects.csv', index=False)

# Create overlay: blend the original image with mask
overlay = cv2.addWeighted(image, 0.7, selected_mask, 0.5, 0)

# Plot all together
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(selected_mask)
axs[1].set_title('Segmentation Mask')
axs[1].axis('off')

axs[2].imshow(overlay)
axs[2].set_title('Overlay Image')
axs[2].axis('off')

# Legend (only once for clarity)
legend_elements = [plt.Line2D([0], [0], color=(c[0]/255, c[1]/255, c[2]/255), lw=4, label=label) 
                   for label, c in simple_colors.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=7, fontsize=12)

plt.tight_layout()
plt.show()

print("✅ Done! Original, Segmentation, and Overlay images displayed together.")