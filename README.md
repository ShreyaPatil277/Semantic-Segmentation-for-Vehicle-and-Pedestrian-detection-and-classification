# Semantic-Segmentation-for-Vehicle-and-Pedestrian-detection-and-classification
This project implements a semantic segmentation pipeline to detect and classify different types of vehicles and pedestrians in complex urban traffic scenes using the [IDD (India Driving Dataset)](https://idd.insaan.iiit.ac.in/).

The goal is to perform **pixel-level segmentation** using polygon annotations and generate masks for objects like cars, trucks, motorcycles, autorickshaws, persons, and roads.


## 🗂️ Dataset

We used the following dataset of  **IDD Road Scene Segmentation Dataset**
IDD-20K Part I
IDD-20K Part II

This captures real Indian traffic with all its complexities like:

- Vehicle heterogeneity (e.g., autorickshaws)
- Unstructured roads and occlusion
- Dense urban traffic conditions

📥 Download the dataset from: [https://idd.insaan.iiit.ac.in/](https://idd.insaan.iiit.ac.in/)

## 🧠 Features

- ✅ Semantic segmentation using JSON polygon annotations
- 🎯 Multi-class detection: car, motorcycle, truck, autorickshaw, person, road
- 🎨 Color-coded masks for visual clarity
- 🗃️ Exports structured object data (class, polygon, bounding box) to CSV
- 📊 Matplotlib-based visualization: original, mask, overlay

## 📸 Example Outputs
| ![Original](outputs/sample_outputs/original.png) | ![Mask](outputs/sample_outputs/mask.png) | ![Overlay](outputs/sample_outputs/overlay.png) |

