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
![1](https://github.com/user-attachments/assets/572b4d0e-1a90-45f1-aa39-68b8c1d89ec9)
![2](https://github.com/user-attachments/assets/143a9834-aad8-4c0d-ba08-aea19d165382)
![3](https://github.com/user-attachments/assets/2fc4026e-c2f1-4c14-a3d6-92c526ba2194)
![4](https://github.com/user-attachments/assets/69168bb8-da39-49f3-9d8c-4f282721cf9f)


