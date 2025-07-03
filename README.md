# 🚗 Driver Drowsiness Detection System

> A real-time computer vision system that monitors driver alertness and prevents accidents by detecting signs of drowsiness through facial analysis.

## 🎯 Overview

This intelligent driver monitoring system leverages advanced computer vision techniques to analyze facial features in real-time, providing immediate alerts when drowsiness is detected. By monitoring eye closure patterns, yawning frequency, and head pose, the system serves as a crucial safety mechanism for preventing fatigue-related accidents.

## ✨ Key Features

### 🔍 **Multi-Modal Detection**
- **Eye Aspect Ratio (EAR)** - Detects prolonged eye closure (>1.5 seconds)
- **Mouth Aspect Ratio (MAR)** - Identifies yawning patterns indicating fatigue
- **Head Pose Analysis** - Monitors head tilt and positioning 

### ⚡ **Real-Time Performance**
- Low-latency video processing
- Immediate alert generation
- Optimized algorithms for smooth operation

### 🔔 **Smart Alert System**
- Visual warnings displayed on screen
- Audio alerts with varying frequencies
- Cross-platform compatibility (Windows/Unix)

### 🎛️ **Advanced Features**
- 68-point facial landmark detection
- 3D head pose estimation
- Configurable sensitivity thresholds
- Automatic camera detection

## 🏗️ System Architecture

```
├── Driver Drowsiness Detection.py  # Main application controller
├── EAR.py                         # Eye Aspect Ratio calculations
├── MAR.py                         # Mouth Aspect Ratio calculations
└── HeadPose.py                   # 3D head pose estimation
```

### 🧠 **Core Algorithms**

#### Eye Aspect Ratio (EAR)
```python
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
- Monitors vertical vs horizontal eye distances
- Triggers alert when EAR falls below threshold for extended periods

#### Mouth Aspect Ratio (MAR)
```python
MAR = (|p2-p10| + |p4-p8|) / (2 * |p1-p6|)
```
- Calculates mouth opening ratio
- Detects yawning through mouth geometry analysis

#### Head Pose Estimation
- Uses 3D model fitting with camera calibration
- Calculates rotation matrices and Euler angles
- Monitors head tilt deviation from normal driving position

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
OpenCV 4.0+
dlib
scipy
numpy
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

2. **Install dependencies**
```bash
pip install opencv-python
pip install dlib
pip install scipy
pip install numpy
```
