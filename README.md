# SafeGuard AI Agent

## Table of Contents

1. [Data Preparation (Simulation from Computer Aided Design based Application - VCrash)](#data-preparation)
   1. [Overview](#data-preparation-overview)
   2. [VCrash Application Software](#vcrach-application-software)
   3. [Steps for Data Preparation](#steps-for-data-preparation)
   4. [Benefits of Using VCrash for Data Preparation](#benefits-of-using-vcrash-for-data-preparation)
2. [Sensor Fusion](#sensor-fusion)
   1. [Overview](#sensor-fusion-overview)
   2. [Principles of Sensor Fusion](#principles-of-sensor-fusion)
   3. [Types of Sensors Involved](#types-of-sensors-involved)
   4. [Data Fusion Techniques](#data-fusion-techniques)
   5. [Implementation in the Project](#implementation-in-the-project)
   6. [Challenges and Solutions](#challenges-and-solutions)
   7. [Results and Performance](#results-and-performance)
3. [Kalman Filtering (Mathematical Imputation)](#kalman-filtering)
   1. [Introduction](#kalman-filtering-introduction)
   2. [Significance in Machine Learning (ML) and Deep Learning (DL)](#kalman-filtering-significance)
   3. [Mathematical Formulation](#kalman-filtering-mathematical-formulation)
   4. [Data Description](#kalman-filtering-data-description)
   5. [Objectives](#kalman-filtering-objectives)
   6. [Methodology](#kalman-filtering-methodology)
   7. [Results and Performance](#kalman-filtering-results)
4. [Fuzzy Logic-Based Techniques](#fuzzy-logic-techniques)
   1. [Introduction](#fuzzy-logic-introduction)
   2. [Significance in Machine Learning (ML) and Deep Learning (DL)](#fuzzy-logic-significance)
   3. [Mathematical Formulation](#fuzzy-logic-mathematical-formulation)
   4. [Methodology](#fuzzy-logic-methodology)
   5. [Results](#fuzzy-logic-results)
5. [Point Cloud Part Segmentation using Dynamic Convolutional Neural Network (DGCNN)](#point-cloud-segmentation)
   1. [Introduction](#point-cloud-introduction)
   2. [Data Preparation](#point-cloud-data-preparation)
   3. [Model Architecture](#point-cloud-model-architecture)
   4. [Training and Validation](#point-cloud-training-validation)
   5. [Results](#point-cloud-results)
6. [LiDAR-Based Accident Reconstruction and Semantic Scene Reconstruction](#lidar-reconstruction)
   1. [Introduction](#lidar-introduction)
   2. [Semantic Scene Reconstruction](#lidar-scene-reconstruction)
   3. [Accident Severity-Based Car Damage Reconstruction](#lidar-accident-severity)
   4. [Visualization and Saving Deformed Point Clouds](#lidar-visualization)
   5. [Future Work](#lidar-future-work)
7. [Creation of a Derived Attribute - Severity](#derived-attribute-severity)
   1. [Loading the Dataset](#loading-dataset)
   2. [Initial Data Inspection](#initial-data-inspection)
   3. [Insertion of Empty Rows](#insertion-empty-rows)
   4. [Creation of the 'Severity' Attribute](#creation-severity-attribute)
   5. [Adjusting Timestamps](#adjusting-timestamps)
   6. [Modifying Sensor Values](#modifying-sensor-values)
   7. [Applying Changes to the Dataset](#applying-changes)
   8. [Re-saving the Modified Dataset](#re-saving-dataset)
8. [Computer Vision Techniques for Accident Detection](#computer-vision-techniques)
   1. [Purpose](#computer-vision-purpose)
   2. [Achievements](#computer-vision-achievements)
   3. [Technical Operations in the Background](#computer-vision-technical-operations)
   4. [Conclusion](#computer-vision-conclusion)
9. [API-Enabled Nearby Emergency Responders Data Scraping](#api-emergency-responders-scraping)
   1. [Overview](#api-scraping-overview)
   2. [Objectives](#api-scraping-objectives)
   3. [Methodology](#api-scraping-methodology)
   4. [Results](#api-scraping-results)
10. [API-Enabled Automated Emergency Responders Notification System](#api-emergency-responders-notification)
   1. [Overview](#api-notification-overview)
   2. [Objectives](#api-notification-objectives)
   3. [Methodology](#api-notification-methodology)
   4. [Results](#api-notification-results)

## The Workflow Process Overview

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/02a48b34-b412-476e-9dbb-3b987a76cfd2)

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/286656d1-d467-49ae-80c6-b9350643135e)

## Data Preparation (Simulation from Computer Aided Design based Application - VCrash)

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/25711cdb-c3af-476e-87e7-2c97b8c5571b)

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/6131a0e4-32a5-4366-80e8-4e8df75bac17)

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/914ae4cb-731e-4e59-b5d8-a24ab73db18a)


### Data Preparation Overview

For the SafeGuard AI project, accurate and detailed accident data is crucial for training and testing the machine learning models. To achieve this, we utilize VCrash application software for simulating accident scenarios and reconstructing real-world accidents. This section outlines the process of data preparation using VCrash, including the steps involved in simulating accidents, capturing sensor data, and integrating this data into our system.

### VCrash Application Software

VCrash is a sophisticated Computer Aided Design tool used for accident reconstruction and simulation. It allows for the creation of detailed accident scenarios, providing valuable data that mimics real-world collisions. This data is instrumental in training machine learning models to accurately detect and analyze accidents.

### Steps for Data Preparation

1. **Scenario Selection and Setup**
   - **Define Accident Scenarios:** Choose a variety of accident types, including frontal collisions, rear-end collisions, side impacts, rollovers, and multi-vehicle crashes.
   - **Set Parameters:** Configure the parameters for each scenario, such as vehicle speeds, angles of impact, road conditions, and environmental factors.
2. **Simulation Execution**
   - **Run Simulations:** Use VCrash to execute the predefined accident scenarios. The software generates detailed simulations, capturing the dynamics of each crash.
   - **Record Sensor Data:** During simulations, record data from virtual sensors, including accelerometers, gyroscopes, pressure sensors, and vehicle dynamics sensors.
3. **Data Extraction and Processing**
   - **Extract Raw Data:** Extract the raw data generated by VCrash, which includes measurements of linear acceleration, angular velocity, pressure changes, and other relevant parameters.
   - **Data Formatting:** Format the extracted data to match the input requirements of the SafeGuard AI machine learning models. Ensure consistency in data structure and labeling.
4. **Integration with Datasets**
   - **Combine with Real-World Data:** Integrate the simulated data with real-world datasets (ONCE, A2D2, NuScenes CAN Bus) to create a comprehensive training dataset. This combined dataset enhances the robustness of the models by providing diverse scenarios and conditions.
   - **Data Augmentation:** Apply data augmentation techniques to increase the variability and volume of the training data. This can include adding noise, simulating sensor errors, and creating variations in accident scenarios.
5. **Validation and Testing**
   - **Model Training:** Use the prepared dataset to train the machine learning models. Ensure that the models are exposed to a wide range of accident scenarios for thorough learning.
   - **Validation:** Validate the trained models using a separate validation dataset to assess their performance. Fine-tune the models based on validation results to improve accuracy and reliability.
   - **Testing:** Conduct rigorous testing using both simulated and real-world accident data to evaluate the effectiveness of the SafeGuard AI system in detecting and responding to accidents.

### Benefits of Using VCrash for Data Preparation

- **Realism:** VCrash provides highly realistic accident simulations, ensuring that the data closely resembles real-world scenarios.
- **Control:** The software allows precise control over accident parameters, enabling the creation of specific and varied crash scenarios.
- **Efficiency:** Simulating accidents using VCrash is faster and more cost-effective than collecting real-world accident data.
- **Safety:** Virtual simulations eliminate the risks associated with staging real accidents, ensuring safety for all involved.

## Sensor Fusion

### Sensor Fusion Overview

Sensor fusion is a critical aspect of modern robotics and automated systems, enabling more accurate and reliable perception of the environment by integrating data from multiple sensors. The process involves combining sensory data from different sources to produce a more consistent, accurate, and useful information output than that provided by any individual sensor alone.

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/3acf8cb7-3060-4b4d-b67d-210e704faaef)

### Principles of Sensor Fusion

The main objective of sensor fusion is to improve the quality of information by reducing uncertainty and increasing accuracy. This is achieved by leveraging the strengths and compensating for the weaknesses of various sensors. The key principles include:

1. **Redundancy:** Using multiple sensors to measure the same parameter enhances reliability through error checking and correction.
2. **Complementarity:** Different sensors provide different types of information that together give a fuller picture of the environment.
3. **Timeliness:** Integrating data in real-time ensures that the system can respond promptly to changes in the environment.

### Types of Sensors Involved

Typically, sensor fusion systems utilize a variety of sensors, including:

- **Inertial Measurement Units (IMUs):** These sensors provide data on orientation, acceleration, and angular velocity. They are essential for tracking the movement and position of the system.
- **GPS:** Provides location data which is crucial for navigation and mapping.
- **Cameras:**

 Offer visual data which is useful for object recognition and environmental mapping.
- **Ultrasonic Sensors:** Measure distances to nearby objects, assisting in obstacle detection and avoidance.

### Data Fusion Techniques

There are several methods used to integrate sensor data, each with its own advantages:

1. **Kalman Filter:** A recursive algorithm used for linear data fusion, ideal for combining data from sensors with Gaussian noise characteristics.
2. **Extended Kalman Filter (EKF):** An extension of the Kalman Filter for non-linear systems.
3. **Particle Filter:** Suitable for highly non-linear and non-Gaussian processes, providing a probabilistic framework for sensor fusion.
4. **Bayesian Networks:** These are used for probabilistic inference, combining data from multiple sensors to update the state of the system.
5. **Neural Networks:** Machine learning models that can learn complex relationships between sensor inputs to produce accurate outputs.

### Implementation in the Project

In our project, sensor fusion plays a crucial role in ensuring accurate and reliable operation. The following steps outline our implementation:

1. **Sensor Selection:** We have selected a combination of IMUs, GPS, cameras, and ultrasonic sensors to cover a wide range of data needs.
2. **Data Acquisition:** Each sensor continuously collects data which is then timestamped and synchronized.
3. **Preprocessing:** The raw data from each sensor is preprocessed to filter out noise and correct for any distortions or biases.
4. **Fusion Algorithm:** We employ an Extended Kalman Filter (EKF) to integrate the sensor data. The EKF is chosen for its ability to handle the non-linearities in our system dynamics and sensor measurements.
5. **State Estimation:** The fused data is used to estimate the system's state, providing information such as position, velocity, orientation, and obstacle proximity.

### Challenges and Solutions

Several challenges were encountered during the implementation of sensor fusion:

- **Data Synchronization:** Ensuring all sensors are synchronized in time is crucial for accurate fusion. We implemented a time-stamping mechanism and used interpolation techniques to align data from different sensors.
- **Noise Management:** Different sensors have varying noise characteristics. We applied filters such as moving average filters and Gaussian filters to mitigate noise.
- **Computational Load:** Sensor fusion, especially using EKF, can be computationally intensive. We optimized our algorithm and utilized parallel processing to enhance performance.

### Results and Performance

The implementation of sensor fusion has significantly improved the accuracy and reliability of our system. The fused data provides a more comprehensive understanding of the environment, enhancing navigation, obstacle avoidance, and overall system performance. The following benefits were observed:

- **Improved Accuracy:** Position and orientation estimates are more precise compared to using individual sensors.
- **Robustness:** The system can better handle sensor failures and inaccuracies, providing continuous reliable operation.
- **Enhanced Perception:** The combination of multiple sensor data provides a richer and more detailed perception of the environment.

## 3. Kalman Filtering Mathematical Imputation

### Introduction
Kalman Filtering is a powerful mathematical tool used for estimating the state of a dynamic system from a series of noisy measurements. It is widely used in various fields such as robotics, aerospace, and automotive engineering for tasks like navigation, control, and signal processing.

### Significance in Machine Learning (ML) and Deep Learning (DL)

![image](https://github.com/MirshaMorningstar/SafeGuard-AI-Agent/assets/84216040/21d51fe6-785b-44f1-b701-460f4d6109c8)

In the context of ML and DL, Kalman Filtering is significant for several reasons:
1. **Noise Reduction:** Kalman Filters help in reducing noise from data, which is crucial for training accurate and robust models.
2. **State Estimation:** They provide a method for estimating hidden states in dynamic systems, which can be used in recurrent neural networks (RNNs) and other time-series models.
3. **Data Smoothing:** They are used for smoothing predictions and improving the accuracy of models dealing with sequential data.
4. **Sensor Fusion:** Kalman Filters are instrumental in combining data from multiple sensors, enhancing the overall quality of the input data for ML models.

### Mathematical Formulation
The Kalman Filter algorithm operates in two main steps: Prediction and Update.

1. **Prediction Step:**
   - **State Prediction:** \(\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1} + Bu_{k-1}\)
   - **Covariance Prediction:** \(P_{k|k-1} = F P_{k-1|k-1} F^T + Q\)

2. **Update Step:**
   - **Kalman Gain:** \(K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}\)
   - **State Update:** \(\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})\)
   - **Covariance Update:** \(P_{k|k} = (I - K_k H) P_{k|k-1}\)

Where:
- \(\hat{x}\) is the state estimate.
- \(P\) is the estimate covariance.
- \(F\) is the state transition model.
- \(B\) is the control-input model.
- \(u\) is the control vector.
- \(Q\) is the process noise covariance.
- \(H\) is the observation model.
- \(R\) is the observation noise covariance.
- \(z\) is the measurement vector.
- \(K\) is the Kalman gain.

### Data Description
The dataset used in this project is a CSV file containing multiple vehicle sensor readings. The dataset includes various sensor measurements such as rotation rates, linear acceleration, brake and throttle sensor values, wheel speeds, and steering data. Each row represents a snapshot of these measurements taken at one-second intervals.

### Objectives
The primary objective of this project is to apply Kalman Filtering to the sensor data to produce a smoothed and more accurate representation of the vehicle’s state over time. This involves:
1. Initializing a Kalman Filter with the appropriate dimensions for state and measurement vectors.
2. Configuring the initial state estimate and covariance matrices.
3. Iteratively applying the predict and update steps of the Kalman Filter to refine the state estimates.

### Methodology
1. **Data Loading and Preprocessing:**
   - The dataset is loaded into a pandas DataFrame.
   - The sensor readings are extracted and transposed to match the expected format for Kalman Filtering.

2. **Kalman Filter Initialization:**
   - A Kalman Filter is initialized with the dimension of the state vector (dim_x) equal to the number of features in the dataset, and the dimension of the measurement vector (dim_z) equal to the number of time steps.
   - Initial state estimates and covariance matrices (P, R, Q) are set with small values to reflect the initial uncertainty.

3. **Kalman Filter Implementation:**
   - For each time step, the Kalman Filter performs the following operations:
     - **Predict Step:** The filter predicts the next state based on the current state estimate and the process model.
     - **Update Step:** The filter updates the state estimate using the new measurement, adjusting for the measurement noise.

4. **Results Storage:**
   - The filtered state estimates are collected and stored in a new DataFrame for comparison with the original data.

### Results
The Kalman Filter successfully processed the vehicle sensor data, producing smoothed estimates of the vehicle’s state over time. The comparison between the original and filtered data shows a reduction in noise, leading to more stable and reliable sensor readings.

- **Original Data:** The original data exhibited significant fluctuations and noise, typical of raw sensor measurements in a dynamic environment.
- **Filtered Data:** The filtered data demonstrated reduced variance and a clearer representation of the underlying vehicle state, confirming the effectiveness of the Kalman Filter in smoothing noisy measurements.

