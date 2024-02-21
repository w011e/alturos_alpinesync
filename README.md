# Alturos Project
![alt text](data/images/Alturos-logo_white.png)

![alt text](https://www.alturos.com/wp-content/themes/yootheme/cache/d2/mobile_on_the_rock2-d2244e8c.webp)

## General Project Abstract

This project aimed to develop a machine learning algorithm capable of interpreting sensor data from skiers' mobile devices to accurately identify whether or not they are on a ski lift and if so identify which ski-lifts they are using in order to expand the capabilities of the Alturos [Skiline](https://www.alturos.com/en/skiline/) app.
<br /> 
This involved real-time data simulation, 
development of ML prediction models, and utilization of clustering algorithms to analyze and categorize lift events.

## Authors
- Tosin Aderanti
- Paul Biesold
- Raphael Penayo Schwartz
- Sebastian Rozo

## Supervisors
- Gilberto Loaker - Alturos CEO
- Ekaterina Butyugina - Constructor Academy PM
- Stephanie Sabel - Constructor Academy TA

## Approach
- Incoming data
    - Utilized sensor data from mobile devices from two different apps: 
        - [Sensor Logger](https://apps.apple.com/us/app/sensor-play-data-recorder/id921385514) for iOS devices
        -  [Sensors Toolbox](https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&hl=en&gl=US) for Android devices
- Data management
    - To train our models effectively, we utilized labeled data for supervised machine learning from 12 different sessions and two geographical locations
    - Datasets were then prepared by incorporating outlier detection and engineered features to maximize predictive performance.
- Modeling
    - Explored various machine learning models, prioritizing accuracy metrics.
    - Selected Random Forest Classifier as the most suitable model.
- Post processing
    - Misclassification handling:
        - Achieved a 96% accuracy rate with our Random Forest Classifier (RFC) model. Through the implementation of post-processing strategies, we enhanced the continuity of classified events and minimized misclassifications. These refinements contribute to a more consistent user experience and improve overall accuracy.

    - On-lift event detection:
        - Engineered "on-lift" event detection for consolidating continuous lift activities.
    - Google API integration:
        - Integrated Google API for fetching lift names and their respective locations.
    - Clustering algorithm development:
        - Devised a clustering algorithm for grouping lift rides based on their origin and destination points.
- UX development
    - Plotting and mapping of ski lift events.
    - Generated real time prediction simulation and visualization
    - Lift statistics generation:
        - Generated statistical insights, summaries, and devised strategies for database management related to lift activities.
## Requirements
- Python environments with necessary libraries found in requirements.txt.
- Access to Google API for location services.
- App to track data (if new data is to be captured and analized)

## How to Work with the Repo
Instructions on setting up the development environment (including the environment.yml file).
Guidelines for contributing to the app development and data analysis pipeline
Guidelines to
how to label data for supervised learning (binary classification or more)
train models on more data
test functions with prediction pipeline notebook
use the streamlit app
Detailed steps for running simulations and using the prediction models.

Idea ... include this somewhere or make this a separate point for the ageinda i.e. "App"
summarise different scripts we have come up for the app, i.e., lift detection, ux stuff etc?
- pipeline schematic
- ![alt text](data/images/pipeline_schematic.png)

## Potential next steps 
- Train and test models on more data from different regions
try pipeline with real time data input (no simulation )
- Add logging capability to functions
- Include real time simulation in streamlit app
- Add planned UX features to streamlit app
- Mapping functionality using google maps APIs
- Relabelling and training models v0.1.1 and v0.4.1


## Sample Results
- Streamlit app link 
- Real-time lift identification accuracy metrics.![alt text](data/images/masked_predictions.png)
- Examples of clustering outcomes for on-lift and off-lift events.
- Visualization of ski-lift usage over time. 
- "on lift" event recognition. ![alt text](data/images/On_lift_event_labeling.png)








Project Timeline
==============================

Week 3 - Day 3 14th of February
- Third meeting with Gilberto
- app updating
- DBScan clustering on_lift event work
- google API work

Week 3 - Day 2 13th of February
- Streamlit App Development
- Work on clustering names from google api
- Fix on lift event script
- turning functions into classes
- Relabeling initial files with exact on lift locations

Week 3 - Day 1 12th of February
- Real Time Simulation finished
- UX work 
- Update lift database
- manual addition of labeling notebook
- pipeline work

Week 2 - Day 5 9th February
- Mapping function work
- Function development
- real time prediction work
- prediction pipeline v04

Week 2 - Day 4, 8th February
- Modularization work started
- misclassification and on lift even detection work
- mapping work

Week 2 - Day 3, 7th February
- came up with pipeline skeleton
- updated models v0.2 and v0.3 using different datasets
- visualization of live pipeline
- realtime prediction work
- kates data added

Week 2 - Day 2, 6th February:
- Second Meeting with Gilberto
- created off format data importing algorithm
- Use Auto ML for other model scouting and tuning
- Graphing work
- added random forest model
- Feature selection work
- Pycaret

Week 2 - Day 1, 5th February:
- Feature selection work
- RF best model benchmark
- new data labeling
- DBScan work

Week 1 - Day 5, 2nd February:
- Resampling datasets to 1 sec frequency
- LSTM model work
- unsupervised models

Week 1 - Day 4, 1st February:
- meeting with constructor team
- focused on categorization supervised models
- shared clean files
- feature engineering work
- Created first LR model - 95% on training 70% on validation
- Auto ML work
- Started with RNN

Week 1 - Day 3, 31st January:
- meeting with Stephanie
- unsupervised clustering work, DBSCAN, KMEAN, etc
- added script for data handling

Week 1 - Day 2, 30th January:
- Attended First meeting with Alturos CEO Gilberto Loeker
- Team meeting to recap, brainstorm & organize
- Assigned tasks to everyone in the group
- Gitlab repo meeting. 
- Gitlab exercises

Week 1 - Day 1, 29th January: 
- Create repo
- Data Exploration
- Problem understanding
- First team meeting
- Created dummy environment.yml file
- explored Skiline app usage and cases
- task division and defining project scope
