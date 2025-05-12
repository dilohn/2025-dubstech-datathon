# 2025 DubsTech x iSchool Datathon

This repository contains the machine learning portion for the 2025 DubsTech x iSchool Datathon, UW’s homegrown data science hackathon.

![Special Mention 3 - Machine Learning Model](https://github.com/user-attachments/assets/7b8ab120-1a7e-45f9-bbe6-c3d50938a29b)  
![1st Place - Data Visualization](https://github.com/user-attachments/assets/2a110fb4-d7ff-494e-9421-9b107166b16b)


## Project Overview

The City of Seattle receives thousands of service requests from its citizens every year, ranging from pothole reports and graffiti cleanup to public safety concerns. These requests come through the Customer Service Requests web portal and the Find It, Fix It mobile app.

Efficiently managing these requests is important for:

- Ensuring a high quality of life for residents
- Maintaining public trust
- Improving city services over time

To support this goal, we hope to uncover insights, create visualizations, and train machine learning models that can help the city:

- Better understand citizen needs
- Allocate resources more effectively
- Predict and prioritize service issues before they escalate

## Tasks & Challenges
### Data Analysis & Visualization
- Visualize customer requests over time. Trends by:
  - Month
  - Quarter
  - Day of Week
  - Hour of Day
  - Year
- Identify areas with the highest service requests
- Total requests by different service types: what trends emerge? Has any grown over time?
- Which department handles the most requests?
- Are certain neighborhoods or council districts consistently reporting more issues? What types?
- Detect seasonality
-  Create a custom metric to rank neighborhoods by performance/resource need:
  - Describe the logic behind the metric
  - Present alternative versions of your formula
  - Rank neighborhoods accordingly

Visualizations can be found in the attached presentation.

### Machine Learning
Predict the number of service requests in the next 3 months for:

- Each Service Request Type
- Each Department
- A specific zip code and service request type

To take the machine learning a step further, we train our model for all combinations of zip codes and service request types.
