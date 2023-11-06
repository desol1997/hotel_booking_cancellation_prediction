# Hotel Booking Cancellation Prediction

This project aims to build a machine learning model to predict the probability of hotel booking cancellations and deploy it as a web server using FastAPI and Docker. The model is trained on the Hotel Reservations Classification Dataset available on Kaggle.

## Table of Contents

- [About](#about)
- [Use Cases](#use-cases)
- [Data](#data)
- [Machine Learning Model](#machine-learning-model)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Deployment](#deployment)

## About

Hotel bookings often involve cancellations, affecting the hospitality business's planning and revenue. This project focuses on using machine learning to predict the likelihood of a booking being canceled. The trained model can be accessed via a web server, making it useful for hotel businesses to assess potential cancellations in advance.

## Use Cases

Utilizing a web server that serves one request at a time, the machine learning model for predicting hotel booking cancellations has a range of real-time use cases in the hotel business:

1. **Booking Confirmation or Assistance**:

   - **On-the-Spot Decision Support**: When a guest makes a booking request, the web server can use the model to assess the likelihood of cancellation in real-time. This information can help hotel staff decide whether to confirm the booking immediately or, in cases of high predicted cancellations, provide additional assistance or incentives to secure the reservation.

2. **Dynamic Pricing and Promotions**:

   - **Dynamic Pricing**: The web server can dynamically adjust room prices based on the model's predictions for the specific date and room type. When the model predicts a low chance of cancellation, room rates can be increased, potentially increasing revenue.

   - **Real-Time Promotions**: For guests with a higher predicted likelihood of cancellation, the server can automatically offer last-minute promotions or flexible cancellation policies to encourage immediate booking.

3. **Optimizing Staff and Resource Allocation**:

   - **Real-Time Staffing**: Based on the predictions for incoming bookings, the hotel can make real-time staffing decisions. For instance, if a booking has a high likelihood of cancellation, the hotel can allocate fewer staff resources while ensuring they are readily available for confirmed bookings.

   - **Resource Allocation**: The server can also adjust housekeeping schedules and room inventory based on real-time predictions, ensuring that staff and resources are allocated efficiently.

4. **Proactive Customer Service**:

   - **Predictive Customer Service**: When a booking request is received, the web server can determine the guest's likelihood of cancellation. In cases of higher predicted cancellations, the server can automatically send personalized messages or incentives to guests to encourage them to keep their reservation.

   - **Alternative Options**: In the event of a likely cancellation, the server can suggest alternative accommodations or dates to the guest, enhancing their experience and potentially retaining their business.

5. **Marketing and Upselling**:

   - **Tailored Marketing**: The web server can analyze the incoming booking requests and provide recommendations for tailored upselling offers or complementary services to maximize revenue. For guests with lower predicted cancellation probabilities, the server can suggest add-ons to enhance their stay.

   - **Targeted Promotions**: Real-time predictions allow for targeted promotions to specific customer segments. The server can automatically apply discounts or incentives based on the guest's predicted behavior.

6. **Inventory and Event Management**:

   - **Real-Time Event Planning**: For hotels hosting events, the server can use real-time predictions to adjust room allocations, ensuring that event attendees have the right accommodations based on their likelihood of cancellation.

   - **Contract Flexibility**: In negotiations with travel agencies or corporate clients, the server can use real-time predictions to adjust room block sizes and contract terms, offering flexibility to accommodate changing demand patterns.

7. **Operational Efficiency**:

   - **Real-Time Energy and Resource Management**: The server can monitor real-time occupancy predictions to optimize energy usage and resource allocation. For example, it can control heating, cooling, and lighting systems based on actual room occupancy.

   - **Supply Chain Optimization**: By analyzing incoming booking requests, the server can optimize procurement and inventory management in real-time to ensure that the hotel has the right amount of supplies on hand to meet demand.

These use cases illustrate how a machine learning model for predicting hotel booking cancellations, delivered through a web server that processes one request at a time, can be a powerful tool in enhancing operational efficiency, improving customer service, and maximizing revenue for hotels. By leveraging the model's insights in real-time, hotels can make immediate, data-driven decisions to adapt to changing booking dynamics in a dynamic and competitive industry.

## Data

The dataset used for training the machine learning model can be found on Kaggle: [Hotel Reservations Classification Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset).

## Machine Learning Model

After reviewing various models, we settled on a gradient boosting model, developed using XGBoost and other well-known machine learning libraries. This model is used to predict the likelihood of a hotel booking getting canceled.

## Dependencies

This project uses `pipenv` for managing the Python environment and dependencies. Ensure you have the following installed:

- Python (version 3.11)
- `pipenv` (Python package for managing virtual environments)

## Installation

### Step 1: Python and Git

If not already installed, download and install Python from [Python's official website](https://www.python.org/).
Additionally, you'll need Git to clone the repository. Install Git from [Git's official website](https://git-scm.com/) if not already present.

### Step 2: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/desol1997/hotel_booking_cancellation_prediction.git
cd hotel_booking_cancellation_prediction
```

### Step 3: Set Up the Environment

Ensure `pipenv` is installed. If not, install it via pip:

```bash
pip install pipenv
```

Then, create the virtual environment and install dependencies using `pipenv`:

```bash
pipenv install
```

### Step 4: Activate the Environment

Activate the virtual environment to work within it:

```bash
pipenv shell
```

### Step 5: Running the Project

Once the environment is activated, follow specific instructions provided in the project's documentation or source code to run the project.

## Deployment

The machine learning model has been deployed as a web service using FastAPI and Docker. You can test the service either locally with Docker or using `awsebcli` to deploy to AWS Elastic Beanstalk.

### Local Deployment with Docker

1. **Build the Docker Image:**
   - Use the following command to build the Docker image:
   ```bash
   docker build -t myapp .
   ```
   - Replace `myapp` with the name you want to assign to the Docker image.

2. **Run the Docker Container:**
   - Start a container locally using the following command:
   ```bash
   docker run -p 8080:8000 myapp
   ```
   - Ensure to replace 8080 with the appropriate port and myapp with your Docker image's name.

3. **Access the Application:**
   - Update the `url` variable in the `test.py` script located in the `testing` folder with `http://localhost:8080/predict/`. After making this change, execute the `test.py` script. This will enable you to view the response in JSON format, displaying the predicted likelihood of a hotel booking being canceled.

### Deployment to AWS Elastic Beanstalk using `awsebcli`

1. **AWS Account and Access Keys:**
   - Before deploying to Elastic Beanstalk using `awsebcli`, ensure you have an active AWS account.
   - You'll need the Access Key ID and Secret Access Key associated with your AWS account to configure `awsebcli`. These Access Keys are used for authentication to AWS services.

2. **Install and Configure `awsebcli`:**
   - Install `awsebcli` by running:
   ```bash
   pipenv install awsebcli
   ```
   - Configure `awsebcli` by running:
   ```bash
   eb init
   ```
   - Follow the prompts to set up your AWS credentials and select your Elastic Beanstalk application.

3. **Deploy to Elastic Beanstalk:**
   - Deploy Dockerized application to Elastic Beanstalk by executing the following command:
   ```bash
   eb create myapp-env
   ```
   - Replace myapp-env with your desired environment name.

4. **Access the Application:**
   - Update the `url` variable in the `test.py` script located in the `testing` folder with the provided URL or endpoint. After making this change, execute the `test.py` script. This will enable you to view the response in JSON format, displaying the predicted likelihood of a hotel booking being canceled.

The outcomes of the deployment are visible in the images stored in the `deployment` directory.
