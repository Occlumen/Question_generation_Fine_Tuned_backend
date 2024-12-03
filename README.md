# Flask Application Setup

## Overview

This is a simple Flask web application. Follow the instructions below to get the Flask app up and running on your local machine.

## Prerequisites

Before you begin, ensure that you have the following installed:

- **Python 3.6+**
- **pip** (Python package installer)
- **Virtualenv** (optional but recommended)

### Install Python
If Python is not installed, you can download and install it from [here](https://www.python.org/downloads/).

### Install pip
pip should be installed automatically with Python. You can verify this by running:

pip --version

If pip is not installed, you can install it by following the instructions [here](https://pip.pypa.io/en/stable/installation/).

### Install Virtualenv (Optional)
It's recommended to use a virtual environment to manage dependencies. Install it globally using:

pip install virtualenv

---

## Step 1: Clone the Repository

Clone your Flask application repository to your local machine. If you're using GitHub:

git clone https://github.com/Occlumen/Question_generation_Fine_Tuned_backend.git
cd Question_generation_Fine_Tuned_backend

---

## Step 2: Set Up a Virtual Environment

Create a virtual environment to manage the dependencies.

# For Linux/MacOS
python3 -m venv venv

# For Windows
python -m venv venv

Activate the virtual environment:

- On **Linux/MacOS**:

source venv/bin/activate

- On **Windows**:

.\venv\Scripts\activate

---

## Step 3: Install Dependencies

With your virtual environment activated, install the required dependencies listed in the `requirements.txt` file.

pip install -r requirements.txt

---

## Step 4: Configure the Environment Variables

Some Flask applications require environment variables for configuration (e.g., secret keys, database URLs, etc.).

Create a `.env` file in the root of the project (if not already there), and add necessary configuration. Example:

hf_api_token=your-huggingface-token

Make sure to replace `your-huggingface-token` with appropriate values.

---

## Step 5: Run the Flask Application

Now that everything is set up, you can run the Flask application locally.

flask run

By default, Flask will run the app on `http://127.0.0.1:5000`.

---

## Step 6: Access the Application

Open your browser and go to `http://127.0.0.1:5000` to access the Flask application.

---

## Troubleshooting

- **If you see a 'ModuleNotFoundError'**: Ensure you've installed the dependencies with `pip install -r requirements.txt`.
  
- **If the server doesn’t start**: Make sure you are in the correct directory and have activated the virtual environment.

- **Permission issues on Windows**: If you encounter issues with permissions, try running the terminal as Administrator.

---
This README provides a general guide to setting up and running a basic Flask application locally. You can adjust it based on your specific application structure or configurations.
