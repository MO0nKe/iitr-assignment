# Local Setup
* git clone the repository
* create a virtual environment and install packages from requirements.txt
* run ```streamlit app.py```

# Deployment
* The application is deployed on HuggingFace spaces.
* Free tier compute is used. No GPU ;(

# Disclaimers
* There is some bug with the app on HF spaces. If the tab is switched after uploading the image, the app gets stuck in the performing ocr phase.
* It should not take more than 5min(20-30s on GPU) to get the output.
