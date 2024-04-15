<p text-align="center">
    <picture>
      <img alt="A sylized piece of coral" src="https://raw.githubusercontent.com/jamesbconner/MADS699/main/docs/images/ProjectCoralBleaching.png" height="200">
    </picture>
</p>

# Coral Bleaching Project

Coral bleaching is a phenomenon that has been taking place across all coral reefs spread out over the world. Healthy coral contains a symbiotic algae; the coral provides a home for the algae and the algae provides food for the coral via photosynthesis. As things like climate change, tourism, and pollution alter this relationship the algae separates from the coral and this results in coral bleaching. Coral bleaching can be easily detected as the coral turns from its normal colors to a stark white color.

# Quick Start Guide

## DeepNote Setup Guide (Easiest Way to Access Project)

Click the link below to see our final report in Deepnote! After clicking on the link, sign-up or log-in to Deepnote with your credentials. The Deepnote environment already has all of the neccessary libraries imported and allows for the reader to interactively run code while reading markdown.

[DeepNote App Link](https://deepnote.com)


## Local Machine Setup Guide

You will need to generate:
1. MAPBOX_TOKEN
2. NEPTUNE_PROJECT
3. NEPTUNE_API_TOKEN
4. AWS_ACCESS_KEY_ID
5. AWS_SECRET_ACCESS_KEY
6. S3_BUCKET_NAME

With these credentials you need to copy the file called "variables.env" and paste these 6 values as they appear in that docuemnt with your newly generated information, save that file in the same location as "variables.env", and rename it to be "private_variables.env". Note that "private_variables.env" is already ignored in the git ignore so you should not have to worry about pushing your personal keys to this project if you choose to contribute.

### **MapBox**
Information to Generate: MAPBOX_TOKEN <br>
<br>
[Mapbox](https://www.mapbox.com/)

Click on the link above and either sign-up or log-in. Click "Create a Token" about half way down the page. Name your token and leave all of the defaults as they are. After you generate your token you should see it in the token list. Copy the token and set it as the variable "MAPBOX_TOKEN" in your "private_variables.env" file.

### **Neptune AI**
Information to Generate: NEPTUNE_PROJECT, NEPTUNE_API_TOKEN <br>
<br>
[Neptune AI](https://neptune.ai/)

Click on the link above and either sign-up or log-in. Click "+ Create Project" in the upper right hand corner and give your neptune space a name. The vaiable "NEPTUNE_PROJECT" will be in the form "Workspace/Project" where the workspace is shown in the upper left and the project is the name you just assigned. Set "NEPTUNE_PROJECT" in your "private_variables.env" file to take your unique name in the form "Workspace/Project".

Now you will need to click on your username in the bottom left of the Neptune homepage. Click "Get your API Token" and generate a token. Copy that token into the "private_variables.env" file, assigning the variable "NEPTUNE_API_TOKEN" your newly generated token.

### **Amazon Web Services S3 Storage**
Information to Generate: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME <br>
<br>
[AWS Amazon](https://aws.amazon.com/)

Click on the "Create an AWS Account" in the upper right if you need to make a new account or simply sign-in. From the dashboard, there should be a service within "Storage" called "S3". Click on that link and in the next page click on the yellow "Create Bucket" button. Fill in a bucket name and copy that information into the "private_variables.env" assigning it to the variable "S3_BUCKET_NAME".

From the AWS dashboard, click on your username in the upper right. From there click on "Security Credentials". Scroll down to the tab called "Access Keys" and click on the "Create Access Key". After generating this key, copy the code under "Access Key" and assign it to the variable "AWS_ACCESS_KEY_ID" in "private_variables.env". Also copy the "Secret Access Key" and assign it to the variable "AWS_SECRET_ACCESS_KEY" in "private_variables.env".

### **Python**

It is recommended that you generate a virtual environment before installing the required libraries for this project!

```
pip install -r requirements.txt
```

### Jupyter Notebook

The notebooks are meant to be run in numerical order starting with "00_Data_Cleansing.ipynb". If all of your variables in "private_variables.env" are set correctly you should be able to uncomment code as you see fit and perform hyperparameter tuning whereever you please. It should be noted that hyperparameter tuning may take a very long time depending on the model choice, region, and features so please be warned!

# Data
Our dataset was built from the following publicly available datasets:

[Global Coral Bleaching Database](https://springernature.figshare.com/collections/_/5314466)<br>
[World Bank’s World Development Indicators (WDI) data](https://databank.worldbank.org/source/world-development-indicators#)<br>
[The National Oceanic and Atmospheric Administration (NOAA)](https://coralreefwatch.noaa.gov/product/index.php)<br>
[The Nature Conservancy Marine Ecoregions Of the World (MEOW)](https://tnc.maps.arcgis.com/home/item.html?id=ed2be4cf8b7a451f84fd093c2e7660e3#overview)<br>
