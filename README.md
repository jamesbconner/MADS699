<p text-align="center">
    <picture>
      <img alt="A sylized piece of coral" src="https://raw.githubusercontent.com/jamesbconner/MADS699/main/docs/images/ProjectCoralBleaching.png" height="200">
    </picture>
</p>

# Coral Bleaching Project

Coral bleaching is a phenomenon that has been taking place across all coral reefs spread out over the world. Healthy coral contains a symbiotic algae; the coral provides a home for the algae and the algae provides food for the coral via photosynthesis. As things like climate change, tourism, and pollution alter this relationship the algae separates from the coral and this results in coral bleaching. Coral bleaching can be easily detected as the coral turns from its normal colors to a stark white color.

This repository/project serves as an intermediate stepping stone building off of the backs of other groups like this article by Madireddy et al titled ["Using machine learning to develop a global coral bleaching predictor"](https://emerginginvestigators.org/articles/22-056) [1] and providing some insights as to next steps that could be looked into in the world of coral bleaching.

In doing some preliminary research for this project, the contributors looked to expand on past findings by answering the following:
1. How does turbidity affect coral bleaching?
2. How does fertilizer runoff affect coral bleaching?
3. Can we predict coral bleaching events based on weather patterns/trends?
4. Can we identify which sites are most at risk to coral bleaching?

# Quick Start Guide

## Running the Report in DeepNote (Easiest Way to Access Project)

Click the link below to see our final report in Deepnote! After clicking on the link, sign-up or log-in to Deepnote with your credentials. The Deepnote environment already has all of the neccessary libraries imported and allows for the reader to interactively run code while reading markdown.

[DeepNote App Link](https://deepnote.com)


# Jupyter Setup Guide
## Setting up Credentials

You will need to generate:
1. MAPBOX_TOKEN
2. NEPTUNE_PROJECT
3. NEPTUNE_API_TOKEN
4. AWS_ACCESS_KEY_ID
5. AWS_SECRET_ACCESS_KEY
6. S3_BUCKET_NAME

With these credentials you need to copy the file called "variables.env" and paste these 6 values as they appear in that docuemnt with your newly generated information, save that file in the same location as "variables.env", and rename it to be "private_variables.env". Note that "private_variables.env" is already ignored in the git ignore so you should not have to worry about pushing your personal keys to this project if you choose to contribute.

### MapBox
Information to Generate: MAPBOX_TOKEN
[Mapbox](https://www.mapbox.com/)

Click on the link above and either sign-up or log-in. Click "Create a Token" about half way down the page. Name your token and leave all of the defaults as they are. After you generate your token you should see it in the token list. Copy the token and set it as the variable "MAPBOX_TOKEN" in your "private_variables.env" file.

### Neptune AI
Information to Generate: NEPTUNE_PROJECT, NEPTUNE_API_TOKEN
[Neptune AI](https://neptune.ai/)

Clionk on the link above and either sign-up or log-in. Click "+ Create Project" in the upper right hand corner and give your neptune space a name. The vaiable "NEPTUNE_PROJECT" will be in the form "Workspace/Project" where the workspace is shown in the upper left and the project is the name you just assigned. Set "NEPTUNE_PROJECT" in your "private_variables.env" file to take your unique name in the form "Workspace/Project".

Now you will need to click on your username in the bottom left of the Neptune homepage. Click "Get your API Token" and generate a token. Copy that token into the "private_variables.env" file, assigning the variable "NEPTUNE_API_TOKEN" your newly generated token.

### Amazon Web Services S3 Storage
[AWS Amazon](https://aws.amazon.com/)

## Setup Runtime Environment

### Dotenv

### Python

### Jupyter Notebook



# Data
Our dataset was built from the following publicly available datasets:

[Global Coral Bleaching Database](https://springernature.figshare.com/collections/_/5314466)<br>
[World Bank’s World Development Indicators (WDI) data](https://databank.worldbank.org/source/world-development-indicators#)<br>
[The National Oceanic and Atmospheric Administration (NOAA)](https://coralreefwatch.noaa.gov/product/index.php)<br>
[The Nature Conservancy Marine Ecoregions Of the World (MEOW)](https://tnc.maps.arcgis.com/home/item.html?id=ed2be4cf8b7a451f84fd093c2e7660e3#overview)<br>


# Technologies

## Modeling
ElasticNet, RandomForest, HistGradientBoosting, XGBoost, LightGBM, Kmeans

## Experimentation
Neptune AI, HyperOpt, HalvingGridSearchCV GridSearchCV

## Evaluation & Visualization
Plotly, Matplotlib, Seaborn, Pearson Correlation, SHAP, Mean Absolute Error, Mean Squared Error, R2 Score

## Code
DeepNote, VSCode, PyCharm

# Citations

[1] JEI. “Using Machine Learning to Develop a Global Coral Bleaching Predictor | Journal of Emerging Investigators.” Emerginginvestigators.org, emerginginvestigators.org/articles/22-056. Accessed 3 Mar. 2024.
