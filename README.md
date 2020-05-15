# nd025-capstone-starbucks
This is the Last and final project of the Data scientist nanodegree


# Project Motivation
This is the final project in the Data Scientist nanodegree by Udacity,
It's about analysing simulated data from starbucks, and try to predict
how a customer will react to an offer based on demographic and past
transaction data.

The blog post for this project can be found here: https://medium.com/@henriksvensson_1896/starbucks-analysis-of-simulated-data-c033e210a4d1?sk=e35372e75fdaa37d7b15b5cdeb3fce51

# File Descriptions
This repo contaians the following files:

.\
├── helper_functions.py                 # All the helper functions used\
├── Starbucks_Capstone_notebook.ipynb   # The main notebook \
├── README.md                           # This file\
└── data                    # Datasets\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── person_offer.csv    # Dataframe with person and offer per row\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── portfolio.json      # Containing offer ids and meta data about each offer (duration, type, etc.)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── profile.json        # Demographic data for each customer\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── transcript.json     # Records for transactions, offers received, offers viewed, and offers completed\

# How to Interact with your project
Clone the project and open the Notebook for further instructions.

# Required libraries
* pandas==1.0.3
* tqdm==4.46.0
* scikit-learn==0.22.2.post1
* xgboost==1.0.2
* scipy==1.4.1

# Licensing, Authors, Acknowledgements, etc.
Thanks to Starbucks for the data, and udacity for a great nanodegree.
