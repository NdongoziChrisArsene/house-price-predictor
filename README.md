# House Price Predictor

A small web app that estimates the price of a house in Rwanda (in million RWF) from its size, rooms, age, distance to the city centre, parking spaces and neighbourhood. It was built for the Machine Learning assignment at Rwanda Polytechnic, Kigali College (ETT Y4 BTech).

**Live app:** PASTE YOUR STREAMLIT LINK HERE

## Dataset
`house_price_prediction_dataset.csv` holds 125 past sales (prices in million RWF). The raw file had missing values, duplicate rows and impossible values (for example a 1,200 m² house and an 85 km distance). After cleaning, 117 rows remain in `cleaned_house_price_dataset.csv`.

| Column | Meaning |
|---|---|
| Area_m2 | Floor area in square metres |
| Bedrooms, Bathrooms | Room counts |
| House_Age_Years | Age of the house |
| Distance_to_City_km | Distance from the city centre |
| Parking_Spaces | Number of parking spaces |
| Neighborhood | Gasabo, Huye, Kicukiro, Kigali City, Musanze, Nyarugenge |
| House_Price_Million_RWF | Target: sale price |

## How the model was built
1. Cleaned the data: dropped rows with no price, removed 5 duplicate rows, dropped 2 rows with impossible prices, and blanked impossible areas and distances.
2. Split the data 80/20 with `random_state=42`.
3. Built one scikit-learn `Pipeline`: median/most-frequent imputation, one-hot encoding of `Neighborhood` (one level dropped), then `LinearRegression`.
4. Saved the whole pipeline with `joblib` as `house_price_model.sav`, so it predicts straight from raw inputs.

## Performance (test set)
| Metric | Value |
|---|---|
| R² | 0.82 |
| RMSE | 23.3 million RWF |
| MAE | 18.9 million RWF |

Training R² is 0.86, so overfitting is mild. The dataset is small (24 test rows), so treat the numbers as approximate.

## Run locally
```bash
git clone https://github.com/YOUR-USERNAME/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit app
- `house_price_model.sav`: trained pipeline
- `house_price_model.ipynb`: full analysis notebook
- `cleaned_house_price_dataset.csv` and `house_price_prediction_dataset.csv`: cleaned and raw data
- `requirements.txt`: dependencies
- `House_Price_Report.pdf`: written report

## Limitations
Small dataset, Bedrooms and Bathrooms are strongly related, and predictions outside the training ranges (above about 430 m² or over 50 years old) are unreliable.
