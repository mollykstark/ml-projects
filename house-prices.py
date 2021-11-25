import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics

sales = pd.read_csv('home_data.csv') 
sales.head()

num_rows = len(sales)

y = sales['price']

num_cols = sales.columns
num_inputs = len(num_cols) - 1

sales = sales[(sales['bedrooms'] == 3)]
price_3_bed = sales['price']
avg_price_3_bed = price_3_bed.mean()

total_sales = len(sales)
filtered_sales = sales[(sales.sqft_living >= 2000) & (sales.sqft_living < 4000)]
num_sqft = len(filtered_sales)
percent_q3 = num_sqft/total_sales

train_data, test_data = train_test_split(sales, test_size=0.2)

plt.scatter(train_data['sqft_living'], train_data['price'], marker='+', label='Train')
plt.scatter(test_data['sqft_living'], test_data['price'], marker='.', label='Test')

plt.legend()
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()

basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = basic_features + [
    'condition',      # condition of the house
    'grade',          # measure of qality of construction
    'waterfront',     # waterfront property
    'view',           # type of view
    'sqft_above',     # square feet above ground
    'sqft_basement',  # square feet in basementab
    'yr_built',       # the year built
    'yr_renovated',   # the year renovated
    'lat',            # the longitude of the parcel
    'long',           # the latitide of the parcel
    'sqft_living15',  # average sq.ft. of 15 nearest neighbors
    'sqft_lot15',     # average lot size of 15 nearest neighbors
]

# basic model
x = train_data[basic_features]
y = train_data['price']
basic_model = LinearRegression().fit(x, y)

# advanced model
a = train_data[advanced_features]
b = train_data['price']
advanced_model = LinearRegression().fit(a, b)
print(type(train_data[advanced_features]))

# basic
basic_prediction = basic_model.predict(x)
train_rmse_basic = np.sqrt(metrics.mean_squared_error(y, basic_prediction))

# advanced
advanced_prediction = advanced_model.predict(a)
train_rmse_advanced = np.sqrt(metrics.mean_squared_error(b, advanced_prediction))

# basic
basic_test_pred = basic_model.predict(test_data[basic_features])
test_rmse_basic = np.sqrt(metrics.mean_squared_error(test_data['price'], basic_test_pred))

# advanced
advanced_test_pred = advanced_model.predict(test_data[advanced_features])
test_rmse_advanced = np.sqrt(metrics.mean_squared_error(test_data['price'], advanced_test_pred))