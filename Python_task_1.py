#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[6]:


import pandas as pd

def generate_car_matrix(df):
    # Selecting relevant columns
    car_data = df[['id_1', 'id_2', 'car']].copy()

    # Creating a pivot table with id_1 as index, id_2 as columns, and car as values
    car_matrix = car_data.pivot(index='id_1', columns='id_2', values='car')

    # Filling NaN values with 0 and setting diagonal values to 0
    car_matrix = car_matrix.fillna(0).astype(float)

    return car_matrix

df = pd.read_csv('dataset-1.csv')
result = generate_car_matrix(df)
print(result)


# # Question 2

# In[10]:


import pandas as pd

def get_type_count(df):
    # Add a new categorical column car_type based on values of the column car
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=choices, right=False)

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

df = pd.read_csv('dataset-1.csv')
result = get_type_count(df)
print(result)


# # Question 3
# 

# In[11]:


import pandas as pd

def get_bus_indexes(df):
    # Calculate the mean value of the bus column
    mean_bus = df['bus'].mean()

    # Identify indices where bus values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


df = pd.read_csv("dataset-1.csv")
bus_indices = get_bus_indexes(df)
print(bus_indices)


# # Question 4

# In[19]:


import pandas as pd

def filter_routes(df):
    # Calculate the average of the truck column for each route
    avg_truck_by_route = df.groupby('route')['truck'].mean()
    
    # Filter routes where the average truck value is greater than 7
    selected_routes = avg_truck_by_route[avg_truck_by_route > 7].index
    
    # Sort and return the selected routes as a list
    return sorted(selected_routes)

df = pd.read_csv('dataset-1.csv')

selected_routes = filter_routes(df)
print(selected_routes)


# # Question 5

# In[26]:


import pandas as pd

def multiply_matrix(car_matrix):
    # Apply the specified logic to modify values
    modified_matrix = car_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    
    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)
    
    return modified_matrix

# Example usage:
df = pd.read_csv('dataset-1.csv')
car_matrix = generate_car_matrix(df)
result_matrix = multiply_matrix(car_matrix)
print(result_matrix)


# In[27]:


result_matrix


# # Question 6

# In[34]:


import pandas as pd

def verify_timestamps(df):
    # Combine 'startDay' and 'startTime' to create 'startTimestamp'
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')

    # Combine 'endDay' and 'endTime' to create 'endTimestamp'
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Check if the timestamps are within a full 24-hour period
    full_day_check = (df['endTimestamp'] - df['startTimestamp']) == pd.Timedelta(days=1)

    # Check if the timestamps span all 7 days of the week
    all_days_check = df.groupby(['id', 'id_2'])['startDay'].nunique() == 7

    # Combine both checks using the same index as the original DataFrame
    result = pd.DataFrame({'full_day_check': full_day_check, 'all_days_check': all_days_check}).all(axis=1)

    return result

# Load your dataset
df = pd.read_csv('dataset-2.csv')

# Apply the function
result_series = verify_timestamps(df)

# Print the result
print(result_series)


# In[ ]:




