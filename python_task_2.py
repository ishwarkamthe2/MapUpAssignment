#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[2]:


import pandas as pd

def calculate_distance_matrix(dataset_path):
    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Create a DataFrame with unique IDs
    unique_ids = pd.Index(df['id_start'].append(df['id_end']).unique(), name='id')
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

    # Initialize the matrix with zeros
    distance_matrix = distance_matrix.fillna(0)

    # Update the matrix with cumulative distances
    for index, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] += distance
        distance_matrix.at[id_end, id_start] += distance  # Accounting for bidirectional distances

    return distance_matrix

dataset_path = 'dataset-3.csv'
resulting_matrix = calculate_distance_matrix(dataset_path)

# Print the resulting matrix
print(resulting_matrix)


# # Question 2
# 

# In[4]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    rows = []
    for i in range(len(distance_matrix.index)):
        for j in range(i+1, len(distance_matrix.columns)):
            id_start = distance_matrix.index[i]
            id_end = distance_matrix.columns[j]
            distance = distance_matrix.iloc[i, j]
            rows.append([id_start, id_end, distance])
            
    return pd.DataFrame(rows, columns=['id_start', 'id_end', 'distance'])

# Example usage:
# Assuming distance_matrix is the DataFrame from Question 1
# Replace it with your actual DataFrame name
result_df = unroll_distance_matrix(resulting_matrix)
print(result_df)


# # Qusetion 3

# In[31]:


import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Calculate average distance for the reference value
    reference_avg_distance = df[df['id_start'] == reference_value]['distance'].mean()
    
    # Calculate the lower and upper bounds within 10% of the average distance
    lower_bound = reference_avg_distance - (0.10 * reference_avg_distance)
    upper_bound = reference_avg_distance + (0.10 * reference_avg_distance)
    
    print(f"Reference Value: {reference_value}")
    print(f"Reference Avg Distance: {reference_avg_distance}")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    
    # Filter the DataFrame to include rows within the specified range
    filtered_df = df[(df['id_start'] != df['id_end']) & (df['id_start'] == reference_value)
                     & (df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
    
    # Return a sorted list of values from the id_start column
    result_ids = sorted(filtered_df['id_start'].unique())
    
    return result_ids

# Example usage:
# Assuming result_df is the DataFrame from Question 2
# Replace it with your actual DataFrame name
reference_value = 1001436  # Replace with the desired reference value
result_ids = find_ids_within_ten_percentage_threshold(result_df, reference_value)
print(result_ids)


# # Question 4

# In[12]:


import pandas as pd

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type and add columns to the DataFrame
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    # Drop the 'distance' column
    df = df.drop(columns=['distance'])

    return df

# Example usage:
# Assuming result_df is the DataFrame from Question 2
# Replace it with your actual DataFrame name
result_with_toll_rate_df = calculate_toll_rate(result_df)
print(result_with_toll_rate_df)


# # Question 5

# In[32]:


import pandas as pd
from datetime import datetime, timedelta

def calculate_time_based_toll_rates(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df_with_toll = input_df.copy()

    # Define time ranges
    weekday_morning_range = (datetime.strptime('00:00:00', '%H:%M:%S').time(), datetime.strptime('10:00:00', '%H:%M:%S').time())
    weekday_afternoon_range = (datetime.strptime('10:00:00', '%H:%M:%S').time(), datetime.strptime('18:00:00', '%H:%M:%S').time())
    weekday_evening_range = (datetime.strptime('18:00:00', '%H:%M:%S').time(), datetime.strptime('23:59:59', '%H:%M:%S').time())
    weekend_range = (datetime.strptime('00:00:00', '%H:%M:%S').time(), datetime.strptime('23:59:59', '%H:%M:%S').time())

    # Define discount factors
    weekday_morning_factor = 0.8
    weekday_afternoon_factor = 1.2
    weekday_evening_factor = 0.8
    weekend_factor = 0.7

    # Iterate over rows and apply discount factors based on time ranges
    for index, row in result_df_with_toll.iterrows():
        # Get the start time of the trip
        start_time = datetime.strptime(row['startTime'], '%H:%M:%S').time()

        # Determine the day type (weekday or weekend)
        if row['startDay'] in ['Saturday', 'Sunday']:
            # Weekend
            discount_factor = weekend_factor
        else:
            # Weekday
            if weekday_morning_range[0] <= start_time <= weekday_morning_range[1]:
                discount_factor = weekday_morning_factor
            elif weekday_afternoon_range[0] <= start_time <= weekday_afternoon_range[1]:
                discount_factor = weekday_afternoon_factor
            elif weekday_evening_range[0] <= start_time <= weekday_evening_range[1]:
                discount_factor = weekday_evening_factor
            else:
                discount_factor = 1.0  # Default to no discount for times outside defined ranges

        # Apply discount factor to toll rates
        result_df_with_toll.at[index, 'moto'] *= discount_factor
        result_df_with_toll.at[index, 'car'] *= discount_factor
        result_df_with_toll.at[index, 'rv'] *= discount_factor
        result_df_with_toll.at[index, 'bus'] *= discount_factor
        result_df_with_toll.at[index, 'truck'] *= discount_factor

    # Add new columns for start_day, start_time, end_day, and end_time
    result_df_with_toll['start_day'] = result_df_with_toll['startDay']
    result_df_with_toll['start_time'] = result_df_with_toll['startTime'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())
    result_df_with_toll['end_day'] = result_df_with_toll['endDay']
    result_df_with_toll['end_time'] = result_df_with_toll['endTime'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())

    return result_df_with_toll


result_df_with_toll = calculate_time_based_toll_rates(result_df)
print(result_df_with_toll)


# In[ ]:





# In[ ]:




