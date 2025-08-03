#Note: Downloaded this and uploaded to S3 bucket at: s3://dockerevalcontainer/processing/input/code/

import tarfile
import pandas as pd
import xgboost as xgb
import os

#S3 paths, aligned to Docker Container
model_tar_path = '/opt/ml/processing/input/model/model.tar.gz'
output_csv_path = '/opt/ml/processing/output/final_with_shap.csv'
data_dir = '/opt/ml/processing/input/data'
target_file = 'sample_data_output.csv' #Hard coding our input file to ensure it doesn't pick up any test  files or other files
input_csv_path = os.path.join(data_dir, target_file)
if not os.path.exists(input_csv_path):
    raise FileNotFoundError(f"{target_file} not found in {data_dir}. Files present: {os.listdir(data_dir)}")


# Cols list
feature_cols = [
    'Encoded-animal_type',
    'Encoded-primary_breed_harmonized',
    'Encoded-primary_color_harmonized',
    'Encoded-sex',
    'Encoded-intake_type_harmonized',
    'Encoded-Is_returned',
    'Encoded-has_name',
    'Encoded-is_mix',
    'age_months',
    'Num_returned',
    'stay_length_days',
    'min_height',
    'max_height',
    'min_weight',
    'max_weight',
    'min_expectancy',
    'max_expectancy',
    'grooming_frequency_value',
    'shedding_value',
    'energy_level_value',
    'trainability_value',
    'demeanor_value'
]

# Model extract
with tarfile.open(model_tar_path) as tar:
    tar.extractall(path='./model')
model_file = './model/xgboost-model'

#load model
booster = xgb.Booster()
booster.load_model(model_file)

# load scored data

df = pd.read_csv(input_csv_path)
print(f"Loaded scored data with shape: {df.shape}")

# Check features
missing = [col for col in feature_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required features in scored file: {missing}")

# Compute shap
dmatrix = xgb.DMatrix(df[feature_cols])
shap_values = booster.predict(dmatrix, pred_contribs=True)

# Create SHAP DataFrame
shap_df = pd.DataFrame(shap_values, columns=feature_cols + ['bias']).drop(columns=['bias'])

# Rename SHAP cols
rename_map = {col: f"SHAP-{col.replace('Encoded-', '').replace('_', ' ').title()}" for col in feature_cols}
shap_df.rename(columns=rename_map, inplace=True)

# Append SHAP columns to original df
df = pd.concat([df, shap_df], axis=1)

# Compute top pos/negative shap

pos1, pos2, pos3, neg1, neg2, neg3 = [], [], [], [], [], []

for i in range(shap_df.shape[0]):
    row = shap_df.iloc[i]

    # Sort positive and negative SHAP 
    pos_sorted = row[row > 0].sort_values(ascending=False)
    neg_sorted = row[row < 0].sort_values(ascending=True)  # ascending for negative (most negative first)

    # Format as "Feature (value)"
    p1 = f"{pos_sorted.index[0]} ({pos_sorted.iloc[0]:.4f})" if len(pos_sorted) > 0 else "None"
    p2 = f"{pos_sorted.index[1]} ({pos_sorted.iloc[1]:.4f})" if len(pos_sorted) > 1 else "None"
    p3 = f"{pos_sorted.index[2]} ({pos_sorted.iloc[2]:.4f})" if len(pos_sorted) > 2 else "None"

    n1 = f"{neg_sorted.index[0]} ({neg_sorted.iloc[0]:.4f})" if len(neg_sorted) > 0 else "None"
    n2 = f"{neg_sorted.index[1]} ({neg_sorted.iloc[1]:.4f})" if len(neg_sorted) > 1 else "None"
    n3 = f"{neg_sorted.index[2]} ({neg_sorted.iloc[2]:.4f})" if len(neg_sorted) > 2 else "None"

    pos1.append(p1)
    pos2.append(p2)
    pos3.append(p3)
    neg1.append(n1)
    neg2.append(n2)
    neg3.append(n3)

# Add feature columns
df['Positive_Feature_1'] = pos1
df['Positive_Feature_2'] = pos2
df['Positive_Feature_3'] = pos3
df['Negative_Feature_1'] = neg1
df['Negative_Feature_2'] = neg2
df['Negative_Feature_3'] = neg3

# Split combined into feature and value
for col in ['Positive_Feature_1', 'Positive_Feature_2', 'Positive_Feature_3',
            'Negative_Feature_1', 'Negative_Feature_2', 'Negative_Feature_3']:
    df[[col, f"{col}_Value"]] = df[col].str.extract(r'^(.*?)\s*\((-?\d+\.\d+)\)$')
    df[f"{col}_Value"] = pd.to_numeric(df[f"{col}_Value"], errors='coerce')


print("output directory contents:", os.listdir('/opt/ml/processing/output/'))

df.to_csv(output_csv_path, index=False)

print(f" SHAP file saved as: {output_csv_path}")

