import pandas as pd 
df=pd.read_csv("data/raw/student_data.csv")

print("Shape: ", df.shape)
print("\nPlacement Distribution:")
print(df["Placed"].value_counts())

print("Correlartion with placement:")
print(df.corr()["placed"].sort_values(ascending=False))

# Basic validation rules

assert df.isnull().sum().sum()==0
assert df["cgpa"].between(5.0,10.0).all()
assert (df.loc[df["placed"]==0,"salary_lpa"]==0).all()

# Save processed copy (no transformation yet)

df.to_csv("data/processed/student_data_clean.csv",index=False)
print("\nProcessed data saved")