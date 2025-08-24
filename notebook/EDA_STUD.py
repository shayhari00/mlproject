import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/stud.csv')

print(df.head())

df.shape
print(df.isna().sum())

print(df.duplicated().sum())

print(df.info())

df.nunique()

print(df.describe())

print("Categories in 'gender' variable:     ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable:  ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental level of education' variable:",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'lunch' variable:     ",end=" " )
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:     ",end=" " )
print(df['test_preparation_course'].unique())


df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score']/3
df.head()

reading_full = df[df['reading_score'] == 100]['average'].count()
writing_full = df[df['writing_score'] == 100]['average'].count()
math_full = df[df['math_score'] == 100]['average'].count()

print(f'Number of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')

reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20 = df[df['math_score'] <= 20]['average'].count()

print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')


fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',kde=True,hue='gender')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='total score',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='total score',kde=True,hue='gender')
plt.show()

plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
sns.histplot(data=df,x='average',kde=True,hue='lunch')
plt.subplot(142)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
plt.subplot(143)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
plt.show()

plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
ax = sns.histplot(
    data=df,
    x='average',
    kde=True,
    hue='parental_level_of_education'
)

plt.title("Distribution of Average Scores by Parental Education Level")
plt.show()
plt.subplot(1, 2, 1)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

ax = sns.histplot(
    data=df[df.gender == 'male'],
    x='average',
    kde=True,
    hue='parental_level_of_education'
)

plt.title("Male")
plt.show()

# Second subplot → Female
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

plt.figure(figsize=(12,5))

# Male
plt.subplot(1, 2, 1)
sns.histplot(
    data=df[df.gender == 'male'],
    x='average',
    kde=True,
    hue='parental_level_of_education'
)
plt.title("Male")

# Female
plt.subplot(1, 2, 2)
sns.histplot(
    data=df[df.gender == 'female'],
    x='average',
    kde=True,
    hue='parental_level_of_education'
)
plt.title("Female")

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()

plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

plt.figure(figsize=(15, 5))

# Plot 1: All students
plt.subplot(1, 3, 1)
sns.histplot(
    data=df,
    x='average',
    kde=True,
    hue='race_ethnicity'
)
plt.title("All Students")

# Plot 2: Female
plt.subplot(1, 3, 2)
sns.histplot(
    data=df[df.gender == 'female'],
    x='average',
    kde=True,
    hue='race_ethnicity'
)
plt.title("Female Students")


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Clean column names for safety
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

# ---------------- 1. Violin Plots ----------------
plt.figure(figsize=(18, 8))

plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math_score', data=df, color='red', linewidth=3)

plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading_score', data=df, color='green', linewidth=3)

plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing_score', data=df, color='blue', linewidth=3)

plt.subplot(1, 4, 4)
plt.title('AVERAGE SCORES')
sns.violinplot(y='average', data=df, color='purple', linewidth=3)

plt.show()


# ---------------- 2. Pie Plots ----------------
plt.figure(figsize=(30, 12))

plt.subplot(1, 5, 1)
plt.pie(df['gender'].value_counts(),
        labels=['Female', 'Male'],
        colors=['red', 'green'], autopct='%1.2f%%')
plt.title('Gender', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 2)
plt.pie(df['race_ethnicity'].value_counts(),
        labels=df['race_ethnicity'].value_counts().index,
        colors=['red', 'green', 'blue', 'cyan', 'orange'], autopct='%1.2f%%')
plt.title('Race/Ethnicity', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 3)
plt.pie(df['lunch'].value_counts(),
        labels=df['lunch'].value_counts().index,
        colors=['red', 'green'], autopct='%1.2f%%')
plt.title('Lunch', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 4)
plt.pie(df['test_preparation_course'].value_counts(),
        labels=df['test_preparation_course'].value_counts().index,
        colors=['red', 'green'], autopct='%1.2f%%')
plt.title('Test Prep Course', fontsize=20)
plt.axis('off')

plt.subplot(1, 5, 5)
plt.pie(df['parental_level_of_education'].value_counts(),
        labels=df['parental_level_of_education'].value_counts().index,
        colors=['red', 'green', 'blue', 'cyan', 'orange', 'grey'], autopct='%1.2f%%')
plt.title('Parental Education', fontsize=20)
plt.axis('off')

plt.tight_layout()
plt.show()


# ---------------- 3. Gender Countplot + Pie ----------------
f, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.countplot(x='gender', data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=20)

ax[1].pie(df['gender'].value_counts(), labels=['Female', 'Male'],
          explode=[0, 0.1], autopct='%1.1f%%', shadow=True,
          colors=['#ff4d4d', '#ff8000'])
plt.show()


# ---------------- 4. Gender Group Bar ----------------
gender_group = df.groupby('gender').mean(numeric_only=True)

plt.figure(figsize=(10, 8))
X = ['Total Average', 'Math Average']

female_scores = [gender_group.loc['female', 'average'], gender_group.loc['female', 'math_score']]
male_scores = [gender_group.loc['male', 'average'], gender_group.loc['male', 'math_score']]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, male_scores, 0.4, label='Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label='Female')

plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average vs Math average marks by gender", fontweight='bold')
plt.legend()
plt.show()


# ---------------- 5. Race/Ethnicity ----------------
f, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.countplot(x='race_ethnicity', data=df, palette='bright', ax=ax[0], saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=20)

ax[1].pie(df['race_ethnicity'].value_counts(), labels=df['race_ethnicity'].value_counts().index,
          explode=[0.1, 0, 0, 0, 0], autopct='%1.1f%%', shadow=True)
plt.show()


# ---------------- 6. Race/Ethnicity Score Means ----------------
Group_data2 = df.groupby('race_ethnicity')

f, ax = plt.subplots(1, 3, figsize=(20, 8))

sns.barplot(x=Group_data2['math_score'].mean().index,
            y=Group_data2['math_score'].mean().values,
            palette='mako', ax=ax[0])
ax[0].set_title('Math score', color='#005ce6', size=20)

for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=15)

sns.barplot(x=Group_data2['reading_score'].mean().index,
            y=Group_data2['reading_score'].mean().values,
            palette='flare', ax=ax[1])
ax[1].set_title('Reading score', color='#005ce6', size=20)

for container in ax[1].containers:
    ax[1].bar_label(container, color='black', size=15)

sns.barplot(x=Group_data2['writing_score'].mean().index,
            y=Group_data2['writing_score'].mean().values,
            palette='coolwarm', ax=ax[2])
ax[2].set_title('Writing score', color='#005ce6', size=20)

for container in ax[2].containers:
    ax[2].bar_label(container, color='black', size=15)

plt.show()


# ---------------- 7. Parental Education ----------------
plt.figure(figsize=(15, 9))
plt.style.use('fivethirtyeight')
sns.countplot(x='parental_level_of_education', data=df, palette='Blues')
plt.title('Comparison of Parental Education', fontweight=30, fontsize=20)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.show()

df.groupby('parental_level_of_education').mean(numeric_only=True).plot(kind='barh', figsize=(10, 10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# ---------------- 8. Lunch Comparison ----------------
plt.figure(figsize=(15, 9))
plt.style.use('seaborn-v0_8-talk')
sns.countplot(x='lunch', data=df, palette='PuBu')
plt.title('Comparison of Lunch Types', fontweight=30, fontsize=20)
plt.xlabel('Lunch Type')
plt.ylabel('Count')
plt.show()


# ---------------- 9. Test Prep vs Parental Education + Lunch ----------------
f, ax = plt.subplots(1, 2, figsize=(20, 8))

sns.countplot(x='parental_level_of_education', data=df, palette='bright',
              hue='test_preparation_course', saturation=0.95, ax=ax[0])
ax[0].set_title('Students vs Test Prep Course', color='black', size=25)
for container in ax[0].containers:
    ax[0].bar_label(container, color='black', size=20)

sns.countplot(x='parental_level_of_education', data=df, palette='bright',
              hue='lunch', saturation=0.95, ax=ax[1])
ax[1].set_title('Students vs Lunch Type', color='black', size=25)
for container in ax[1].containers:
    ax[1].bar_label(container, color='black', size=20)

plt.show()


# ---------------- 10. Lunch & Scores ----------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.barplot(x='lunch', y='math_score', hue='test_preparation_course', data=df)

plt.subplot(2, 2, 2)
sns.barplot(x='lunch', y='reading_score', hue='test_preparation_course', data=df)

plt.subplot(2, 2, 3)
sns.barplot(x='lunch', y='writing_score', hue='test_preparation_course', data=df)

plt.tight_layout()
plt.show()


# ---------------- 11. Boxplots ----------------
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
sns.boxplot(x=df['math_score'], color='skyblue')

plt.subplot(1, 4, 2)
sns.boxplot(x=df['reading_score'], color='hotpink')

plt.subplot(1, 4, 3)
sns.boxplot(x=df['writing_score'], color='yellow')

plt.subplot(1, 4, 4)
sns.boxplot(x=df['average'], color='lightgreen')

plt.tight_layout()
plt.show()


# ---------------- 12. Pairplot ----------------
sns.pairplot(df, hue='gender')
plt.show()
