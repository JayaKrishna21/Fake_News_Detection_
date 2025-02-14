import os
import pandas as pd
from src.utils import load_object

# File paths
model_path = os.path.join('artifacts', 'model.pkl')
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

# ✅ Load the model and preprocessor
model = load_object(model_path)
preprocessor = load_object(preprocessor_path)

# ✅ Generate test samples
sample_data = pd.DataFrame({
    "news": [
        "Government announces new policies for healthcare",
        "Aliens have landed on Earth, scientists confirm",
        "Stock markets hit record high after economic growth",
        "Man claims he traveled through time and met dinosaurs",
        "New AI model outperforms humans in medical diagnosis",
        "Fake news spreads about celebrity scandal",
        "Study finds coffee extends life expectancy",
        "Breaking: Moon is made of cheese, says viral post",
        "NASA confirms successful Mars landing",
        "Secret pyramid discovered in Antarctica"
    ]
})

# ✅ Transform data
X_test = preprocessor.transform(sample_data['news'])

# ✅ Make predictions
predictions = model.predict(X_test)

# ✅ Add predictions to the DataFrame
sample_data['Prediction'] = predictions
sample_data['Label'] = sample_data['Prediction'].apply(lambda x: "REAL" if x == 1 else "FAKE")

# ✅ Filter and print samples predicted as REAL
real_news = sample_data[sample_data['Prediction'] == 1]
print("🔍 Samples predicted as REAL:\n", real_news)

# ✅ Save REAL samples to CSV for further analysis
real_news.to_csv('real_predictions.csv', index=False)

print("✅ REAL predictions saved to 'real_predictions.csv'")
