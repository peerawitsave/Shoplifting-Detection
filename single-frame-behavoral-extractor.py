import joblib
import pandas as pd

sample = pd.DataFrame([{
    'num_person': 1,
    'num_bag': 1,
    'overlap_count': 2,
    'unique_classes': 3
}])

model = joblib.load('shoplifting_classifier.pkl')

pred = model.predict(sample)
print("Prediction:", "Shoplifting" if pred[0] == 1 else "Normal")
