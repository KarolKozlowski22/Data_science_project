from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def classify_evalute(train_images_flat, train_labels, test_images_flat, test_labels):
    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images_flat)
    test_images_scaled = scaler.transform(test_images_flat)

    model = LogisticRegression(max_iter=100, verbose=1, solver='saga')
    model.fit(train_images_scaled, train_labels)
    
    predictions = model.predict(test_images_scaled)
    print(classification_report(test_labels, predictions))
    with open('classification_report.md', 'w') as file:
        file.write(classification_report(test_labels, predictions))