import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def main():
    data_file = Path("data") / "student_results.csv"
    output_folder = Path("outputs")
    output_file = output_folder / "deep_learning_results.csv"

    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(data_file)

    print("Student Result Dataset")
    print("----------------------")
    print(df.head())

    print()
    print("Result Counts")
    print("-------------")
    print(df["Result"].value_counts())

    X = df[["StudyHours", "Attendance", "AssignmentScore"]]

    y = df["Result"].map({
        "Fail": 0,
        "Pass": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(8, activation="relu", input_shape=(3,)),
        Dense(4, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=4,
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test_scaled,
        y_test,
        verbose=0
    )

    probabilities = model.predict(X_test_scaled, verbose=0)

    predicted_classes = (probabilities >= 0.5).astype(int).flatten()

    actual_labels = [
        "Pass" if value == 1 else "Fail"
        for value in y_test
    ]

    predicted_labels = [
        "Pass" if value == 1 else "Fail"
        for value in predicted_classes
    ]

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "Attendance": X_test["Attendance"],
        "AssignmentScore": X_test["AssignmentScore"],
        "ActualResult": actual_labels,
        "PredictedResult": predicted_labels,
        "PassProbability": probabilities.flatten().round(3)
    })

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    print()
    print("Model Evaluation")
    print("----------------")
    print(f"Test accuracy: {accuracy:.2f}")
    print(f"Test loss: {loss:.4f}")

    print()
    print("Classification Report")
    print("---------------------")
    print(classification_report(actual_labels, predicted_labels))

    new_student = pd.DataFrame({
        "StudyHours": [6],
        "Attendance": [78],
        "AssignmentScore": [70]
    })

    new_student_scaled = scaler.transform(new_student)

    new_probability = model.predict(new_student_scaled, verbose=0)[0][0]

    if new_probability >= 0.5:
        new_prediction = "Pass"
    else:
        new_prediction = "Fail"

    print()
    print("New Student")
    print("-----------")
    print(new_student)

    print()
    print(f"Predicted result: {new_prediction}")
    print(f"Probability of Pass: {new_probability * 100:.2f}%")

    results.to_csv(output_file, index=False)

    print()
    print(f"Prediction results saved to: {output_file}")


main()
