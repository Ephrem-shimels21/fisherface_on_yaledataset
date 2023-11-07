from dataLoad import process_data
from BaseModel import FisherfaceModel


train_data, train_label, test_data, test_label = process_data()

model = FisherfaceModel(train_data, train_label)

corrects, total = 0, 0
for index, test in enumerate(test_data):
    if test_label[index] == model.predict(test_data[index]):
        corrects += 1
    total += 1
    print(
        "Expected =",
        test_label[index],
        " / Predicted =",
        model.predict(test_data[index]),
    )

print(f"Accuracy, {corrects / total * 100 }%")
