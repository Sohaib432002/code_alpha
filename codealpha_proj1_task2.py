import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os



dataset_path = 'D:/Python/dataset_codeaplpha/Training'


face_images = []
labels = []

# Load images and labels
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(root, file)
            label = os.path.basename(root)  # Assuming each subdirectory is a different person
            face_images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            labels.append(label)



# Preprocessing - Resize images and prepare data for training

face_images_resized=[]
for image in face_images:
    face_images_resized.append(cv2.resize(image, (200, 200)))
face_images_resized = np.array(face_images_resized)
labels = np.array(labels)

# Feature Extraction - Extract features using a simple method (e.g., flatten pixels)
X = face_images_resized.reshape(len(face_images_resized), -1)
y = labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the KNN classifier: {accuracy}")


RFC = RandomForestClassifier(n_estimators=100,max_depth=10,max_features='sqrt',bootstrap=True)
RFC.fit(X_train, y_train)
# Step 5: Model Evaluation - Evaluate the model
y_pred = RFC.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the RandomForest classifier: {accuracy}")
from sklearn.model_selection import GridSearchCV


"""
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
# Step 5: Model Evaluation - Evaluate the model
y_pred = DTC.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Dession Tree classifier: {accuracy}")

"""


print('Using RandomForestClassifer Model')
def recognize_faces(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (200, 200))
    img_flattened = img_resized.reshape(1, -1)
    predicted_label = RFC.predict(img_flattened)
    return predicted_label[0]

# Example usage:
test_image_path = input("Enter the Path : ")
predicted_person = recognize_faces(test_image_path)
print(f"Predicted person in the image: {predicted_person}")

image = cv2.imread(test_image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
cv2.imshow(predicted_person, image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()