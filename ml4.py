
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
emails = [
    "Win free money now",
    "You won a lottery prize",
    "Claim your free gift",
    "Please submit your assignment",
    "Meeting is tomorrow",
    "Send me your project report"
]

labels = [1, 1, 1, 0, 0, 0]  # 1=Spam, 0=Not Spam

# Convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(emails)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test email
email = ["Congratulations you won free money"]
result = model.predict(cv.transform(email))

print("Email:", email[0])
print("Prediction:", "Spam" if result[0] == 1 else "Not Spam")
