import random
import pandas as pd

spam_phrases = [
    "URGENT: Claim your $1000 prize now! Call 09061743832",
    "Congratulations! You won a FREE lottery ticket. Visit http://win-now.xyz",
    "Alert: Your account has been suspended. Click here to verify http://bank-update.cc",
    "Earn $500 daily working from home! WhatsApp +1234567890",
    "Final reminder: 50% discount expires today! Click http://deal-grab.biz",
]

ham_phrases = [
    "Hey, are we still meeting for lunch today at 1 PM?",
    "Please find attached the notes for theory of computation lecture.",
    "Can you share the assignment submission link?",
    "Call me once you reach the campus.",
    "The meeting has been rescheduled to tomorrow morning at 10 AM.",
]

data = []
for i in range(5000):
    if random.random() < 0.25:  # ~25% spam
        text = random.choice(spam_phrases) + f" [Ref:{random.randint(1000, 9999)}]"
        label = "Spam"
    else:
        text = random.choice(ham_phrases) + f" [ID:{random.randint(1000, 9999)}]"
        label = "Ham"
    data.append({"Message_ID": i + 1, "Text": text, "True_Label": label})

df = pd.DataFrame(data)
df.to_csv("dataset_5000.csv", index=False)
print("5,000-row dataset generated successfully!")
