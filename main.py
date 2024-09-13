"""Customer feedback is collected via a survey or form, saved in a CSV file including the following
	•	CustomerID: A unique identifier for each customer.
	•	SatisfactionScore: A rating from 1 to 10 on the customer’s satisfaction.
	•	FeatureRequests: A text field where customers can suggest features they want.
	•	Complaints: A text field where customers mention any issues or negative experiences.
	•	Comments: A general feedback field."""

#CustomerID, SatisfactionScore, FeatureRequests, Complaints, Comments
#1,9,"Faster payment processing","No issues","Great service"
#2,6,"Mobile app integration","Slow response times",""
#3,4,"More payment methods","Unreliable service","Poor experience"

import pandas as pd
import re
import cohere
import os

co = cohere.Client('O2t727l4FVv5GzufLEuYqDpi2oNAFRKrjhXB2ZVJ') # Always hide API key

data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'SatisfactionScore': [9, 6, 4, 8, 5],
    'FeatureRequests': [
        "Faster payment processing",
        "Speed up payments",
        "More payment methods",
        "Mobile app integration",
        "Support for more payment types"
    ],
    'Complaints': [
        "Slow customer service",
        "Customer service response time too long",
        "Unreliable service",
        "Payment process takes too long",
        "Long wait for customer service"
    ]
}

feedback_data = pd.DataFrame(data)
print(feedback_data.head())

average_satisfaction = feedback_data["SatisfactionScore"].mean()
print(f"Average satisfaction score (1-10): {average_satisfaction}\n")


complaint_keywords = {
    'Customer Service': ['customer service', 'response time', 'wait'],
    'Payment Process': ['payment process', 'payment takes', 'billing'],
    'Service Reliability': ['unreliable', 'connection', 'down']
}

# Define keywords for common feature request categories
feature_keywords = {
    'Faster Payments': ['faster payment', 'speed up', 'quick payment'],
    'More Payment Methods': ['more payment methods', 'payment types', 'support payments'],
    'Mobile App': ['mobile app', 'app integration', 'mobile support']
}

def categorize_complaint(complaint):
    complaint = complaint.lower()
    for category, words in complaint_keywords.items():
        for word in words:
            if re.search(word, complaint):
                return category
    return 'Other'

def categorize_feature_request(feature_request):
    feature_request = feature_request.lower()
    for category, words in feature_keywords.items():
        for word in words:
            if re.search(word, feature_request):
                return category
    return 'Other'

feedback_data['ComplaintCategory'] = [categorize_complaint(complaint) for complaint in feedback_data['Complaints']]
feedback_data['FeatureRequestCategory'] = [categorize_feature_request(request) for request in feedback_data['FeatureRequests']]

complaint_category_counts = feedback_data['ComplaintCategory'].value_counts()
feature_category_counts = feedback_data['FeatureRequestCategory'].value_counts()

print("\nFeature Requests Categorized by Themes:")
print(feature_category_counts)
print("Complaints Categorized by Themes:")
print(complaint_category_counts)

# Example of a complaint summary the team at RevSend wrote up
complaint_summary = """
We have identified the following complaint categories:
1. Customer Service Issues - 3 occurrences
2. Payment Process Delays - 1 occurrences
3. Service Reliability Issues - 1 occurrences

Feature request categories:
1. Faster Payments - 2 requests
2. Mobile App Integration - 1 requests
3. More Payment Methods - 1 requests
"""

prompt = f"""
Based on the following summary of customer complaints and feature requests:
{complaint_summary}

Can you suggest a timeline to prioritize and address these issues in a practical manner for the RevSend team? 
Please include recommendations for which issues should be addressed first and how long each might take to implement.
"""

response = co.generate(
    model='command-xlarge-nightly',  # Choose an appropriate model
    prompt=prompt,
    max_tokens=150,
    temperature=0.7
)

print("\nCohere's suggestions for a timeline:")
print(response.generations[0].text)


