"""
Given a non-empty list of integers nums and an integer k, return the k most frequent elements.
Your solution should be optimized for both time and space complexity.
"""

from typing import List


def most_frequent(nums: List[int], k: int):

    frequency = {}

    for item in nums:
        if item not in frequency:
            frequency[item] = 1
        else:
            frequency[item] += 1

    sorted_frequency = sorted(frequency, reverse=True, key=lambda x: frequency[x])

    return sorted_frequency[:k]


"""
Given a string, find the first non-repeating character in it and return its index.
If it doesn't exist, return -1.
"""


def non_repeating(s: str):

    frequency = {}

    for item in s:
        if item not in frequency:
            frequency[item] = 1
        else:
            frequency[item] += 1

    for i in frequency:
        if frequency[i] == 1:
            return i


customers = [
    {"id": "customer_1", "startDate": "2021-01-04", "endDate": "2021-02-07"},
    {"id": "customer_2", "startDate": "2021-01-04", "endDate": "2021-09-07"},
    {"id": "customer_3", "startDate": "2021-01-14", "endDate": "2021-03-07"},
    {"id": "customer_4", "startDate": "2021-02-15", "endDate": None},
]
from datetime import datetime


def update_cohort_retention(customers):

    count_map = {item["startDate"][:7]: {x: 0 for x in ['3', '6', '9']} for item in customers}
    for customer in customers:
        if customer["endDate"] and customer["startDate"]:
            start_date = datetime.strptime(customer["startDate"], "%Y-%m-%d")
            end_date = datetime.strptime(customer["endDate"], "%Y-%m-%d")
            time_delta = end_date - start_date
            days = time_delta.days
            months = days / 30 
        else:
            current_date = datetime.today()
            days = (current_date - start_date).days
            months = days / 30
            print(months)
        temp = customer["startDate"][:7]

        if months <= 3:
            count_map[temp]["3"] += 1
        elif months <= 6:
            count_map[temp]["6"] += 1
            count_map[temp]["3"] += 1

        else:
            count_map[temp]["3"] += 1
            count_map[temp]["6"] += 1
            count_map[temp]["9"] += 1

    return count_map

from datetime import datetime

def update_cohort_retention(customers):
    # Initialize a dictionary for cohort data
    cohort_data = {}

    for customer in customers:
        # Parse start and end dates
        start_date = datetime.strptime(customer["startDate"], "%Y-%m-%d")
        cohort_month = start_date.strftime("%Y-%m")

        # Handle active customers (endDate is None)
        if customer["endDate"]:
            end_date = datetime.strptime(customer["endDate"], "%Y-%m-%d")
        else:
            end_date = datetime.today()

        # Calculate the number of months the customer was active
        months_active = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        # Ensure the cohort month exists in the dictionary
        if cohort_month not in cohort_data:
            cohort_data[cohort_month] = {"3": 0, "6": 0, "9": 0, "total": 0}
        # Increment total customers for the cohort
        cohort_data[cohort_month]["total"] += 1

        # Update retention metrics
        if months_active >= 3:
            cohort_data[cohort_month]["3"] += 1
        if months_active >= 6:
            cohort_data[cohort_month]["6"] += 1
        if months_active >= 9:
            cohort_data[cohort_month]["9"] += 1

    # Calculate retention rates as fractions
    retention_rates = {}
    for month, data in cohort_data.items():
        total = data["total"]
        retention_rates[month] = {
            "3": round(data["3"] / total, 2) if total > 0 else 0,
            "6": round(data["6"] / total, 2) if total > 0 else 0,
            "9": round(data["9"] / total, 2) if total > 0 else 0,
        }

    return retention_rates




if __name__ == "__main__":
    # print(non_repeating("releveler"))


    # Example usage
    customers = [
    {"id": "customer_1", "startDate": "2021-01-04", "endDate": "2021-02-07"},
    {"id": "customer_2", "startDate": "2021-01-04", "endDate": "2021-09-07"},
    {"id": "customer_3", "startDate": "2021-01-14", "endDate": "2021-03-07"},
    {"id": "customer_4", "startDate": "2021-02-15", "endDate": None},
    ]
    cohort_retention = update_cohort_retention(customers)
    print(cohort_retention)