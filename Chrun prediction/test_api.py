import requests
import json

def get_user_input():
    print("\n=== Customer Churn Prediction System ===")
    try:
        credit_score = int(input("Enter Credit Score (300-850): "))
        age = int(input("Enter Age: "))
        tenure = int(input("Enter Tenure (years): "))
        balance = float(input("Enter Balance: "))
        num_products = int(input("Enter Number of Products (1-4): "))
        has_credit_card = int(input("Has Credit Card? (1 for Yes, 0 for No): "))
        is_active = int(input("Is Active Member? (1 for Yes, 0 for No): "))
        salary = float(input("Enter Estimated Salary: "))
        gender = int(input("Enter Gender (1 for Male, 0 for Female): "))
        
        # Geography input
        print("\nSelect Geography:")
        print("1. France")
        print("2. Germany")
        print("3. Spain")
        geo_choice = int(input("Enter choice (1-3): "))
        
        # Convert geography choice to one-hot encoding
        geography = {
            'Geography_France': 1 if geo_choice == 1 else 0,
            'Geography_Germany': 1 if geo_choice == 2 else 0,
            'Geography_Spain': 1 if geo_choice == 3 else 0
        }

        # Create customer dictionary
        customer_data = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_credit_card,
            'IsActiveMember': is_active,
            'EstimatedSalary': salary,
            'Gender': gender,
            **geography  # Unpack geography dictionary
        }

        return customer_data

    except ValueError as e:
        print(f"\nError: Please enter valid numeric values. {str(e)}")
        return None

def predict_churn():
    # Get user input
    customer_data = get_user_input()
    
    if customer_data is None:
        return
    
    try:
        # Make prediction request
        response = requests.post('http://localhost:5000/predict', 
                               json=customer_data)
        
        # Print results
        result = response.json()
        print("\n=== Prediction Results ===")
        print(f"Churn Probability: {result['churn_probability']:.2%}")
        print(f"Likely to Churn: {'Yes' if result['likely_to_churn'] else 'No'}")
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server. Make sure the server is running.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    while True:
        predict_churn()
        if input("\nWould you like to make another prediction? (y/n): ").lower() != 'y':
            break