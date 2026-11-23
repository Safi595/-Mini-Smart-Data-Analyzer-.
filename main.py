import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()


def get_user_numbers(min_count=10):
    """Collects and validates at least `min_count` numerical inputs from the user."""
    numbers = []
    print(f"Please enter at least {min_count} numerical values (type 'done' when finished):")

    while True:
        user_input = input(f"Enter number {len(numbers) + 1}: ").strip()

        if user_input.lower() == "done":
            if len(numbers) >= min_count:
                break
            else:
                print(f"Error: You have only entered {len(numbers)} numbers. Please enter at least {min_count}.")
                continue

        try:
            value = float(user_input)
            numbers.append(value)
        except ValueError:
            print("Invalid input! Please enter a valid number or type 'done'.")

    return np.array(numbers)


def calculate_statistics(data_array):
    """Calculates basic descriptive statistics using NumPy."""
    stats = {
        "mean": float(np.mean(data_array)),
        "median": float(np.median(data_array)),
        "max": float(np.max(data_array)),
        "min": float(np.min(data_array)),
        "sum": float(np.sum(data_array)),
    }
    return stats


def perform_array_operations(data_array, mean_val):
    """Sorts the array in ascending order and filters values greater than the mean."""
    sorted_data = np.sort(data_array)
    above_mean = data_array[data_array > mean_val]
    return sorted_data, above_mean


def get_ai_insight(stats, above_mean_count, total_count):
    """Sends calculated statistics to Groq API and retrieves a simple English explanation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found. Please set it in your .env file."

    client = Groq(api_key=api_key)

    prompt = (
        f"Analyze the following numerical dataset statistics and explain what they mean "
        f"in 3-4 short, simple English sentences:\n"
        f"- Total numbers: {total_count}\n"
        f"- Mean (Average): {stats['mean']:.2f}\n"
        f"- Median: {stats['median']:.2f}\n"
        f"- Minimum: {stats['min']:.2f}\n"
        f"- Maximum: {stats['max']:.2f}\n"
        f"- Sum: {stats['sum']:.2f}\n"
        f"- Count of numbers above the mean: {above_mean_count}"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data analyst who explains statistics in simple, clear English.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"API Request Failed: {e}"


def main():
    print("=== Mini Smart Data Analyzer ===")
    data_array = get_user_numbers(min_count=10)

    stats = calculate_statistics(data_array)
    sorted_data, above_mean = perform_array_operations(data_array, stats["mean"])

    print("\n--- Entered Data (NumPy Array) ---")
    print(data_array)

    print("\n--- Calculated Statistics ---")
    print(f"Mean   : {stats['mean']:.2f}")
    print(f"Median : {stats['median']:.2f}")
    print(f"Max    : {stats['max']:.2f}")
    print(f"Min    : {stats['min']:.2f}")
    print(f"Sum    : {stats['sum']:.2f}")

    print("\n--- Array Operations ---")
    print("Sorted Data (Ascending):", sorted_data)
    print("Values Above Mean      :", above_mean)

    print("\n--- AI Insight (Groq) ---")
    insight = get_ai_insight(stats, len(above_mean), len(data_array))
    print(insight)


if __name__ == "__main__":
    main()
    
