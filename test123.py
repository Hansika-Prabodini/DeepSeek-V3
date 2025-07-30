# Function to process a string and return its length and uppercase version
def process_string(input_string):
    """
    This function takes an input string, calculates its length,
    and converts it to uppercase.
    
    Parameters:
    input_string (str): The string to be processed.

    Returns:
    tuple: A tuple containing the length of the string and the uppercase version.
    """
    # Calculate the length of the input string
    length = len(input_string)
    
    # Convert the input string to uppercase
    uppercase_string = input_string.upper()
    
    return length, uppercase_string

# Example usage of the process_string function
if __name__ == "__main__":
    input_data = "fhjfhgfhgwfdhAKHlfgv"
    result = process_string(input_data)
    
    # Output the results
    print(f"Length of the string: {result[0]}")
    print(f"Uppercase version: {result[1]}")