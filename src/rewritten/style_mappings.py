import json
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))
file_path = os.path.join(project_root, 'rewritten_data', 'config', 'style_mappings.json')
#load json file
with open(file_path) as f:
    STYLE_MAPPINGS = json.load(f)


# Function to get style description
def get_style_description(category, style_code):
    """
    Get style description for the given category and style code

    Args:
        category (str): Style category ('CTA', 'AMOUNT', 'SENSITIVE')
        style_code (str): Style code ('E/S/H' for CTA, 'N/V/O' for AMOUNT, 'K/R/A' for SENSITIVE')

    Returns:
        str: Corresponding English description
    """
    return STYLE_MAPPINGS.get(category, {}).get(style_code, "")

# Example usage
if __name__ == "__main__":
    # When cta is E, return the corresponding English description
    cta_e_description = get_style_description("CTA", "E")
    print(cta_e_description)

    # When amount is N, return the corresponding English description
    amount_n_description = get_style_description("AMOUNT", "N")
    print(amount_n_description)

    # When sensitive is K, return the corresponding English description
    sensitive_k_description = get_style_description("SENSITIVE", "K")
    print(sensitive_k_description)
