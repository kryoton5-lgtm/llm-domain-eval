"""
Adversarial Template Definitions — Reusable prompt templates for stress-testing LLMs.
"""

TEMPLATES = {
    "stem_numerical": {
        "description": "STEM question with numerical values to perturb",
        "template": "A {object} of mass {mass} kg is moving at {velocity} m/s. "
                     "Calculate the kinetic energy in Joules.",
        "variables": {
            "object": ["ball", "block", "car", "satellite", "projectile"],
            "mass": [1, 2, 5, 10, 50, 100],
            "velocity": [3, 5, 10, 20, 50],
        },
        "answer_formula": "0.5 * mass * velocity**2",
    },
    "pharmacy_dosage": {
        "description": "Pharmacy dosage calculation with safety implications",
        "template": "A patient weighing {weight} kg requires {drug} at a dose of "
                     "{dose_per_kg} mg/kg/day, divided into {divisions} equal doses. "
                     "Calculate the amount per dose in mg.",
        "variables": {
            "weight": [50, 60, 70, 80, 90],
            "drug": ["amoxicillin", "ibuprofen", "metformin"],
            "dose_per_kg": [10, 15, 20, 25],
            "divisions": [2, 3, 4],
        },
        "answer_formula": "(weight * dose_per_kg) / divisions",
    },
    "finance_valuation": {
        "description": "Basic DCF valuation question",
        "template": "A company generates free cash flow of ${fcf}M annually, "
                     "growing at {growth}% per year. Using a discount rate of {discount}%, "
                     "estimate the terminal value using the Gordon Growth Model.",
        "variables": {
            "fcf": [10, 25, 50, 100],
            "growth": [2, 3, 4, 5],
            "discount": [8, 10, 12, 15],
        },
        "answer_formula": "fcf * (1 + growth/100) / (discount/100 - growth/100)",
    },
    "context_injection_benign": {
        "description": "Irrelevant context sentences for injection testing",
        "distractors": [
            "The experiment was conducted in a temperature-controlled laboratory.",
            "Similar problems appear in Chapter 7 of most standard textbooks.",
            "This type of question is commonly tested in competitive examinations.",
            "The original formulation used imperial units before being converted.",
            "Note: all measurements were taken using calibrated instruments.",
            "The problem assumes ideal conditions with no friction or air resistance.",
            "A related but different approach was published in a 2019 paper.",
        ],
    },
    "negation_variants": {
        "description": "Question variants with negation to test comprehension",
        "pairs": [
            ("Which of the following IS a property of...",
             "Which of the following is NOT a property of..."),
            ("Select the correct statement:",
             "Select the INCORRECT statement:"),
            ("This reaction will proceed because...",
             "This reaction will NOT proceed because..."),
            ("The value increases when...",
             "The value decreases when..."),
        ],
    },
}
