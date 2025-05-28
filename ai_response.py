
# Set up OpenAI API key

import openai
import fitz  # PyMuPDF
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set your OpenAI API key
openai.api_key = open_api_key

# ─────────────────────────────────────────────────────────────
# Utility: Read text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in document)
    except Exception as e:
        return f"Error reading PDF: {e}"

# ─────────────────────────────────────────────────────────────
# Load GTOC placements (for answering normal questions)
def load_gtoc_pdfs_by_placement(base_folder="Gtoc information"):
    gtoc_data = {}

    for gtoc_folder in os.listdir(base_folder):
        gtoc_path = os.path.join(base_folder, gtoc_folder)
        if not os.path.isdir(gtoc_path):
            continue

        placement_map = {}
        for subfolder in os.listdir(gtoc_path):
            subfolder_path = os.path.join(gtoc_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            folder_name = subfolder.lower()
            placement = "other"
            if "1st" in folder_name:
                placement = "1st"
            elif "2nd" in folder_name:
                placement = "2nd"
            elif "3rd" in folder_name:
                placement = "3rd"

            for file in os.listdir(subfolder_path):
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(subfolder_path, file)
                    placement_map[placement] = extract_text_from_pdf(pdf_path)
                    break

        if placement_map:
            gtoc_data[gtoc_folder.lower()] = placement_map

    return gtoc_data

# ─────────────────────────────────────────────────────────────
# Load all past GTOC problem statements
def load_past_problem_statements(base_folder="Gtoc information"):
    problems = []

    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith(".pdf") and "-problem" in file.lower():
                text = extract_text_from_pdf(os.path.join(folder_path, file))
                if text:
                    snippet = text.strip().replace("\n", " ")[:1000]
                    problems.append(snippet)

    return problems

# ─────────────────────────────────────────────────────────────
# XAI evaluation: get feedback from GPT
def get_xai_feedback(generated_problem, past_problems_sample):
    prompt = (
        "You are an expert judge of Global Trajectory Optimization Competition (GTOC) problem statements.\n\n"
        "Evaluate the following GENERATED PROBLEM compared to PAST PROBLEMS:\n"
        "- Score it 0 to 100 (alignment with past GTOCs).\n"
        "- Explain why you gave that score.\n"
        "- Suggest improvements.\n\n"
        "PAST PROBLEMS:\n"
        f"{past_problems_sample}\n\n"
        "GENERATED PROBLEM:\n"
        f"{generated_problem}\n\n"
        "Your response format:\n"
        "Score: (number)\n"
        "Explanation: (text)\n"
        "Suggestions: (text)\n"
    )

    response = get_openai_response(prompt)
    return response

# ─────────────────────────────────────────────────────────────
# OpenAI call
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful GTOC assistant. Only use provided information unless asked to create new problems."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# ─────────────────────────────────────────────────────────────
# Create a new GTOC problem + data with XAI checking
def create_fake_gtoc(gtoc_number, base_folder="Gtoc information"):
    past_problems_list = load_past_problem_statements(base_folder)
    past_problems_sample = "\n\n".join(past_problems_list[:5])

    # Step 1: Generate initial GTOC problem
    prompt = (
        f"Based on the style and content of previous GTOC problem statements below, create a new problem "
        f"for a hypothetical GTOC{gtoc_number}. Include a mission description and objectives.\n\n"
        f"Then generate fake mission data in CSV format (columns: Target Name, Distance (AU), Delta-V (km/s), Time Window (days)).\n\n"
        f"PAST PROBLEMS:\n{past_problems_sample}"
    )

    response = get_openai_response(prompt)

    # Separate into problem and CSV
    problem_text, csv_data = "No problem generated", "Target Name,Distance (AU),Delta-V (km/s),Time Window (days)"
    if "Target Name" in response:
        parts = response.split("Target Name", 1)
        problem_text = parts[0].strip()
        csv_data = "Target Name" + parts[1].strip()

    # Step 2: Get XAI feedback
    xai_feedback = get_xai_feedback(problem_text, past_problems_sample)
    print(f"🧠 XAI Feedback:\n{xai_feedback}")

    # Extract score from feedback
    match = re.search(r"Score:\s*(\d+)", xai_feedback)
    if match:
        score = int(match.group(1))
    else:
        score = 0  # Assume bad if no score found

    # Step 3: Auto-fix if similarity is too low
    if score < 70:
        print("⚠️ XAI feedback score too low, requesting improvement...")

        fix_prompt = (
            f"The following GTOC problem scored poorly ({score}). Please revise it to better match the style and depth of real GTOCs.\n\n"
            f"Problem to fix:\n{problem_text}\n\n"
            f"Real examples:\n{past_problems_sample}\n\n"
            f"Suggestions:\n{xai_feedback}"
        )

        improved_response = get_openai_response(fix_prompt)
        if "Target Name" in improved_response:
            parts = improved_response.split("Target Name", 1)
            problem_text = parts[0].strip()
            csv_data = "Target Name" + parts[1].strip()

        print("✅ Problem improved based on XAI feedback.")

    # Step 4: Save outputs
    new_folder = os.path.join(base_folder, f"Gtoc{gtoc_number}")
    os.makedirs(new_folder, exist_ok=True)

    with open(os.path.join(new_folder, f"GTOC{gtoc_number}-PROBLEM.txt"), "w", encoding="utf-8") as f:
        f.write(problem_text)

    with open(os.path.join(new_folder, f"GTOC{gtoc_number}-data.csv"), "w", encoding="utf-8") as f:
        f.write(csv_data)

    with open(os.path.join(new_folder, f"GTOC{gtoc_number}-XAI-Feedback.txt"), "w", encoding="utf-8") as f:
        f.write(xai_feedback)

    print(f"✅ GTOC{gtoc_number} created and saved in '{new_folder}'.")

# ─────────────────────────────────────────────────────────────
# Parse user input
def parse_gtoc_request(user_input):
    match = re.search(r"gtoc\s*(\d+)", user_input.lower())
    if match:
        number = int(match.group(1))
        if number > 99:
            return None, None
        gtoc_key = f"gtoc{number}"
    else:
        gtoc_key = None

    place_match = re.search(r"(1st|first|2nd|second|3rd|third)", user_input.lower())
    place_map = {
        "1st": "1st", "first": "1st",
        "2nd": "2nd", "second": "2nd",
        "3rd": "3rd", "third": "3rd"
    }
    placement_key = place_map.get(place_match.group(1)) if place_match else None

    return gtoc_key, placement_key

# ─────────────────────────────────────────────────────────────
# Main chatbot loop
def start_chatbot(gtoc_data):
    print("Chatbot ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        gen_match = re.search(r"(?:create|generate|make).*?gtoc\s*(\d+)", user_input.lower())
        if gen_match and gen_match.group(1):
            future_gtoc = int(gen_match.group(1))
            if f"gtoc{future_gtoc}" not in gtoc_data:
                create_fake_gtoc(future_gtoc)
                continue
            else:
                print(f"GTOC{future_gtoc} already exists.")
                continue

        # Normal answering
        gtoc_key, placement_key = parse_gtoc_request(user_input)

        if gtoc_key:
            folder_data = gtoc_data.get(gtoc_key)
            if folder_data:
                if placement_key:
                    content = folder_data.get(placement_key)
                    if content:
                        prompt = f"Based only on this content:\n\n{content[:2000]}\n\nAnswer this:\n{user_input}"
                    else:
                        prompt = f"No info for {placement_key} in {gtoc_key.upper()}."
                else:
                    combined = "\n\n".join(
                        f"{place.upper()} PLACE:\n{txt[:800]}" for place, txt in folder_data.items()
                    )
                    prompt = f"Here's info for {gtoc_key.upper()}:\n{combined}\n\nUser asked: {user_input}"
            else:
                prompt = f"{gtoc_key.upper()} not found."
        else:
            all_text = ""
            for g_key, placements in gtoc_data.items():
                for place, txt in placements.items():
                    all_text += f"{g_key.upper()} {place.upper()}:\n{txt[:500]}\n\n"
            prompt = f"Based on all known GTOCs:\n\n{all_text[:3000]}\n\nUser's question: {user_input}"

        response = get_openai_response(prompt)
        print("Bot:", response)

# ─────────────────────────────────────────────────────────────
# Main runner
def main():
    gtoc_data = load_gtoc_pdfs_by_placement("Gtoc information")
    start_chatbot(gtoc_data)

if __name__ == "__main__":
    main()
