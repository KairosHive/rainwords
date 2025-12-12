
import os
import time
import json
import click
import logging
import warnings
import random

warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("You need to install google-generativeai: pip install google-generativeai (requires Python>=3.10)")

def get_gemini_key(key_path='gemini.txt'):
    with open(key_path, "r") as f:
        return f.read().strip()

METAPHOR_PROMPTS = [
    'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. She had a teal death. Her death was ___ 1. Peaceful 2. Unexpected 3. Violent',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. She had a red death. Her death was ___ 1. Violent 2. Calm 3. Prolonged',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. It was a pink idea. The idea was ___ 1. Exciting 2. Bad 3. Confusing',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The soccer player was green. They were ___ 1. New 2. Tough 3. Large',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. He had a crimson day. The day made him feel ___ 1. Angry 2. Calm 3. Slow',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. Her solitude was blue. The solitude was ___ 1. Peaceful 2. Excruciating 3. Lucky',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The hike was yellow. The hike was ___ 1. Tiring 2. Energizing 3. Peaceful',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The photos made him feel purple. He felt ___ 1. Nostalgic 2. Alive 3. Happy',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The photos made him feel purple. He felt ___ 1. Confused 2. Enamored 3. Calm',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The meeting made him feel crimson. He felt ___ 1. Enraged 2. Apologetic 3. Calm',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The meeting made him feel teal. He felt ___ 1. Peaceful 2. Annoyed 3. Enamored',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The meeting made him feel burgundy. He felt ___ 1. Sad 2. Powerful 3. Calm',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. She had an aqua death. Her death was ___ 1. Quick 2. Painful 3. Tranquil',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. His speech made me feel orange. I felt ___ 1. Hopeless 2. Calm 3. Energized',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. His speech made me feel yellow. I felt ___ 1. Inspired 2. Depressed 3. Calm',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. This day is moving by in octaves. This day is ___ 1. moving fast 2. boring 3. hot',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The day is moving in staccato. The day was moving ___ 1. abruptly 2. smoothly 3. in a great way',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. His advice was a serenade. His advice was ___ 1. persuasive 2. irritating 3. boring',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. Their conversation became a soprano’s scale. The conversation became ___ 1. beautiful 2. forgotten 3. sad',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The day ended with a crescendo. The day ___ 1. improved for the better 2. was continuously bad 3. was continuously good',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The pet’s arrival was falsetto. The pet’s arrival was ___ 1. exciting 2. frequent 3. uneventful ',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The announcement was a legion of tenors. The announcement was ___ 1. bold 2. good 3. just okay',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The speaker’s ideas were a yodel. The speaker’s ideas were ___ 1. rapid and oscillating 2. interrupted and noncontinuous 3. slowly evolving',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The president’s arrival was a thud. The president’s arrival was ___ 1. sudden and impactful 2. welcome 3. not nice',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The brunch was a clatter. The brunch was ___ 1. busy 2. cold 3. delicious',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. When the hour came, it was a dong. When the hour came, it was ___ 1. finally time 2. irrelevant  3. hard to measure',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The lobby was a gabble. The lobby was ___ 1. full of people talking 2. breathtaking 3. old',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. His execution was a honk. His execution was ___ 1. a mistake 2. legal 3. an example',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. His advice was a delivery truck beep. The advice was ___ 1. annoying 2. welcomed 3. confusing',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The room was a hum. The room was ___ 1. harmonious 2. large 3. dangerous',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. Her temperament was a growl. Her temperament was ___ 1. angry 2. unexpected 3. predictable',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The baby’s whimper was a chiming memory. The memory was ___ 1. familiar 2. unfamiliar 3. unclear',
 'Your task is to complete the following sentence. Please select the answer that you think best describes its meaning. Please answer using the metaphorical meaning of the sentence rather than the literal meaning. I will give you three choices of response. You need to answer only with the word that is your response. The speech was a roar of footsteps. The speech was ___ 1. loud and clear 2. dull 3. hopeful']

def generate_gemini_response(prompt, temp, model="gemini-3-pro-preview"):
    from google.generativeai.types import GenerationConfig
    generation_config = GenerationConfig(
        max_output_tokens=1000,
        temperature=temp,
        top_p=1.0,
        top_k=1,
    )
    model_obj = genai.GenerativeModel(model_name=model)
    response = model_obj.generate_content(prompt, generation_config=generation_config)
    print("DEBUG: Raw Gemini response:", response)
    if hasattr(response, 'candidates') and response.candidates:
        print("DEBUG: Candidates found:", len(response.candidates))
        for idx, candidate in enumerate(response.candidates):
            finish_reason = getattr(candidate, 'finish_reason', None)
            print(f"DEBUG: Candidate {idx} finish_reason:", finish_reason)
            content = getattr(candidate, "content", None)
            if content and hasattr(content, "parts") and content.parts:
                text = "".join([getattr(part, "text", "") for part in content.parts])
                print(f"DEBUG: Candidate {idx} text:", text)
                if text.strip():
                    return text.strip()
            else:
                print(f"DEBUG: Candidate {idx} has no parts or content.")
        print("DEBUG: No valid candidate.")
        return "[No valid response: content filtered or incomplete]"
    if hasattr(response, 'text'):
        print("DEBUG: Fallback text:", response.text)
        return response.text.strip()
    print("DEBUG: No valid response returned.")
    return "[No valid response returned]"

def safe_generate_gemini_response(prompt, temp, model="gemini-3-pro-preview", retries=5):
    for attempt in range(retries):
        response = generate_gemini_response(prompt, temp, model)
        if not response.startswith("[No valid response"):
            return response
        print(f"DEBUG: Retry {attempt+1} for prompt: {prompt[:40]}...")
        time.sleep(random.uniform(2, 5))
    return response

@click.command()
@click.argument("filename", type=str)
@click.option("--file_path", type=str, default="./")
@click.option("--temp", type=float, default=1.0)
@click.option("--n_valid", type=int, default=130, help="Number of valid responses to collect per prompt")
@click.option("--key_path", type=str, default="gemini.txt")
def main(filename, file_path, temp, n_valid, key_path):
    logger = logging.getLogger(__name__)

    # Setup Gemini
    api_key = get_gemini_key(key_path)
    genai.configure(api_key=api_key)

    output = {}
    for i, prompt in enumerate(METAPHOR_PROMPTS):
        output[i] = []
        print(f"Starting prompt {i+1}/{len(METAPHOR_PROMPTS)}")
        attempt = 0
        while len(output[i]) < n_valid:
            attempt += 1
            logger.info(f"Prompt {i} - GEMINI API Attempt {attempt} (valid so far: {len(output[i])})")
            response = safe_generate_gemini_response(prompt, temp)
            logger.info(f"Response: \n{response}")
            if not response.startswith("[No valid response"):
                output[i].append(response)
                # Save after each new valid response
                output_path = os.path.join(file_path, f"{filename}_temp{temp}_metaphor_{n_valid}.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as outfile:
                    json.dump(output, outfile, indent=2)
                print(f"Saved valid response #{len(output[i])} for prompt {i}")
            else:
                print("No valid response, will retry...")
            # Randomize sleep to avoid rate limiting
            time.sleep(random.uniform(2, 7))
        print(f"Completed prompt {i+1} ({n_valid} valid responses)")

    logger.info(f"done \n {'-'*80}")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
