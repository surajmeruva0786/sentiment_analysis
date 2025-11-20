import pandas as pd
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from costitu2 import bias_score, toxicity_score

def evaluate_tweet(text):
    """
    Evaluates a single tweet for bias and toxicity.
    Returns a dictionary with scores and violation status.
    """
    if not isinstance(text, str):
        text = str(text)
        
    bias_count, bias_matches = bias_score(text)
    toxicity = toxicity_score(text)
    
    # Thresholds from costitu2.py
    # bias_threshold=1, toxicity_threshold=0.2
    violated = (bias_count > 1) or (toxicity > 0.2)
    
    return {
        "bias_count": bias_count,
        "bias_matches": ", ".join([f"{m[0]}({m[1]})" for m in bias_matches]) if bias_matches else "",
        "toxicity": round(toxicity, 3),
        "violated": violated
    }

def main():
    input_file = '11th_hour_political_tweets.csv'
    
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='|')
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    if 'full_text' not in df.columns:
        print("Error: 'full_text' column not found in CSV.")
        return

    print("Processing tweets...")
    # Apply evaluation
    results = df['full_text'].apply(evaluate_tweet)
    
    # Expand results into columns
    results_df = pd.DataFrame(results.tolist())
    
    # Concatenate with original dataframe
    df_processed = pd.concat([df, results_df], axis=1)
    
    # Segregate
    safe_tweets = df_processed[~df_processed['violated']]
    violated_tweets = df_processed[df_processed['violated']]
    
    # Save
    safe_file = 'segregated_safe.csv'
    violated_file = 'segregated_violated.csv'
    
    safe_tweets.to_csv(safe_file, index=False)
    violated_tweets.to_csv(violated_file, index=False)
    
    print(f"Processing complete.")
    print(f"Safe tweets: {len(safe_tweets)} saved to {safe_file}")
    print(f"Violated tweets: {len(violated_tweets)} saved to {violated_file}")

if __name__ == "__main__":
    main()
