import tweepy
import pandas as pd
import os
from datetime import datetime
import costitu2 as constitu2  # Importing the existing module

# ---------------------------
# Twitter API Setup
# ---------------------------
# Credentials provided by user
API_KEY = os.getenv("TWITTER_API_KEY", "JOw8vAb6yEnIUlrwuH4wAYEiX")
API_SECRET = os.getenv("TWITTER_API_SECRET", "dRZQrGA9CRfk9lmzCk5B5XQqUMKecr4Lqo9TybvQXDz1eDqSls")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "1958893707916451842-sWBbUZohzByzeoojFbOtOzP2yvzi4Z")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "oJ5CTSdyHkUZnttAAZqruMcgaUR9H38yhMcfcuOeVJVeP")

def get_twitter_client():
    """Authenticates with Twitter API v2 using Consumer Keys and Access Tokens."""
    if not all([API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET]):
        print("⚠️  WARNING: One or more Twitter API credentials are missing.")
        return None
    
    try:
        client = tweepy.Client(
            consumer_key=API_KEY,
            consumer_secret=API_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET
        )
        
        # Verify credentials
        me = client.get_me()
        if me.data:
            print(f"✅ Authenticated as: @{me.data.username}")
        else:
            print("⚠️  Authentication succeeded but could not fetch user details.")
            
        return client
    except Exception as e:
        print(f"❌ Error authenticating with Twitter: {e}")
        return None

def fetch_tweets(query, max_results=10):
    """Fetches tweets matching the query."""
    client = get_twitter_client()
    if not client:
        return []

    # API requires max_results to be between 10 and 100
    if max_results < 10:
        print("⚠️  max_results must be at least 10. Setting to 10.")
        max_results = 10
        
    print(f"🔍 Searching for tweets matching: '{query}'...")
    try:
        # search_recent_tweets is for the standard v2 endpoint (requires Basic or Pro access usually, 
        # but some levels allow it. If using Essential, might be limited).
        response = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at", "lang"])
        
        if not response.data:
            print("No tweets found.")
            return []
        
        return [tweet.text for tweet in response.data]
    except Exception as e:
        print(f"❌ Error fetching tweets via search: {e}")
        
        # Fallback: Try fetching user's own tweets (Free Tier often allows this or at least 'get_me')
        print("⚠️  Search failed (likely due to Free Tier limitations). Attempting to fetch your own recent tweets...")
        try:
            me = client.get_me()
            if me.data:
                response = client.get_users_tweets(id=me.data.id, max_results=max_results, tweet_fields=["created_at", "lang"])
                if response.data:
                    print(f"✅ Successfully fetched {len(response.data)} tweets from your timeline (@{me.data.username}).")
                    return [tweet.text for tweet in response.data]
                else:
                    print("No tweets found in your timeline.")
                    return []
        except Exception as e2:
            print(f"❌ Error fetching user timeline: {e2}")
            
        return []

def analyze_tweets(tweets):
    """Applies constitu2 analysis on a list of tweet texts."""
    results = []
    
    for text in tweets:
        # Use the constitution_aware_decode function from constitu2.py
        # We assume the function signature matches what we saw in the file.
        analysis = constitu2.constitution_aware_decode(text)
        
        # Extract relevant fields
        # The analysis returns a dict with keys like 'candidates', 'selected', 'violated', etc.
        # Since we are analyzing an existing text (the tweet), not generating one, 
        # we might need to adapt how we interpret 'candidates'. 
        # However, constitu2.constitution_aware_decode generates candidates *based on* the input prompt.
        # Wait, looking at constitu2.py:
        # simulated_generator(prompt) -> creates templates like "The {prompt} are lazy..."
        # This seems to be designed to TEST prompts for bias, not analyze existing text for bias directly.
        # 
        # IF the user wants to analyze the TWEET itself for bias:
        # We should use `bias_score(tweet_text)` and `toxicity_score(tweet_text)` directly.
        # 
        # Let's look at constitu2.py again.
        # bias_score(text) -> returns count, matches
        # toxicity_score(text) -> returns score
        
        bcount, bmatches = constitu2.bias_score(text)
        tox = constitu2.toxicity_score(text)
        flu = constitu2.fluency_score(text)
        
        # Determine violation based on thresholds in constitu2 (defaults: bias > 1 or tox > 0.2)
        violated = bcount > 1 or tox > 0.2
        
        justification = "Violated thresholds" if violated else "Safe"
        
        results.append({
            "tweet_text": text,
            "bias_count": bcount,
            "bias_matches": ", ".join([f"{m[0]}({m[1]})" for m in bmatches]) if bmatches else "",
            "toxicity": round(tox, 3),
            "fluency": round(flu, 3),
            "violated": violated,
            "justification": justification
        })
        
    return results

def save_results(results, filename="twitter_analysis_results.csv"):
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)
    
    # Segregate into Violated and Safe
    violated_df = df[df["violated"] == True]
    safe_df = df[df["violated"] == False]
    
    print(f"\n📊 Analysis Complete: {len(df)} tweets processed.")
    print(f"   🔴 Violated: {len(violated_df)}")
    print(f"   🟢 Safe: {len(safe_df)}")
    
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ Results saved to: {filename}")

if __name__ == "__main__":
    # Example usage
    print("🐦 Twitter Constitution-Aware Analysis")
    query = input("Enter search query (e.g., 'politics'): ").strip() or "politics"
    count = int(input("Number of tweets to fetch (default 10): ").strip() or 10)
    
    tweets = fetch_tweets(query, max_results=count)
    
    if tweets:
        analysis_results = analyze_tweets(tweets)
        save_results(analysis_results)
