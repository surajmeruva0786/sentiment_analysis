import twitter_analysis

# Mock tweets
mock_tweets = [
    "India is a great country with diverse culture.",
    "Some people are just lazy and stupid.",
    "The festival was celebrated with joy.",
    "Dalit communities face challenges."
]

print("Testing analysis logic with mock tweets...")
results = twitter_analysis.analyze_tweets(mock_tweets)
twitter_analysis.save_results(results, filename="mock_results.csv")

print("Mock analysis complete.")
