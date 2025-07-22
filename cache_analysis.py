import pandas as pd

cache_data = {
    'Metric': ['Cloud Requests/Hour', 'Cost Savings', 'Response Time'],
    'Without Cache': [1000, '$12/hour', '400ms'],
    'With Cache (90% hit rate)': [100, '$1.20/hour', '200ms']
}

df = pd.DataFrame(cache_data)
print(df.to_string(index=False))
