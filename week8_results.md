\# Week 8: Two-Tier AWS Deployment Results



\## Deployment Configuration

\- \*\*Edge Server\*\*: EC2 t2.micro (ec2-54-173-203-87.compute-1.amazonaws.com)

&nbsp; - Role: LSTM combat prediction

&nbsp; - Port: 5000



\- \*\*Cloud Server\*\*: EC2 t2.micro (13.220.117.61)

&nbsp; - Role: Q-learning slice decisions

&nbsp; - Port: 5001



\## Performance Results



\### Prediction Accuracy

\- Overall Accuracy: \*\*78.7%\*\*

\- Combat Detection: High combat phases matched well

\- False Positives: Few short spikes



\### Latency Performance

\- Average Total Latency: \*\*425 ms\*\*

\- Edge Processing: (approx from plot, ~300 ms)

\- Cloud Processing: (approx from plot, ~100 ms)

\- Target Achievement (<50 ms): \*\*0%\*\*



\### Resource Optimization

\- Premium Slice Usage: ~78% (from earlier stats)

\- Cost Savings vs Always Premium: \*\*10.1%\*\*

\- QoS Maintained: \*\*Low (violations 100%)\*\*



\## Key Findings

1\. LSTM model predicted combat phases with ~79% accuracy.

2\. Q-learning reduced premium slice cost by ~10%.

3\. Latency was too high to meet strict gaming QoS.



\## Evidence

\- Screenshot: !\[AWS Performance Graph](aws\_performance\_20250714\_221245.png)

\- Data: aws\_test\_results\_20250714\_221245.csv

\- Comparison: baseline\_comparison.png



