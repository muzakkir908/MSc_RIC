import matplotlib.pyplot as plt

systems = ['Expected\n(Proposal)', 'Actual\n(t2.micro)', 'Projected\n(t3.medium)']
latencies = [50, 400, 100]
colors = ['green', 'red', 'orange']

plt.figure(figsize=(10, 6))
bars = plt.bar(systems, latencies, color=colors, alpha=0.7)
plt.axhline(y=50, color='green', linestyle='--', label='Target (<50ms)')
plt.ylabel('Total Latency (ms)')
plt.title('System Latency: Actual vs Expected Performance')
plt.legend()

for bar, val in zip(bars, latencies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val}ms', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('latency_analysis.png')
plt.show()
