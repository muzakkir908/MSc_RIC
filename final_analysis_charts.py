import matplotlib.pyplot as plt
import numpy as np

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Architecture Evolution
architectures = ['2-Tier\n(Week 8)', '3-Tier\n(Week 9)', '3-Tier+Opt\n(Week 10)']
latencies = [450, 400, 100]
ax1.bar(architectures, latencies, color=['red', 'orange', 'green'], alpha=0.7)
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Architecture Evolution')
ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5)

# 2. Cost Savings
scenarios = ['No Caching', 'With Fog Cache']
costs = [100, 10]
ax2.bar(scenarios, costs, color=['red', 'green'], alpha=0.7)
ax2.set_ylabel('Relative Cost')
ax2.set_title('Cost Reduction with Fog Layer')

# 3. Cache Performance
metrics = ['Cache Hits', 'Cloud Requests', 'Avg Response Time']
without_cache = [0, 100, 400]
with_cache = [90, 10, 200]
x = np.arange(len(metrics))
width = 0.35
ax3.bar(x - width/2, without_cache, width, label='Without Cache', color='red', alpha=0.7)
ax3.bar(x + width/2, with_cache, width, label='With Cache', color='green', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.set_title('Fog Cache Benefits')
ax3.legend()

# 4. System Reliability
phases = ['Development', 'Testing', 'Deployment', 'Production']
success_rates = [95, 90, 85, 80]
ax4.plot(phases, success_rates, 'o-', color='blue', linewidth=2, markersize=10)
ax4.set_ylabel('Success Rate (%)')
ax4.set_title('System Reliability Across Phases')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3)

plt.suptitle('Cloud Gaming System - Complete Analysis', fontsize=16)
plt.tight_layout()
plt.savefig('final_system_analysis.png', dpi=150)
plt.show()
