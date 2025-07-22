import sys
sys.path.append('.')

print("Running all tests...")

# Add your test imports here
from tests.integration_tests import test_aws_complete
from tests.performance_tests import test_baseline

# Run tests
test_aws_complete.main()
test_baseline.main()
