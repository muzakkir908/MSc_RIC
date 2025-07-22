#!/usr/bin/env python3
"""
AWS Environment Setup Script for Cloud Gaming Project
Sets up EC2 instances for Edge and Cloud deployment with network slicing simulation
"""

import boto3
import json
import time
import os
from datetime import datetime

class AWSEnvironmentSetup:
    def __init__(self, region='us-east-1'):
        """Initialize AWS clients"""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.vpc_client = boto3.client('ec2', region_name=region)
        
        # Instance configurations
        self.edge_config = {
            'instance_type': 't3.medium',  # 2 vCPU, 4GB RAM
            'ami_id': None,  # Will be set based on region
            'key_name': 'cloud-gaming-key',
            'security_group': 'cloud-gaming-sg',
            'instance_name': 'CloudGaming-Edge-Server'
        }
        
        self.cloud_config = {
            'instance_type': 't3.large',  # 2 vCPU, 8GB RAM
            'ami_id': None,
            'key_name': 'cloud-gaming-key',
            'security_group': 'cloud-gaming-sg',
            'instance_name': 'CloudGaming-Cloud-Server'
        }
        
        self.setup_log = []
        
    def log(self, message):
        """Log setup progress"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
    
    def get_latest_ubuntu_ami(self):
        """Get the latest Ubuntu 22.04 AMI for the region"""
        self.log("🔍 Finding latest Ubuntu 22.04 AMI...")
        
        response = self.ec2.describe_images(
            Filters=[
                {'Name': 'name', 'Values': ['ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*']},
                {'Name': 'state', 'Values': ['available']},
                {'Name': 'owner-id', 'Values': ['099720109477']}  # Canonical
            ],
            Owners=['099720109477']
        )
        
        # Sort by creation date and get the latest
        images = sorted(response['Images'], key=lambda x: x['CreationDate'], reverse=True)
        
        if images:
            ami_id = images[0]['ImageId']
            self.edge_config['ami_id'] = ami_id
            self.cloud_config['ami_id'] = ami_id
            self.log(f"✅ Found AMI: {ami_id}")
            return ami_id
        else:
            raise Exception("No suitable Ubuntu AMI found")
    
    def create_key_pair(self):
        """Create EC2 key pair for SSH access"""
        key_name = self.edge_config['key_name']
        
        try:
            # Check if key already exists
            self.ec2.describe_key_pairs(KeyNames=[key_name])
            self.log(f"🔑 Key pair '{key_name}' already exists")
            return True
        except:
            # Create new key pair
            self.log(f"🔑 Creating key pair '{key_name}'...")
            
            response = self.ec2.create_key_pair(KeyName=key_name)
            
            # Save private key
            private_key = response['KeyMaterial']
            key_file = f"{key_name}.pem"
            
            with open(key_file, 'w') as f:
                f.write(private_key)
            
            # Set proper permissions
            os.chmod(key_file, 0o400)
            
            self.log(f"✅ Key pair created and saved to {key_file}")
            self.log("⚠️  Keep this file safe - it's needed for SSH access!")
            
            return True
    
    def create_security_group(self):
        """Create security group with required rules"""
        sg_name = self.edge_config['security_group']
        
        try:
            # Check if security group exists
            response = self.ec2.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': [sg_name]}]
            )
            
            if response['SecurityGroups']:
                sg_id = response['SecurityGroups'][0]['GroupId']
                self.log(f"🛡️ Security group '{sg_name}' already exists: {sg_id}")
                return sg_id
        except:
            pass
        
        # Create new security group
        self.log(f"🛡️ Creating security group '{sg_name}'...")
        
        response = self.ec2.create_security_group(
            GroupName=sg_name,
            Description='Security group for cloud gaming project'
        )
        
        sg_id = response['GroupId']
        
        # Add inbound rules
        rules = [
            # SSH access
            {
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
            },
            # HTTP for testing
            {
                'IpProtocol': 'tcp',
                'FromPort': 80,
                'ToPort': 80,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'HTTP access'}]
            },
            # Custom ports for gaming simulation
            {
                'IpProtocol': 'tcp',
                'FromPort': 8000,
                'ToPort': 8010,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Gaming ports'}]
            },
            # UDP for game traffic
            {
                'IpProtocol': 'udp',
                'FromPort': 5000,
                'ToPort': 5010,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Game UDP traffic'}]
            }
        ]
        
        self.ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=rules
        )
        
        self.log(f"✅ Security group created: {sg_id}")
        return sg_id
    
    def create_user_data_script(self, instance_type='edge'):
        """Create initialization script for instances"""
        
        script = '''#!/bin/bash
# Update system
apt-get update
apt-get upgrade -y

# Install Python and pip
apt-get install -y python3 python3-pip python3-venv

# Install system monitoring tools
apt-get install -y htop iotop iftop

# Install network tools
apt-get install -y tc iproute2 iperf3

# Create project directory
mkdir -p /home/ubuntu/cloud-gaming
cd /home/ubuntu/cloud-gaming

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install numpy pandas scikit-learn torch matplotlib seaborn flask boto3

# Create a test script
cat > test_instance.py << 'EOF'
import platform
import psutil
import json
from datetime import datetime

def get_system_info():
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
        "instance_type": "''' + instance_type + '''",
        "status": "ready"
    }
    return info

if __name__ == "__main__":
    print(json.dumps(get_system_info(), indent=2))
EOF

# Create network slicing simulation script
cat > simulate_network_slice.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys

def set_network_slice(slice_type):
    """Simulate network slicing using tc (traffic control)"""
    
    slices = {
        "low": {"rate": "1mbit", "latency": "100ms", "loss": "0.5%"},
        "medium": {"rate": "10mbit", "latency": "60ms", "loss": "0.1%"},
        "high": {"rate": "100mbit", "latency": "20ms", "loss": "0.01%"}
    }
    
    if slice_type not in slices:
        print(f"Invalid slice type. Choose from: {list(slices.keys())}")
        return
    
    config = slices[slice_type]
    interface = "eth0"  # Default network interface
    
    # Clear existing rules
    subprocess.run(["sudo", "tc", "qdisc", "del", "dev", interface, "root"], 
                   capture_output=True)
    
    # Add new rules
    commands = [
        ["sudo", "tc", "qdisc", "add", "dev", interface, "root", "handle", "1:", "htb"],
        ["sudo", "tc", "class", "add", "dev", interface, "parent", "1:", "classid", 
         "1:1", "htb", "rate", config["rate"]],
        ["sudo", "tc", "qdisc", "add", "dev", interface, "parent", "1:1", "netem", 
         "delay", config["latency"], "loss", config["loss"]]
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(f"✓ Executed: {' '.join(cmd)}")
    
    print(f"\\nNetwork slice '{slice_type}' configured:")
    print(f"  Rate: {config['rate']}")
    print(f"  Latency: {config['latency']}")
    print(f"  Loss: {config['loss']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        set_network_slice(sys.argv[1])
    else:
        print("Usage: python simulate_network_slice.py [low|medium|high]")
EOF

chmod +x simulate_network_slice.py

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu/cloud-gaming

# Create startup message
echo "Cloud Gaming ''' + instance_type.upper() + ''' Instance Ready!" > /etc/motd
echo "Project directory: /home/ubuntu/cloud-gaming" >> /etc/motd
echo "Python environment: source /home/ubuntu/cloud-gaming/venv/bin/activate" >> /etc/motd
'''
        
        return script
    
    def launch_instance(self, config, instance_type='edge'):
        """Launch EC2 instance with specified configuration"""
        self.log(f"🚀 Launching {instance_type} instance...")
        
        # Get user data script
        user_data = self.create_user_data_script(instance_type)
        
        # Launch instance
        response = self.ec2.run_instances(
            ImageId=config['ami_id'],
            InstanceType=config['instance_type'],
            KeyName=config['key_name'],
            SecurityGroups=[config['security_group']],
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': config['instance_name']},
                        {'Key': 'Project', 'Value': 'CloudGaming'},
                        {'Key': 'Type', 'Value': instance_type}
                    ]
                }
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        self.log(f"✅ Instance launched: {instance_id}")
        
        # Wait for instance to be running
        self.log("⏳ Waiting for instance to start...")
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get instance details
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        
        public_ip = instance.get('PublicIpAddress', 'N/A')
        private_ip = instance.get('PrivateIpAddress', 'N/A')
        
        self.log(f"✅ Instance is running!")
        self.log(f"   Public IP: {public_ip}")
        self.log(f"   Private IP: {private_ip}")
        
        return {
            'instance_id': instance_id,
            'public_ip': public_ip,
            'private_ip': private_ip,
            'instance_type': config['instance_type'],
            'name': config['instance_name']
        }
    
    def test_instance_connectivity(self, instance_info):
        """Test SSH connectivity to instance"""
        self.log(f"\n🔗 Testing connectivity to {instance_info['name']}...")
        
        # Create test commands
        test_commands = f"""
# Test SSH connection (after instance is fully initialized):
ssh -i {self.edge_config['key_name']}.pem ubuntu@{instance_info['public_ip']}

# Once connected, test the instance:
cd /home/ubuntu/cloud-gaming
source venv/bin/activate
python test_instance.py

# Test network slicing:
sudo python simulate_network_slice.py medium
"""
        
        return test_commands
    
    def setup_complete_environment(self):
        """Setup complete AWS environment"""
        self.log("🎮 STARTING AWS ENVIRONMENT SETUP FOR CLOUD GAMING")
        self.log("="*60)
        
        try:
            # Step 1: Get AMI
            self.get_latest_ubuntu_ami()
            
            # Step 2: Create key pair
            self.create_key_pair()
            
            # Step 3: Create security group
            self.create_security_group()
            
            # Step 4: Launch Edge instance
            edge_info = self.launch_instance(self.edge_config, 'edge')
            
            # Step 5: Launch Cloud instance
            cloud_info = self.launch_instance(self.cloud_config, 'cloud')
            
            # Step 6: Generate connection instructions
            self.log("\n✅ SETUP COMPLETE!")
            self.log("="*60)
            
            # Save instance information
            instances_info = {
                'setup_date': datetime.now().isoformat(),
                'region': self.region,
                'edge_instance': edge_info,
                'cloud_instance': cloud_info,
                'key_file': f"{self.edge_config['key_name']}.pem",
                'security_group': self.edge_config['security_group']
            }
            
            with open('aws_instances.json', 'w') as f:
                json.dump(instances_info, f, indent=2)
            
            self.log("\n📝 Instance information saved to aws_instances.json")
            
            # Print connection instructions
            self.log("\n🔗 CONNECTION INSTRUCTIONS:")
            self.log("-"*40)
            self.log(f"\nEDGE SERVER:")
            print(self.test_instance_connectivity(edge_info))
            
            self.log(f"\nCLOUD SERVER:")
            print(self.test_instance_connectivity(cloud_info))
            
            # Print next steps
            self.log("\n📋 NEXT STEPS:")
            self.log("1. Wait 2-3 minutes for instances to fully initialize")
            self.log("2. SSH into each instance using the commands above")
            self.log("3. Test the Python environment and network slicing")
            self.log("4. Deploy your trained model to the edge instance")
            self.log("5. Setup communication between edge and cloud instances")
            
            # Save setup log
            with open('setup_log.txt', 'w') as f:
                f.write('\n'.join(self.setup_log))
            
            return instances_info
            
        except Exception as e:
            self.log(f"❌ Error during setup: {e}")
            raise

def create_deployment_script():
    """Create a script to deploy model to AWS instances"""
    deployment_script = '''#!/bin/bash
# Model Deployment Script for AWS Instances

INSTANCE_IP=$1
KEY_FILE=$2
MODEL_DIR=$3

if [ $# -ne 3 ]; then
    echo "Usage: ./deploy_model.sh <instance_ip> <key_file> <model_directory>"
    exit 1
fi

echo "📦 Deploying model to $INSTANCE_IP..."

# Copy model files
scp -i $KEY_FILE -r $MODEL_DIR ubuntu@$INSTANCE_IP:/home/ubuntu/cloud-gaming/

# Copy inference script
scp -i $KEY_FILE inference_server.py ubuntu@$INSTANCE_IP:/home/ubuntu/cloud-gaming/

# SSH and setup
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP << 'EOF'
cd /home/ubuntu/cloud-gaming
source venv/bin/activate

# Install additional dependencies if needed
pip install flask gunicorn

# Create systemd service for model server
sudo tee /etc/systemd/system/combat-predictor.service > /dev/null << 'SERVICE'
[Unit]
Description=Combat Prediction Model Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cloud-gaming
Environment="PATH=/home/ubuntu/cloud-gaming/venv/bin"
ExecStart=/home/ubuntu/cloud-gaming/venv/bin/gunicorn -w 2 -b 0.0.0.0:8000 inference_server:app
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable combat-predictor
sudo systemctl start combat-predictor

echo "✅ Model deployed and service started!"
echo "Test at: http://$INSTANCE_IP:8000/predict"
EOF
'''
    
    with open('deploy_model.sh', 'w') as f:
        f.write(deployment_script)
    
    os.chmod('deploy_model.sh', 0o755)
    print("✅ Created deploy_model.sh script")

def create_inference_server():
    """Create Flask inference server for deployment"""
    server_code = '''from flask import Flask, request, jsonify
import torch
import numpy as np
import pickle
import json
from inference import CombatPredictor

app = Flask(__name__)

# Load model on startup
predictor = CombatPredictor()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from request
        data = request.json
        features = data.get('features', {})
        
        # Make prediction
        prob, pred = predictor.predict(features)
        
        # Determine network slice recommendation
        if pred == 1 or prob > 0.7:
            recommended_slice = 'high'
        elif prob > 0.3:
            recommended_slice = 'medium'
        else:
            recommended_slice = 'low'
        
        response = {
            'combat_probability': float(prob),
            'prediction': 'combat' if pred == 1 else 'non-combat',
            'recommended_slice': recommended_slice,
            'timestamp': data.get('timestamp', '')
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Get batch of features
        data = request.json
        batch_features = data.get('batch', [])
        
        results = []
        for features in batch_features:
            prob, pred = predictor.predict(features)
            results.append({
                'combat_probability': float(prob),
                'prediction': 'combat' if pred == 1 else 'non-combat'
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
'''
    
    with open('inference_server.py', 'w') as f:
        f.write(server_code)
    
    print("✅ Created inference_server.py")

def main():
    """Main setup function"""
    print("🎮 AWS CLOUD GAMING ENVIRONMENT SETUP")
    print("=" * 60)
    print("This script will set up AWS EC2 instances for your project")
    print("Prerequisites:")
    print("  - AWS CLI configured with credentials")
    print("  - Sufficient EC2 quota in your region")
    print("  - Budget awareness (instances will incur charges)")
    print("=" * 60)
    
    # Get user confirmation
    confirm = input("\nProceed with AWS setup? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Setup cancelled.")
        return
    
    # Get region preference
    region = input("Enter AWS region (default: us-east-1): ").strip() or 'us-east-1'
    
    try:
        # Initialize setup
        setup = AWSEnvironmentSetup(region=region)
        
        # Run complete setup
        instances_info = setup.setup_complete_environment()
        
        # Create additional helper scripts
        create_deployment_script()
        create_inference_server()
        
        print("\n✅ ALL SETUP COMPLETE!")
        print("\n📁 Created files:")
        print("  - aws_instances.json: Instance information")
        print("  - cloud-gaming-key.pem: SSH key (keep safe!)")
        print("  - deploy_model.sh: Model deployment script")
        print("  - inference_server.py: Flask server for predictions")
        print("  - setup_log.txt: Complete setup log")
        
        # Estimate costs
        print("\n💰 ESTIMATED COSTS:")
        print("  Edge instance (t3.medium): ~$0.0416/hour")
        print("  Cloud instance (t3.large): ~$0.0832/hour")
        print("  Total: ~$0.1248/hour (~$3/day if running continuously)")
        print("\n⚠️  Remember to STOP or TERMINATE instances when not in use!")
        
        # Cleanup commands
        print("\n🧹 CLEANUP COMMANDS (save for later):")
        print(f"  Stop instances: aws ec2 stop-instances --instance-ids {instances_info['edge_instance']['instance_id']} {instances_info['cloud_instance']['instance_id']}")
        print(f"  Terminate instances: aws ec2 terminate-instances --instance-ids {instances_info['edge_instance']['instance_id']} {instances_info['cloud_instance']['instance_id']}")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()