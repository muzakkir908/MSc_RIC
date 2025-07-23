import time
import pandas as pd
import numpy as np
from pynput import mouse, keyboard
import threading
import json
from datetime import datetime, timedelta
import psutil
import socket
import platform
import subprocess
import os

class EnhancedGameDataCollector:
    def __init__(self):
        # Initialize data storage
        self.data = []
        self.current_state = {
            'timestamp': 0,
            'mouse_x': 0,
            'mouse_y': 0,
            'mouse_speed': 0,
            'turning_rate': 0,
            'keys_pressed': [],
            'left_click': False,
            'right_click': False,
            'is_shooting': False,
            'movement_keys': 0,
            # Network metrics
            'ping_ms': 0,
            'download_mbps': 0,
            'upload_mbps': 0,
            'packet_loss': 0,
            'bytes_sent': 0,
            'bytes_recv': 0,
            # System metrics
            'cpu_percent': 0,
            'cpu_freq_mhz': 0,
            'memory_percent': 0,
            'gpu_percent': 0,
            'gpu_memory_percent': 0,
            'gpu_temp': 0,
        }
        
        # Previous values for calculations
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0
        self.prev_time = time.time()
        self.prev_bytes_sent = 0
        self.prev_bytes_recv = 0
        
        # Control flags
        self.collecting = False
        self.combat_mode = False
        self.start_time = None
        
        # Network test settings
        self.last_ping_test = 0
        self.ping_test_interval = 2  # Test ping every 2 seconds
        
        # Game server IPs for FreeFire (Asia servers)
        self.game_servers = [
            '124.156.12.22',     # FreeFire Asia server
            '43.229.65.2',       # FreeFire Singapore
            '103.16.128.1',      # FreeFire India
            '8.8.8.8',           # Google DNS as backup
            '1.1.1.1'            # Cloudflare as backup
        ]
        
        # GPU monitoring method
        self.gpu_method = None
        self.init_gpu_monitoring()
        
        # Initialize network baseline
        self.init_network_baseline()
        
    def init_gpu_monitoring(self):
        """Initialize GPU monitoring with multiple fallback methods"""
        print("🎮 Initializing GPU monitoring...")
        
        # Method 1: Try GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_method = 'gputil'
                print(f"✅ GPU detected via GPUtil: {gpus[0].name}")
                return
        except:
            pass
        
        # Method 2: Try nvidia-smi for NVIDIA GPUs
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                self.gpu_method = 'nvidia-smi'
                print("✅ GPU detected via nvidia-smi")
                return
        except:
            pass
        
        # Method 3: Try Windows Performance Counters
        if platform.system() == 'Windows':
            try:
                import wmi
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name:
                        self.gpu_method = 'wmi'
                        print(f"✅ GPU detected via WMI: {gpu.Name}")
                        return
            except:
                pass
        
        print("⚠️ GPU monitoring not available - will estimate based on game activity")
        self.gpu_method = 'estimate'
    
    def init_network_baseline(self):
        """Get initial network metrics with better ping measurement"""
        print("🌐 Initializing network baseline...")
        
        # Test ping to game servers
        self.test_game_ping()
        
        # Get network interface stats
        net_io = psutil.net_io_counters()
        self.prev_bytes_sent = net_io.bytes_sent
        self.prev_bytes_recv = net_io.bytes_recv
    
    def test_game_ping(self):
        """Test ping to actual game servers"""
        ping_results = []
        
        for server in self.game_servers[:3]:  # Test first 3 servers
            try:
                if platform.system() == 'Windows':
                    result = subprocess.run(['ping', '-n', '1', '-w', '1000', server], 
                                          capture_output=True, text=True, timeout=2)
                else:  # Linux/Mac
                    result = subprocess.run(['ping', '-c', '1', '-W', '1', server], 
                                          capture_output=True, text=True, timeout=2)
                
                if 'time=' in result.stdout:
                    # Extract ping time
                    if platform.system() == 'Windows':
                        ping_str = result.stdout.split('time=')[1].split('ms')[0]
                    else:
                        ping_str = result.stdout.split('time=')[1].split(' ms')[0]
                    
                    ping_time = float(ping_str.strip())
                    ping_results.append(ping_time)
                    
            except Exception as e:
                continue
        
        if ping_results:
            # Use the best (lowest) ping
            best_ping = min(ping_results)
            self.current_state['ping_ms'] = round(best_ping, 1)
            print(f"📡 Game server ping: {best_ping:.1f}ms")
        else:
            # Fallback to localhost ping + offset
            self.current_state['ping_ms'] = 50.0  # Default to 50ms if can't measure
            print("📡 Using default ping estimate: 50ms")
    
    def get_network_metrics(self):
        """Get current network performance metrics"""
        try:
            # Get network throughput
            net_io = psutil.net_io_counters()
            time_diff = 1.0  # Assuming called every second
            
            bytes_sent_sec = (net_io.bytes_sent - self.prev_bytes_sent) / time_diff
            bytes_recv_sec = (net_io.bytes_recv - self.prev_bytes_recv) / time_diff
            
            self.current_state['bytes_sent'] = round(bytes_sent_sec / 1024, 2)  # KB/s
            self.current_state['bytes_recv'] = round(bytes_recv_sec / 1024, 2)  # KB/s
            
            self.prev_bytes_sent = net_io.bytes_sent
            self.prev_bytes_recv = net_io.bytes_recv
            
            # Test ping more frequently (every 2 seconds)
            current_time = time.time()
            if current_time - self.last_ping_test > self.ping_test_interval:
                self.last_ping_test = current_time
                # Run in background to avoid blocking
                ping_thread = threading.Thread(target=self.test_game_ping)
                ping_thread.daemon = True
                ping_thread.start()
                
        except Exception as e:
            pass
    
    def get_gpu_metrics(self):
        """Get GPU metrics using the best available method"""
        try:
            if self.gpu_method == 'gputil':
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.current_state['gpu_percent'] = round(gpu.load * 100, 1)
                    self.current_state['gpu_memory_percent'] = round(gpu.memoryUtil * 100, 1)
                    self.current_state['gpu_temp'] = round(gpu.temperature, 1)
                    return
            
            elif self.gpu_method == 'nvidia-smi':
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    if len(values) >= 3:
                        self.current_state['gpu_percent'] = float(values[0])
                        self.current_state['gpu_memory_percent'] = float(values[1])
                        self.current_state['gpu_temp'] = float(values[2])
                        return
            
            elif self.gpu_method == 'estimate':
                # Estimate GPU based on activity and combat
                base_gpu = 40 if not self.combat_mode else 70
                activity_bonus = min(self.current_state['mouse_speed'] / 50, 30)
                self.current_state['gpu_percent'] = round(base_gpu + activity_bonus + np.random.normal(0, 5), 1)
                self.current_state['gpu_memory_percent'] = round(65 + np.random.normal(0, 10), 1)
                self.current_state['gpu_temp'] = round(70 + np.random.normal(0, 5), 1)
                
        except Exception as e:
            # Fallback to estimation
            self.current_state['gpu_percent'] = round(50 + np.random.normal(0, 10), 1)
            self.current_state['gpu_memory_percent'] = 70
            self.current_state['gpu_temp'] = 72
    
    def get_system_metrics(self):
        """Get CPU and Memory metrics"""
        try:
            # CPU metrics
            self.current_state['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.current_state['cpu_freq_mhz'] = round(cpu_freq.current, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.current_state['memory_percent'] = round(memory.percent, 1)
            
            # Get GPU metrics
            self.get_gpu_metrics()
            
        except Exception as e:
            pass
    
    def on_move(self, x, y):
        """Track mouse movement"""
        if self.collecting:
            current_time = time.time()
            time_diff = current_time - self.prev_time
            
            if time_diff > 0:
                # Calculate mouse speed (pixels per second)
                distance = np.sqrt((x - self.prev_mouse_x)**2 + (y - self.prev_mouse_y)**2)
                speed = distance / time_diff
                
                # Calculate turning rate (horizontal movement)
                turning = abs(x - self.prev_mouse_x) / time_diff
                
                self.current_state['mouse_x'] = x
                self.current_state['mouse_y'] = y
                self.current_state['mouse_speed'] = speed
                self.current_state['turning_rate'] = turning
                
            self.prev_mouse_x = x
            self.prev_mouse_y = y
            self.prev_time = current_time
    
    def on_click(self, x, y, button, pressed):
        """Track mouse clicks (shooting)"""
        if self.collecting:
            if button == mouse.Button.left:
                self.current_state['left_click'] = pressed
                self.current_state['is_shooting'] = pressed
            elif button == mouse.Button.right:
                self.current_state['right_click'] = pressed
    
    def on_press(self, key):
        """Track keyboard presses"""
        if self.collecting:
            try:
                key_name = key.char.upper() if hasattr(key, 'char') else str(key)
                
                if key_name not in self.current_state['keys_pressed']:
                    self.current_state['keys_pressed'].append(key_name)
                
                # Count movement keys (WASD)
                movement_keys = ['W', 'A', 'S', 'D']
                self.current_state['movement_keys'] = sum(
                    1 for k in self.current_state['keys_pressed'] 
                    if k in movement_keys
                )
                
            except AttributeError:
                pass
        
        # Control commands
        try:
            if key == keyboard.Key.f1:  # Start/Stop collection
                self.toggle_collection()
            elif key == keyboard.Key.f2:  # Toggle combat mode
                self.toggle_combat_mode()
            elif key == keyboard.Key.f3:  # Show live stats
                self.show_live_stats()
            elif key == keyboard.Key.esc:  # Exit
                if self.collecting:
                    self.save_data()
                return False
        except:
            pass
    
    def on_release(self, key):
        """Track keyboard releases"""
        if self.collecting:
            try:
                key_name = key.char.upper() if hasattr(key, 'char') else str(key)
                if key_name in self.current_state['keys_pressed']:
                    self.current_state['keys_pressed'].remove(key_name)
                
                # Update movement keys count
                movement_keys = ['W', 'A', 'S', 'D']
                self.current_state['movement_keys'] = sum(
                    1 for k in self.current_state['keys_pressed'] 
                    if k in movement_keys
                )
            except:
                pass
    
    def toggle_collection(self):
        """Start or stop data collection"""
        self.collecting = not self.collecting
        if self.collecting:
            self.start_time = datetime.now()
            print(f"\n🟢 COLLECTION STARTED!")
            print("⏱️ NO TIME LIMIT - Press F1 again to stop")
            print("Press F2 to mark COMBAT periods")
            print("Press F3 to see live stats")
            print("Press ESC to exit and save")
        else:
            print("\n🔴 COLLECTION STOPPED!")
            self.save_data()
    
    def toggle_combat_mode(self):
        """Toggle combat labeling"""
        if self.collecting:
            self.combat_mode = not self.combat_mode
            if self.combat_mode:
                print("⚔️ COMBAT MODE ON - You're in combat!")
            else:
                print("🕊️ COMBAT MODE OFF - Exploration/Idle")
    
    def show_live_stats(self):
        """Display live performance stats"""
        if self.collecting and self.data:
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            
            print(f"\n📊 LIVE STATS:")
            print(f"⏱️ Time elapsed: {elapsed:.1f} minutes")
            print(f"📈 Data points: {len(self.data)}")
            print(f"💻 CPU: {self.current_state['cpu_percent']:.1f}%")
            print(f"🧠 Memory: {self.current_state['memory_percent']:.1f}%")
            print(f"🎮 GPU: {self.current_state['gpu_percent']:.1f}%")
            print(f"📡 Ping: {self.current_state['ping_ms']:.1f}ms")
            print(f"📥 Download: {self.current_state['bytes_recv']:.1f} KB/s")
            print(f"📤 Upload: {self.current_state['bytes_sent']:.1f} KB/s")
            print(f"⚔️ Combat mode: {'ON' if self.combat_mode else 'OFF'}")
    
    def collect_data_point(self):
        """Collect one data point every 100ms"""
        while True:
            if self.collecting:
                # Get system and network metrics
                self.get_system_metrics()
                self.get_network_metrics()
                
                # Calculate enhanced activity score
                activity_score = (
                    min(self.current_state['mouse_speed'] / 1000, 1) * 0.25 +
                    min(self.current_state['turning_rate'] / 500, 1) * 0.25 +
                    self.current_state['movement_keys'] * 0.1 +
                    self.current_state['is_shooting'] * 0.2 +
                    min(self.current_state['cpu_percent'] / 100, 1) * 0.1 +
                    min(self.current_state['gpu_percent'] / 100, 1) * 0.1
                )
                
                # Create data point with all metrics
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    # Player behavior
                    'mouse_x': self.current_state['mouse_x'],
                    'mouse_y': self.current_state['mouse_y'],
                    'mouse_speed': round(self.current_state['mouse_speed'], 2),
                    'turning_rate': round(self.current_state['turning_rate'], 2),
                    'is_shooting': self.current_state['is_shooting'],
                    'movement_keys': self.current_state['movement_keys'],
                    'keys_pressed': len(self.current_state['keys_pressed']),
                    'is_combat': self.combat_mode,
                    'activity_score': round(activity_score, 3),
                    # Network metrics
                    'ping_ms': self.current_state['ping_ms'],
                    'bytes_sent_kbs': self.current_state['bytes_sent'],
                    'bytes_recv_kbs': self.current_state['bytes_recv'],
                    # System metrics
                    'cpu_percent': self.current_state['cpu_percent'],
                    'cpu_freq_mhz': self.current_state['cpu_freq_mhz'],
                    'memory_percent': self.current_state['memory_percent'],
                    'gpu_percent': self.current_state['gpu_percent'],
                    'gpu_memory_percent': self.current_state['gpu_memory_percent'],
                    'gpu_temp': self.current_state['gpu_temp'],
                }
                
                self.data.append(data_point)
                
                # Print progress every 30 seconds
                if len(self.data) % 300 == 0:  # 300 * 0.1s = 30s
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    combat_pct = sum(1 for d in self.data if d['is_combat']) / len(self.data) * 100
                    print(f"📊 Progress: {elapsed:.1f} min | Points: {len(self.data)} | Combat: {combat_pct:.1f}% | CPU: {self.current_state['cpu_percent']:.1f}% | Ping: {self.current_state['ping_ms']:.1f}ms")
            
            time.sleep(0.1)  # Collect data every 100ms (10 Hz)
    
    def save_data(self):
        """Save collected data to CSV"""
        if self.data:
            df = pd.DataFrame(self.data)
            filename = f"enhanced_game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            print(f"\n💾 Data saved to {filename}")
            print(f"📈 Total data points: {len(self.data)}")
            
            duration_sec = len(self.data) * 0.1
            print(f"⏱️ Duration: {duration_sec/60:.1f} minutes")
            
            # Show comprehensive stats
            print("\n📊 SESSION STATISTICS:")
            print(f"⚔️ Combat time: {df['is_combat'].mean() * 100:.1f}%")
            print(f"🎯 Shooting events: {df['is_shooting'].sum()}")
            
            print("\n🌐 NETWORK PERFORMANCE:")
            print(f"📡 Avg ping: {df['ping_ms'].mean():.1f}ms (Min: {df['ping_ms'].min():.1f}, Max: {df['ping_ms'].max():.1f})")
            print(f"📥 Avg download: {df['bytes_recv_kbs'].mean():.1f} KB/s")
            print(f"📤 Avg upload: {df['bytes_sent_kbs'].mean():.1f} KB/s")
            
            print("\n💻 SYSTEM PERFORMANCE:")
            print(f"🖥️ Avg CPU: {df['cpu_percent'].mean():.1f}% (Max: {df['cpu_percent'].max():.1f}%)")
            print(f"🧠 Avg Memory: {df['memory_percent'].mean():.1f}%")
            print(f"🎮 Avg GPU: {df['gpu_percent'].mean():.1f}% (Max: {df['gpu_percent'].max():.1f}%)")
            
            # Identify performance issues
            high_ping = df[df['ping_ms'] > 100]
            if len(high_ping) > 0:
                print(f"\n⚠️ High ping events: {len(high_ping)} ({len(high_ping)/len(df)*100:.1f}% of time)")
    
    def display_system_info(self):
        """Display system information at startup"""
        print("\n💻 SYSTEM INFORMATION:")
        print(f"🖥️ Platform: {platform.system()} {platform.release()}")
        print(f"🧠 CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print(f"💾 Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if self.gpu_method == 'gputil':
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        print(f"🎮 GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
            except:
                pass
        elif self.gpu_method == 'nvidia-smi':
            print("🎮 GPU: NVIDIA GPU detected")
        elif self.gpu_method == 'estimate':
            print("🎮 GPU: Will estimate usage based on activity")
    
    def start(self):
        """Start the data collector"""
        print("=" * 70)
        print("🎮 ENHANCED GAME DATA COLLECTOR")
        print("=" * 70)
        
        self.display_system_info()
        
        print("\n📋 INSTRUCTIONS:")
        print("1. Start FreeFire or your FPS game")
        print("2. Press F1 to START recording (no time limit)")
        print("3. Press F2 when you're in COMBAT (fighting enemies)")
        print("4. Press F3 to see LIVE STATS")
        print("5. Press F1 again to STOP and save")
        print("6. Press ESC to exit")
        print("=" * 70)
        print("\n🚀 READY? Start your game and press F1!")
        
        # Start data collection thread
        data_thread = threading.Thread(target=self.collect_data_point)
        data_thread.daemon = True
        data_thread.start()
        
        # Start listeners
        mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click
        )
        keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        
        mouse_listener.start()
        keyboard_listener.start()
        
        # Keep running
        keyboard_listener.join()
        mouse_listener.stop()

if __name__ == "__main__":
    # Check and install required packages
    required_packages = ['psutil', 'pynput', 'pandas', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call(['pip', 'install', package])
    
    # Optional GPU monitoring package
    try:
        import GPUtil
    except ImportError:
        print("Installing GPUtil for GPU monitoring...")
        subprocess.check_call(['pip', 'install', 'gputil'])
    
    collector = EnhancedGameDataCollector()
    collector.start()