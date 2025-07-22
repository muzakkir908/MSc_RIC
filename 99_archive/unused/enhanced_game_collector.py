import time
import pandas as pd
import numpy as np
from pynput import mouse, keyboard
import threading
import json
from datetime import datetime, timedelta
import psutil
import socket
import speedtest
import platform
import GPUtil

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
            # New network metrics
            'ping_ms': 0,
            'download_mbps': 0,
            'upload_mbps': 0,
            'packet_loss': 0,
            'bytes_sent': 0,
            'bytes_recv': 0,
            # New system metrics
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
        self.duration_minutes = 30  # 30 minutes collection
        
        # Network test interval (every 30 seconds)
        self.last_network_test = 0
        self.network_test_interval = 30
        
        # Initialize network baseline
        self.init_network_baseline()
        
    def init_network_baseline(self):
        """Get initial network metrics"""
        print("🌐 Initializing network baseline...")
        try:
            # Quick ping test
            import subprocess
            result = subprocess.run(['ping', '-n', '1', '8.8.8.8'], 
                                  capture_output=True, text=True, timeout=5)
            if 'time=' in result.stdout:
                ping_time = result.stdout.split('time=')[1].split('ms')[0]
                self.current_state['ping_ms'] = float(ping_time)
                print(f"📡 Initial ping: {ping_time}ms")
        except:
            print("⚠️ Could not measure ping")
            
        # Get network interface stats
        net_io = psutil.net_io_counters()
        self.prev_bytes_sent = net_io.bytes_sent
        self.prev_bytes_recv = net_io.bytes_recv
    
    def get_network_metrics(self):
        """Get current network performance metrics"""
        try:
            # Get network throughput
            net_io = psutil.net_io_counters()
            bytes_sent_sec = (net_io.bytes_sent - self.prev_bytes_sent) / 1
            bytes_recv_sec = (net_io.bytes_recv - self.prev_bytes_recv) / 1
            
            self.current_state['bytes_sent'] = round(bytes_sent_sec / 1024, 2)  # KB/s
            self.current_state['bytes_recv'] = round(bytes_recv_sec / 1024, 2)  # KB/s
            
            self.prev_bytes_sent = net_io.bytes_sent
            self.prev_bytes_recv = net_io.bytes_recv
            
            # Periodic comprehensive network test
            current_time = time.time()
            if current_time - self.last_network_test > self.network_test_interval:
                self.last_network_test = current_time
                self.run_network_test()
                
        except Exception as e:
            pass
    
    def run_network_test(self):
        """Run comprehensive network test (every 30 seconds)"""
        def test_network():
            try:
                print("🔄 Running network test...")
                # Quick ping test to game servers
                import subprocess
                
                # Ping to common game server locations
                servers = {
                    'Google DNS': '8.8.8.8',
                    'Cloudflare': '1.1.1.1',
                    'Local Router': '192.168.1.1'
                }
                
                ping_results = []
                for name, server in servers.items():
                    try:
                        result = subprocess.run(['ping', '-n', '1', server], 
                                              capture_output=True, text=True, timeout=2)
                        if 'time=' in result.stdout:
                            ping_time = float(result.stdout.split('time=')[1].split('ms')[0])
                            ping_results.append(ping_time)
                    except:
                        pass
                
                if ping_results:
                    avg_ping = sum(ping_results) / len(ping_results)
                    self.current_state['ping_ms'] = round(avg_ping, 1)
                    print(f"📡 Avg ping: {avg_ping:.1f}ms")
                    
            except Exception as e:
                print(f"⚠️ Network test error: {e}")
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=test_network)
        thread.daemon = True
        thread.start()
    
    def get_system_metrics(self):
        """Get CPU, Memory, and GPU metrics"""
        try:
            # CPU metrics
            self.current_state['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.current_state['cpu_freq_mhz'] = round(cpu_freq.current, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.current_state['memory_percent'] = round(memory.percent, 1)
            
            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    self.current_state['gpu_percent'] = round(gpu.load * 100, 1)
                    self.current_state['gpu_memory_percent'] = round(gpu.memoryUtil * 100, 1)
                    self.current_state['gpu_temp'] = round(gpu.temperature, 1)
            except:
                # If GPUtil fails, try alternative method
                pass
                
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
            print(f"\n🟢 COLLECTION STARTED! Will run for {self.duration_minutes} minutes")
            print("Press F2 to mark COMBAT periods")
            print("Press F3 to see live stats")
            print(f"⏰ Will auto-stop at {(self.start_time + timedelta(minutes=self.duration_minutes)).strftime('%H:%M:%S')}")
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
            remaining = self.duration_minutes - elapsed
            
            print(f"\n📊 LIVE STATS:")
            print(f"⏱️ Time: {elapsed:.1f}/{self.duration_minutes} min (Remaining: {remaining:.1f} min)")
            print(f"📈 Data points: {len(self.data)}")
            print(f"💻 CPU: {self.current_state['cpu_percent']:.1f}%")
            print(f"🧠 Memory: {self.current_state['memory_percent']:.1f}%")
            print(f"🎮 GPU: {self.current_state['gpu_percent']:.1f}%")
            print(f"📡 Ping: {self.current_state['ping_ms']:.1f}ms")
            print(f"📥 Download: {self.current_state['bytes_recv']:.1f} KB/s")
            print(f"📤 Upload: {self.current_state['bytes_sent']:.1f} KB/s")
    
    def collect_data_point(self):
        """Collect one data point every 100ms"""
        while True:
            if self.collecting:
                # Check if 30 minutes have passed
                if (datetime.now() - self.start_time).total_seconds() >= self.duration_minutes * 60:
                    print(f"\n⏰ {self.duration_minutes} minutes reached! Stopping collection...")
                    self.collecting = False
                    self.save_data()
                    break
                
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
                    print(f"📊 Progress: {elapsed:.1f}/{self.duration_minutes} min | Points: {len(self.data)} | CPU: {self.current_state['cpu_percent']:.1f}% | Ping: {self.current_state['ping_ms']:.1f}ms")
            
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
            
            high_cpu = df[df['cpu_percent'] > 80]
            if len(high_cpu) > 0:
                print(f"⚠️ High CPU events: {len(high_cpu)} ({len(high_cpu)/len(df)*100:.1f}% of time)")
    
    def display_system_info(self):
        """Display system information at startup"""
        print("\n💻 SYSTEM INFORMATION:")
        print(f"🖥️ Processor: {platform.processor()}")
        print(f"🧠 CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print(f"💾 Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"🎮 GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
        except:
            print("🎮 GPU: Detection not available")
    
    def start(self):
        """Start the data collector"""
        print("=" * 70)
        print("🎮 ENHANCED GAME DATA COLLECTOR - 30 MINUTE SESSION")
        print("=" * 70)
        
        self.display_system_info()
        
        print("\n📋 INSTRUCTIONS:")
        print("1. Start your FPS game (CS:GO, Valorant, Krunker.io, etc.)")
        print("2. Press F1 to START recording (will run for 30 minutes)")
        print("3. Press F2 during COMBAT (fighting enemies)")
        print("4. Press F3 to see LIVE STATS")
        print("5. Press ESC to exit early")
        print("\n✨ NEW FEATURES:")
        print("- Network performance monitoring (ping, bandwidth)")
        print("- CPU/GPU usage tracking")
        print("- Automatic 30-minute collection")
        print("- Enhanced activity scoring")
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
    required_packages = ['psutil', 'GPUtil']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess
            subprocess.check_call(['pip', 'install', package])
    
    collector = EnhancedGameDataCollector()
    collector.start()