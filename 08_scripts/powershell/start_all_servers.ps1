Write-Host 'Starting all servers...' -ForegroundColor Green

# Start Edge Server
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd 04_deployment/01_servers/edge; python edge_server.py'

# Wait a bit
Start-Sleep -Seconds 2

# Start Cloud Server  
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd 04_deployment/01_servers/cloud; python cloud_server.py'

# Start Fog Server (if needed)
# Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd 04_deployment/01_servers/fog; python fog_server.py'

Write-Host 'All servers started!' -ForegroundColor Green
