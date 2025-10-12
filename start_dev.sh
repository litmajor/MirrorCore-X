
#!/bin/bash

echo "Starting MirrorCore-X Development Environment..."

# Start frontend in background
echo "Starting frontend on port 3000..."
npm run dev &

# Start dashboard server in background  
echo "Starting dashboard WebSocket server on port 5000..."
python dashboard_server.py &

# Start API server
echo "Starting API server on port 8000..."
python api.py

# Keep script running
wait
