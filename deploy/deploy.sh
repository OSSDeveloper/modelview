#!/bin/bash
# Deploy script for prd_vps

set -e

echo "=== Model View Deployment ==="
echo "Target: prd_vps (135.125.102.151)"
echo ""

# Check if Docker is running on remote
ssh mcp_ssh_manager_ssh_execute server='prd_vps' 'docker --version && docker-compose --version'

# Create deployment directory
ssh mcp_ssh_manager_ssh_execute server='prd_vps' 'mkdir -p ~/modelview'

# Copy project files (excluding .git and node_modules)
rsync -avz --exclude='.git' --exclude='node_modules' --exclude='__pycache__' \
  ~/Desktop/model\ view/ \
  mcp_ssh_manager_ssh_execute server='prd_vps':~/modelview/

# Build and start containers
ssh mcp_ssh_manager_ssh_execute server='prd_vps' 'cd ~/modelview && docker-compose down 2>/dev/null || true'
ssh mcp_ssh_manager_ssh_execute server='prd_vps' 'cd ~/modelview && docker-compose up --build -d'

# Verify deployment
sleep 10
ssh mcp_ssh_manager_ssh_execute server='prd_vps' 'curl -f http://localhost:8000/health && curl -f http://localhost:3000'

echo ""
echo "=== Deployment Complete ==="
echo "Frontend: http://135.125.102.151:3000"
echo "Backend API: http://135.125.102.151:8000"
echo "API Docs: http://135.125.102.151:8000/docs"