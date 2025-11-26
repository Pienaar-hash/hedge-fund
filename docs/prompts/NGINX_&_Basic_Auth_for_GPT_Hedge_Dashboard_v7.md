# PATCH SCOPE: NGINX + Basic Auth for GPT-Hedge Dashboard v7
#
# GOAL:
#   - Add secure investor-facing endpoint for the Streamlit dashboard.
#   - Require HTTP Basic Auth for all dashboard pages.
#   - Expose health endpoints WITHOUT authentication.
#   - Preserve WebSockets (Streamlit requires this).
#
# TARGET FILE:
#   /etc/nginx/sites-available/hedge-dashboard
#
# ACTIONS:
#   1. Replace any existing server block with the configuration below.
#   2. Set up Basic Auth via /etc/nginx/.htpasswd (manual step noted).
#   3. Ensure reverse-proxy forwarding is Streamlit-compatible.
#
# PATCH (FULL FILE REPLACEMENT):
#
#   server {
#       listen 80;
#       server_name _;
#
#       # Redirect all HTTP to HTTPS (if SSL is enabled later)
#       # Uncomment when SSL cert is installed:
#       # return 301 https://$host$request_uri;
#
#       # Health endpoints (NO AUTH)
#       location /health {
#           proxy_pass http://localhost:8501/health;
#       }
#
#       location /ping {
#           proxy_pass http://localhost:8501/ping;
#       }
#
#       location /status {
#           proxy_pass http://localhost:8501/status;
#       }
#
#       # Protected dashboard prefix
#       location /hedge/ {
#
#           # === BASIC AUTH ===
#           auth_basic "Restricted Hedge Dashboard";
#           auth_basic_user_file /etc/nginx/.htpasswd;
#
#           # === REQUIRED FOR STREAMLIT ===
#           proxy_set_header Host $host;
#           proxy_set_header X-Real-IP $remote_addr;
#           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#           proxy_set_header X-Forwarded-Proto $scheme;
#
#           # WebSocket upgrade
#           proxy_http_version 1.1;
#           proxy_set_header Upgrade $http_upgrade;
#           proxy_set_header Connection "upgrade";
#
#           # Path rewrite: strip "/hedge/"
#           proxy_pass http://localhost:8501/;
#       }
#   }
#
# POST-PATCH ACTIONS (manual):
#   - Create .htpasswd with:
#         sudo htpasswd -c /etc/nginx/.htpasswd investor1
#   - Test configuration:
#         sudo nginx -t
#   - Reload:
#         sudo systemctl reload nginx

# == Patch Notes == #
• - Installed the requested NGINX site config at /etc/nginx/sites-available/hedge-dashboard with the
    provided server block:
      - /health, /ping, /status proxied without auth.
      - /hedge/ proxied to the Streamlit dashboard with required headers/WebSocket upgrade and HTTP
        Basic Auth via /etc/nginx/.htpasswd.
      - HTTPS redirect stub left commented per instructions.
  - Created the sites-available directory as needed.

  Note: You still need to create the htpasswd file, test, and reload NGINX:

  - sudo htpasswd -c /etc/nginx/.htpasswd <user>
  - sudo nginx -t && sudo systemctl reload nginx

• - Installed the requested NGINX site config at /etc/nginx/sites-available/hedge-dashboard with the
    provided server block:
      - /health, /ping, /status proxied without auth.
      - /hedge/ proxied to the Streamlit dashboard with required headers/WebSocket upgrade and HTTP
        Basic Auth via /etc/nginx/.htpasswd.
      - HTTPS redirect stub left commented per instructions.
  - sudo nginx -t && sudo systemctl reload nginx

  If you want me to enable the site (symlink into sites-enabled) or run the test/reload, let me
  know.

# 🛠 5. DEPLOYMENT INSTRUCTIONS (You perform these on the Hetzner box)

### **1. Apply the patch via Codex**

Let Codex write/update:

```
/etc/nginx/sites-available/hedge-dashboard
```

### **2. Enable site (if not yet enabled)**

```
sudo ln -s /etc/nginx/sites-available/hedge-dashboard /etc/nginx/sites-enabled/hedge-dashboard
```

Remove the default site if needed:

```
sudo rm /etc/nginx/sites-enabled/default
```

### **3. Create Basic Auth credentials**

```
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd investor1
```

Creates a password prompt.
Repeat for more investors:

```
sudo htpasswd /etc/nginx/.htpasswd investor2
sudo htpasswd /etc/nginx/.htpasswd analystA
```

### **4. Test NGINX**

```
sudo nginx -t
```

### **5. Reload NGINX**

```
sudo systemctl reload nginx
```

### **6. Access dashboard**

```
http://<server-ip>/hedge
```

You should see a username/password prompt.
Once authenticated, the Streamlit dashboard loads normally.

Health endpoints accessible:

```
http://<server-ip>/health
http://<server-ip>/ping
http://<server-ip>/status
```

---

# 📡 6. SSL / HTTPS (Optional Follow-up)

When ready to enable HTTPS:

1. Add domain (A record → server IP)
2. Install certbot:

```
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

3. Switch the listener to:

```
listen 443 ssl;
```