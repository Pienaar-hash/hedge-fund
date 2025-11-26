## 🔥 Codex Patch Prompt — NGINX + SSL / HTTPS for Hedge Dashboard

Paste this into Codex IDE/CLI:

````text
# PATCH SCOPE: NGINX + SSL / HTTPS for Hedge Dashboard (v7)
#
# GOAL:
#   Provide an NGINX config + runbook so the hedge dashboard runs behind:
#     - HTTPS (Let’s Encrypt)
#     - Basic Auth (investors)
#     - Reverse proxy → Streamlit on localhost:8501
#     - /health endpoint open (no auth) for probes
#
# FILES TO ADD:
#   - docs/v7_NGINX_SSL_Dashboard_Runbook.md
#   - ops/nginx/hedge_dashboard.conf.example
#
# No changes to Python code. This is ops / docs only.

# --------------------------------------------------------------------
# 1) Create docs/v7_NGINX_SSL_Dashboard_Runbook.md

# CONTENT (create the file with this markdown):

"""
# v7 NGINX + SSL / HTTPS Deployment — Hedge Dashboard

This runbook describes how to expose the Hedge v7 dashboard on HTTPS with
Basic Auth via NGINX on the Hetzner Ubuntu box.

## 1. Assumptions

- OS: Ubuntu (Hetzner VPS).
- Hedge repo at: `/root/hedge-fund`.
- Streamlit dashboard running via supervisor on `localhost:8501`.
- Domain name pointing to this box, e.g. `hedge.example.com`.
- NGINX and Certbot available from apt.

Adjust paths/usernames as needed.

## 2. Install NGINX + Certbot

```bash
sudo apt update
sudo apt install -y nginx python3-certbot-nginx
````

Ensure NGINX is running:

```bash
sudo systemctl status nginx
```

## 3. Create NGINX site configuration

We keep a template in the repo:

* `ops/nginx/hedge_dashboard.conf.example`

Copy it to NGINX config directory on the server:

```bash
sudo cp /root/hedge-fund/ops/nginx/hedge_dashboard.conf.example /etc/nginx/sites-available/hedge-dashboard.conf
```

Edit `/etc/nginx/sites-available/hedge-dashboard.conf` and set:

* `server_name hedge.example.com;`
* Adjust any paths if your repo lives elsewhere.

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/hedge-dashboard.conf /etc/nginx/sites-enabled/hedge-dashboard.conf
sudo nginx -t
sudo systemctl reload nginx
```

At this point, HTTP (port 80) should proxy to the dashboard on port 8501.

## 4. Configure Basic Auth

Create an htpasswd file (replace `investor` with desired username):

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd-hedge investor
# Enter password when prompted
```

The NGINX config references this file for Basic Auth on all routes except `/health`.

## 5. Obtain Let’s Encrypt TLS Certificate

Run Certbot with NGINX plugin:

```bash
sudo certbot --nginx -d hedge.example.com
```

Follow prompts:

* Select the `hedge-dashboard` server block
* Choose option to redirect HTTP → HTTPS

Certbot will:

* Obtain certificates under `/etc/letsencrypt/live/hedge.example.com/`
* Inject `ssl_certificate` and `ssl_certificate_key` into the NGINX config
* Add HTTP→HTTPS redirects.

Auto-renewal is handled by `certbot.timer`:

```bash
systemctl status certbot.timer
sudo certbot renew --dry-run
```

## 6. Health Endpoint (Unauthenticated)

The NGINX config exposes `/health` without Basic Auth so uptime probes and
load balancers can check the dashboard endpoint.

By default it proxies to:

* `http://127.0.0.1:8501/health`

If your dashboard does not expose `/health`, you can either:

* Add a simple health route, or
* Point the upstream location to `/` (it will just check HTTP 200).

## 7. Firewall / Ports

Ensure:

* Port 80 (HTTP) and 443 (HTTPS) are open in Hetzner firewall / ufw.
* Internal dashboard port 8501 is bound to 127.0.0.1 (only NGINX can reach it).

Example ufw:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

## 8. Supervisor / Process Layout (Reference)

Supervisor already manages:

* hedge-dashboard → Streamlit on :8501
* hedge-executor, hedge-sync_state, etc.

No change is required here; NGINX just proxies to 127.0.0.1:8501.

## 9. Verification

1. Check NGINX syntax:

   ```bash
   sudo nginx -t
   ```

2. Visit:

   * `http://hedge.example.com` – should redirect to HTTPS
   * `https://hedge.example.com` – should prompt for Basic Auth
   * `https://hedge.example.com/health` – should NOT require auth

3. Confirm TLS details are correct in the browser (valid cert, correct domain).

## 10. Troubleshooting

* Check NGINX logs:

  ```bash
  sudo tail -n 100 /var/log/nginx/error.log
  sudo tail -n 100 /var/log/nginx/access.log
  ```

* Check hedge dashboard logs:

  ```bash
  sudo tail -n 100 /var/log/hedge-dashboard.err.log
  sudo tail -n 100 /var/log/hedge-dashboard.out.log
  ```

* If Certbot fails HTTP-01 challenge:

  * Confirm DNS A record points to the correct server IP.
  * Make sure no firewall blocks port 80.
    """

# --------------------------------------------------------------------

# 2) Create ops/nginx/hedge_dashboard.conf.example

# CONTENT:

"""

# NGINX reverse proxy for Hedge v7 Dashboard

# Template: copy to /etc/nginx/sites-available/hedge-dashboard.conf

# and enable via symlink into sites-enabled.

upstream hedge_dashboard_upstream {
server 127.0.0.1:8501;
}

server {
listen 80;
listen [::]:80;
server_name hedge.example.com;

```
# Redirect HTTP to HTTPS will be injected by certbot, or can be
# manually added later. Initially, we proxy HTTP to test.

# Health endpoint (no auth) for uptime checks
location /health {
    proxy_pass http://hedge_dashboard_upstream/health;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

# Everything else proxied to Streamlit dashboard, with Basic Auth
location / {
    auth_basic           "Hedge Investor Dashboard";
    auth_basic_user_file /etc/nginx/.htpasswd-hedge;

    proxy_pass http://hedge_dashboard_upstream/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 300;
}

# Basic security headers
add_header X-Frame-Options "DENY";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "no-referrer-when-downgrade";

# Optional: HSTS (enable after HTTPS works)
# add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
```

}

"""

# --------------------------------------------------------------------

# END PATCH SCOPE

```

---

## ✅ What you do now on the box

1. **Pull latest code** with these new files.  
2. Copy the example NGINX config into place, edit `server_name`.  
3. Create the htpasswd file.  
4. Run `certbot --nginx -d your-domain`.  
5. Verify HTTP→HTTPS redirect + Basic Auth + `/health` open.
