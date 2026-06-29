# Deployment Guide

## Option 1: GitHub Pages (Recommended)

### Step 1: Push to Repository

```bash
cd /tmp/self-criteach-website
git init
git add .
git commit -m "Initial commit: Self-CriTeach project website"
git remote add origin https://github.com/markli1hoshipu/self-criteach-website.git
git push -u origin main
```

### Step 2: Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source: Select "Deploy from a branch"
3. Branch: Select `main` → `/` (root)
4. Click Save

### Step 3: Configure Base URL

If deploying to `username.github.io/self-criteach`, the site will be available at:
```
https://markli1hoshipu.github.io/self-criteach
```

The `_config.yml` is already configured with:
```yaml
baseurl: "/self-criteach"
url: "https://markli1hoshipu.github.io"
```

### Step 4: Wait for Deployment

GitHub Actions will automatically build and deploy. Check the Actions tab for progress.

## Option 2: Custom Domain

### Step 1: Add CNAME File

```bash
echo "selfcriteach.ai" > CNAME
```

### Step 2: Configure DNS

Add these DNS records at your domain provider:

**A Records:**
```
@    A    185.199.108.153
@    A    185.199.109.153
@    A    185.199.110.153
@    A    185.199.111.153
```

**CNAME Record:**
```
www  CNAME  markli1hoshipu.github.io
```

### Step 3: Update _config.yml

```yaml
url: "https://selfcriteach.ai"
baseurl: ""
```

### Step 4: Enable Custom Domain in GitHub

1. Go to repository Settings → Pages
2. Custom domain: Enter `selfcriteach.ai`
3. Check "Enforce HTTPS"

## Option 3: Netlify

### Step 1: Connect Repository

1. Sign up at https://netlify.com
2. Click "Add new site" → "Import an existing project"
3. Connect to GitHub and select repository

### Step 2: Configure Build

**Build settings:**
- Build command: `jekyll build`
- Publish directory: `_site`

**Environment variables:**
- `JEKYLL_ENV` = `production`

### Step 3: Deploy

Netlify will automatically deploy on every push to main branch.

**Features:**
- Automatic HTTPS
- Deploy previews for pull requests
- Custom domain support
- Edge CDN

## Option 4: Vercel

Similar to Netlify:

1. Import repository at https://vercel.com
2. Framework preset: Jekyll
3. Build command: `jekyll build`
4. Output directory: `_site`

## Local Testing Before Deployment

```bash
# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve --livereload

# Open http://localhost:4000/self-criteach
```

## Troubleshooting

### Jekyll Build Fails on GitHub Pages

- Ensure `Gemfile.lock` is committed
- Check that all plugins are GitHub Pages compatible
- Review build logs in Actions tab

### CSS Not Loading

- Verify `baseurl` in `_config.yml` matches your deployment path
- Check browser console for 404 errors
- Ensure `assets/css/style.css` exists

### Links Not Working

- Use relative links with `{{ site.baseurl }}`
- Test locally with `jekyll serve`
- Clear browser cache

## Performance Optimization

### Before Deployment

1. **Optimize Images:**
   ```bash
   # Install imagemagick
   brew install imagemagick  # Mac
   apt-get install imagemagick  # Linux

   # Optimize images
   find assets/images -name "*.png" -exec convert {} -strip -quality 85 {} \;
   ```

2. **Minify CSS (optional):**
   ```bash
   npm install -g cssnano-cli
   cssnano assets/css/style.css assets/css/style.min.css
   ```

3. **Test Performance:**
   - Google Lighthouse
   - WebPageTest
   - GTmetrix

## Security

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Enable HTTPS (automatic on GitHub Pages)
- Regular dependency updates: `bundle update`

## Monitoring

### Google Analytics (Optional)

Add to `_config.yml`:
```yaml
google_analytics: UA-XXXXXXXXX-X
```

Then add tracking code to `_includes/google-analytics.html`.

### Uptime Monitoring

- UptimeRobot (free)
- Pingdom
- StatusCake

## Maintenance

### Regular Updates

```bash
# Update dependencies
bundle update

# Check for security issues
bundle audit

# Test locally
bundle exec jekyll serve
```

### Content Updates

1. Edit `index.html` or markdown files
2. Commit and push
3. GitHub/Netlify/Vercel will auto-deploy

## Rollback

If deployment fails:

```bash
# Revert to previous commit
git revert HEAD
git push

# Or roll back to specific commit
git reset --hard <commit-hash>
git push --force
```

## Support

- Jekyll Documentation: https://jekyllrb.com/docs/
- GitHub Pages: https://docs.github.com/pages
- Netlify Docs: https://docs.netlify.com/
