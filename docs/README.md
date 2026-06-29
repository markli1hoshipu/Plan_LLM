# Self-CriTeach Project Website

Official project website for "Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation"

## Local Development

### Prerequisites

- Ruby 2.7 or higher
- Bundler

### Setup

```bash
# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Open browser to http://localhost:4000/self-criteach
```

### Building for Production

```bash
bundle exec jekyll build
```

The site will be generated in the `_site` directory.

## Deployment

### GitHub Pages (Automatic)

1. Push to `gh-pages` branch
2. Enable GitHub Pages in repository settings
3. Set source to `gh-pages` branch

### Manual Deployment

Copy the contents of `_site/` to your web server.

## Structure

```
self-criteach-website/
├── index.html              # Main page
├── assets/
│   ├── css/style.css      # Custom styling
│   ├── js/                # JavaScript files
│   ├── images/            # Images and figures
│   └── data/              # Data for visualizations
├── papers/
│   └── self-criteach.pdf  # Paper PDF
├── _config.yml            # Jekyll configuration
└── Gemfile                # Ruby dependencies
```

## Customization

### Colors

Edit CSS variables in `assets/css/style.css`:

```css
:root {
  --primary-color: #1a365d;
  --accent-color: #3182ce;
  --success-color: #38a169;
  /* ... */
}
```

### Content

Edit sections directly in `index.html` or create separate markdown files in `_includes/`.

## Features

- Responsive design (mobile-friendly)
- Smooth scrolling navigation
- Copy-to-clipboard for BibTeX
- Fade-in animations
- Professional academic styling

## License

Website code is licensed under MIT License.
Content and paper are copyright of the authors.
