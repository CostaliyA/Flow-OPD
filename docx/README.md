# Flow-OPD: On-Policy Distillation for Generalist Flow Matching Text-to-Image Generation

**NeurIPS 2026**

A Jekyll-based research paper website for [Flow-OPD](https://github.com/CostaliyA/Flow-OPD) — the first unified post-training framework that integrates on-policy distillation into Flow Matching models.

## Live Demo

> Coming soon — deploy to GitHub Pages using the instructions below.

## Quick Start (Local Development)

### Prerequisites

- Ruby ≥ 2.7
- Bundler

### Setup

```bash
cd jekyll-site

# Install dependencies
bundle install

# Start local server
bundle exec jekyll serve

# Open http://localhost:4000
```

## Deploy to GitHub Pages

### Option 1: Use GitHub Actions (Recommended)

The repository already includes a GitHub Actions workflow. Push to `main` and GitHub Pages will automatically build and deploy the site.

Make sure your repository settings enable GitHub Pages from the `gh-pages` branch.

### Option 2: Manual Build

```bash
cd jekyll-site
bundle exec jekyll build

# The built site is in _site/
# You can serve it with any static file server
```

### Option 3: GitHub Pages with `github-pages` gem

Ensure your `_config.yml` uses the `github-pages` gem group. GitHub Pages will automatically build the site on push.

## Site Structure

```
jekyll-site/
├── _config.yml          # Jekyll configuration
├── _layouts/
│   └── default.html     # Main page layout with navigation
├── assets/
│   └── css/
│       └── main.css     # Styles
├── index.md             # Homepage (abstract, contributions)
├── methods.md           # Methodology page
├── experiments.md       # Experiments & analysis page
├── results.md           # Quantitative results page
├── 404.md               # 404 page
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## Customization

### Update Paper Information

Edit `_config.yml` to update:
- `title`, `description`, `url`
- Author list and affiliations
- GitHub repository URL
- Conference information

### Add Your Own Images

Place images in `assets/images/` and reference them with:

```liquid
{{ '/assets/images/your-image.png' | relative_url }}
```

## Technologies Used

- [Jekyll](https://jekyllrb.com/) — static site generator
- [GitHub Pages](https://pages.github.com/) — free hosting
- [Kramdown](https://kramdown.gettalong.org/) — Markdown processor
- Google Fonts (Inter, JetBrains Mono)

## Citation

```bibtex
@article{flowopd2026,
  title={Flow-OPD: On-Policy Distillation for Generalist Flow Matching Text-to-Image Generation},
  author={Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, Feng Zhao},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

MIT
