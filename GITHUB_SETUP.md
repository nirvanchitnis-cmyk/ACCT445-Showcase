# How to Push to GitHub

## 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `ACCT445-Showcase`
3. Description: `Bank Disclosure Opacity & Market Performance Analysis`
4. Visibility: **Public** (for academic portfolio) or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## 2. Push to GitHub

```bash
cd /Users/nirvanchitnis/ACCT445-Showcase

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/ACCT445-Showcase.git

# Push to GitHub
git push -u origin main
```

## 3. Update README with Your Info

Edit `README.md` and replace placeholders:
- `[Your University]` → Your actual university
- `[Your Email]` → Your contact email
- `YOUR_USERNAME` → Your GitHub username (in URLs)

Then commit and push:

```bash
git add README.md
git commit -m "Update README with personal info"
git push
```

## 4. Enable GitHub Pages (Optional)

To host documentation:

1. Go to repository Settings → Pages
2. Source: Deploy from branch `main`
3. Folder: `/` (root)
4. Save

Your documentation will be live at:
`https://YOUR_USERNAME.github.io/ACCT445-Showcase/`

## 5. Add Topics/Tags

Go to repository homepage → Click gear icon next to "About" → Add topics:
- `finance`
- `econometrics`
- `disclosure-quality`
- `event-study`
- `panel-data`
- `bank-regulation`
- `cecl`
- `accounting-research`

## 6. Create GitHub Release (Optional)

For v1.0.0 release:

```bash
git tag -a v1.0.0 -m "First release: ACCT445 Bank Disclosure Showcase"
git push origin v1.0.0
```

Then go to GitHub → Releases → Draft a new release → Select `v1.0.0` tag

## Repository Stats

- **Size:** 672 KB (lightweight!)
- **Files:** 14 source files
- **Sample data:** 336 KB (411 filings, 40 banks)
- **Dependencies:** 11 core packages (all standard)
- **No large files:** ✓ GitHub-friendly

## What's Included

✅ **Code:**
- CIK → Ticker mapper (SEC API)
- Decile backtest with Newey-West SEs
- Event study (SVB collapse)
- Data loading utilities

✅ **Documentation:**
- Comprehensive README with findings
- Quickstart Jupyter notebook
- Inline code documentation

✅ **Sample Data:**
- Top 20 + Bottom 20 banks by CNOI
- 411 SEC filings (2023-2025)
- ~80% smaller than full dataset

✅ **Best Practices:**
- MIT License
- .gitignore for Python/Jupyter
- pyproject.toml + requirements.txt
- Git history with co-authorship

## Next Steps After Pushing

1. **Star the repo** (if public) to show it's active
2. **Add a banner image** (optional): Create `docs/banner.png` showing sample results
3. **Write blog post** (optional): Medium/LinkedIn article linking to repo
4. **Share on LinkedIn** with hashtags: #Finance #Econometrics #DataScience
5. **Add to CV/Resume** under "Projects" section

---

**Questions?** Check GitHub docs: https://docs.github.com/en/get-started
