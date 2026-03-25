"""
scraper.py - Web Scraper for IIT Jodhpur Website
=================================================
Scrapes textual data from IIT Jodhpur official website across multiple categories:
1. Department pages (CSE, EE, ME, Chemistry, Physics, etc.)
2. Academic regulation / programs pages (required)
3. Research pages and highlights
4. Faculty profile pages
5. About / institute pages
6. News and announcements

Usage:
    python problem1/scraper.py
    
Output:
    problem1/raw/*.txt  — one text file per scraped page
"""

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_URL = "https://iitj.ac.in"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Delay between requests (seconds) — be respectful to the server
REQUEST_DELAY = 1.0

# ──────────────────────────────────────────────
# URLs organized by source category
# ──────────────────────────────────────────────
URLS = {
    # ── Source 1: Department Pages ──
    "dept_cse_about": f"{BASE_URL}/computer-science-engineering/en/about-department",
    "dept_cse_research": f"{BASE_URL}/computer-science-engineering/en/about-research",
    "dept_cse_programs": f"{BASE_URL}/computer-science-engineering/en/academic-programs",
    "dept_ee_about": f"{BASE_URL}/electrical-engineering/en/about-department",
    "dept_ee_research": f"{BASE_URL}/electrical-engineering/en/about-research",
    "dept_me_about": f"{BASE_URL}/mechanical-engineering/en/about-department",
    "dept_me_research": f"{BASE_URL}/mechanical-engineering/en/about-research",
    "dept_chemistry_about": f"{BASE_URL}/chemistry/en/about-department",
    "dept_chemistry_research": f"{BASE_URL}/chemistry/en/about-research",
    "dept_physics_about": f"{BASE_URL}/physics/en/about-department",
    "dept_physics_research": f"{BASE_URL}/physics/en/about-research",
    "dept_math_about": f"{BASE_URL}/mathematics/en/about-department",
    "dept_math_research": f"{BASE_URL}/mathematics/en/about-research",
    "dept_bio_about": f"{BASE_URL}/bioscience-bioengineering/en/about-department",
    "dept_bio_research": f"{BASE_URL}/bioscience-bioengineering/en/about-research",
    "dept_civil_about": f"{BASE_URL}/civil-and-infrastructure-engineering/en/about-department",
    "dept_metallurgy_about": f"{BASE_URL}/metallurgical-materials-engineering/en/about-department",
    "dept_chemical_about": f"{BASE_URL}/chemical-engineering/en/about-department",
    "dept_digital_humanities": f"{BASE_URL}/digital-humanities/en/about-department",
    "dept_digital_humanities_research": f"{BASE_URL}/digital-humanities/en/about-research-2",
    
    # ── Source 2: Academic Regulations & Programs (Required) ──
    "academics_main": f"{BASE_URL}/office-of-academics/en/academics",
    "academic_programs": f"{BASE_URL}/office-of-academics/en/academic-programs",
    "academic_regulations": f"{BASE_URL}/office-of-academics/en/Academic-Regulations",
    "academic_calendar": f"{BASE_URL}/office-of-academics/en/academic-calendar",
    "btech_about": f"{BASE_URL}/bachelor-of-technology/en/about-faqs",
    "btech_facilities": f"{BASE_URL}/bachelor-of-technology/en/academic-research-facilities",
    "pg_admission": f"{BASE_URL}/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
    "mtech_admission": f"{BASE_URL}/admission-postgraduate-programs/en/Admission-to-M.Tech.-Degree-Programmes,-AY-2025-26?ep=fw",
    "admission_links": f"{BASE_URL}/main/en/admission-links",
    
    # ── Source 3: Research Pages ──
    "research_highlights": f"{BASE_URL}/main/en/Research-Highlight",
    "research_development": f"{BASE_URL}/office-of-research-development/en/office-of-research-and-development",
    "research_about": f"{BASE_URL}/office-of-research-development/en/About-Us",
    "ai_ds_research": f"{BASE_URL}/school-of-artificial-intelligence-data-science/en/about-research",
    "ai_ds_coes": f"{BASE_URL}/school-of-artificial-intelligence-data-science/en/about-coes",
    "cete_research": f"{BASE_URL}/cete/en/about-research",
    "crdsi": f"{BASE_URL}/crdsi/en/center-for-research-and-development-of-scientific-instruments",
    "sustainability": f"{BASE_URL}/center-for-emerging-technologies-for-sustainable-development/en/about-sustainability-plan",
    "techscape": f"{BASE_URL}/techscape/en/Techscape",
    "institute_repository": f"{BASE_URL}/Institute-Repository/en/Institute-Repository",
    
    # ── Source 4: Faculty & People ──
    "faculty_members": f"{BASE_URL}/main/en/faculty-members",
    "adjunct_faculty": f"{BASE_URL}/main/en/adjunct-faculty-members",
    "visiting_faculty": f"{BASE_URL}/main/en/visiting-faculty-members",
    "scholars_residence": f"{BASE_URL}/main/en/scholars-in-residence",
    "faculty_positions": f"{BASE_URL}/faculty-positions/en/faculty-positions",
    
    # ── Source 5: Institute / About ──
    "about_iitj": f"{BASE_URL}/main/en/iitj",
    "institute": f"{BASE_URL}/Main/en/Institute",
    "director": f"{BASE_URL}/main/en/director",
    "chairman": f"{BASE_URL}/main/en/chairman",
    "about_director": f"{BASE_URL}/office-of-director/en/about-director",
    "campus_life": f"{BASE_URL}/office-of-students/en/Campus-Life",
    "office_students": f"{BASE_URL}/office-of-students/en/office-of-students",
    
    # ── Source 6: News & Announcements ──
    "news": f"{BASE_URL}/main/en/Latest-News",
    "announcements": f"{BASE_URL}/main/en/Announcements",
    "events": f"{BASE_URL}/main/en/Events",
    
    # ── Source 7: Schools ──
    "schools": f"{BASE_URL}/schools/",
    "school_sme": f"{BASE_URL}/schools/en/School-of-Management-&-Entrepreneurship",
    "school_liberal_arts": f"{BASE_URL}/school-of-liberal-arts/en/academic-outreach",
    "school_coe_ip": f"{BASE_URL}/schools/en/about-coe-ip",
    
    # ── Additional sources ──
    "library": f"{BASE_URL}/library/en/About-Discover-Resources",
    "library_srrlh": f"{BASE_URL}/library/en/About-SRRLH",
    "technology_policy": f"{BASE_URL}/center-for-technology-foresight-and-policy/en/about-us",
    "digital_infra": f"{BASE_URL}/digital-infrastructure-automation/en/About-Digital-Infrastructure-Automation",
}


def fetch_page(url):
    """Fetch a webpage and return its HTML content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  ✗ Failed to fetch {url}: {e}")
        return None


def extract_text(html):
    """
    Extract clean text from HTML, removing scripts, styles,
    navigation, footers and other boilerplate.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted tags
    for tag in soup.find_all(["script", "style", "noscript", "iframe",
                              "nav", "footer", "header", "aside",
                              "form", "button", "input", "select"]):
        tag.decompose()
    
    # Remove common boilerplate classes/IDs
    boilerplate_selectors = [
        ".navbar", ".footer", ".sidebar", ".breadcrumb",
        ".menu", ".navigation", ".cookie", ".popup",
        "#navbar", "#footer", "#sidebar", "#menu",
        "[role='navigation']", "[role='banner']",
    ]
    for selector in boilerplate_selectors:
        for el in soup.select(selector):
            el.decompose()
    
    # Get the main content — try common content containers first
    main_content = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", class_=re.compile(r"content|main|body", re.I)) or
        soup.find("div", id=re.compile(r"content|main|body", re.I)) or
        soup.body or
        soup
    )
    
    # Extract text
    text = main_content.get_text(separator="\n", strip=True)
    
    # Clean up
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just special characters or very short
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Skip very short lines that are likely nav artifacts
        if len(line) < 3 and not line.isalpha():
            continue
        # Skip lines that are mostly special characters
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.3 and len(line) > 5:
            continue
        lines.append(line)
    
    return "\n".join(lines)


def scrape_all():
    """Scrape all configured URLs and save raw text files."""
    print("=" * 60)
    print("IIT JODHPUR WEB SCRAPER")
    print("=" * 60)
    print(f"\nTotal URLs to scrape: {len(URLS)}")
    print(f"Output directory: {RAW_DIR}\n")
    
    results = {}
    successful = 0
    failed = 0
    
    for name, url in URLS.items():
        print(f"[{successful + failed + 1}/{len(URLS)}] Scraping: {name}")
        print(f"  URL: {url}")
        
        html = fetch_page(url)
        if html is None:
            failed += 1
            results[name] = {"url": url, "status": "failed", "chars": 0}
            continue
        
        text = extract_text(html)
        
        if len(text.strip()) < 50:
            print(f"  ⚠ Very little text extracted ({len(text)} chars), skipping...")
            failed += 1
            results[name] = {"url": url, "status": "too_short", "chars": len(text)}
            continue
        
        # Save raw text
        filepath = os.path.join(RAW_DIR, f"{name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        successful += 1
        results[name] = {"url": url, "status": "success", "chars": len(text)}
        print(f"  ✓ Saved {len(text)} characters → {filepath}")
        
        time.sleep(REQUEST_DELAY)
    
    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Total:      {successful + failed}")
    
    total_chars = sum(r["chars"] for r in results.values() if r["status"] == "success")
    print(f"Total characters scraped: {total_chars:,}")
    
    # Save results log
    log_path = os.path.join(RAW_DIR, "_scraping_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed log saved to: {log_path}")
    
    return results


if __name__ == "__main__":
    scrape_all()
