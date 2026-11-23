"""Catálogo canónico de categorías y ejemplares semánticos para el clasificador de noticias."""

CATEGORIES: dict[str, list[str]] = {
    "earnings": [
        "Quarterly earnings report beats Wall Street estimates with record revenue and profit",
        "Bank reports quarterly net profit decline and EPS drop as trading revenues fall",
        "Board of directors approves quarterly dividend increase and share buyback program",
        "Quarterly financial balance sheet, net interest income, and operating margin results",
    ],
    "m&a": [
        "Bank confirms cutting thousands of jobs in massive corporate workforce reduction",
        "Major layoffs announced across consumer banking, wealth management, and mortgage divisions",
        "Bank agrees to acquire asset management unit in multi-billion dollar buyout deal",
        "Bank sells wealth advisory business in strategic divestiture agreement",
        "Investment bank hired as financial advisor for strategic review or underwriting company IPO",
        "Bank agrees to acquire financial services firm or fintech company in all-cash deal with price per share",
        "Board reaches agreement to acquire financial software partner for cash consideration per share",
    ],
    "macro": [
        "Federal Reserve decides to pause interest rate hikes at FOMC policy meeting",
        "Fed signals benchmark rate cuts as inflation CPI cools and Treasury yields decline",
        "Central bank monetary policy, bond yield curve inversions, and macroeconomic forecast",
        "Federal Reserve Chairman comments on economic growth, inflation targets, and banking liquidity",
    ],
    "legal": [
        "Bank reaches multi-billion dollar settlement with SEC and DOJ over mortgage securities",
        "Regulators fine major bank millions for anti-money laundering and compliance failures",
        "Consumer Financial Protection Bureau CFPB investigates banking sales practices and penalties",
        "Banking authorities issue consent order and legal sanctions following probe",
        "Banking subsidiary or foreign unit faces regulatory investigation or probe by department of justice DOJ",
    ],
    "credit": [
        "Bank sets aside billions in loan loss provisions as borrower credit defaults rise",
        "Commercial real estate distress drives sharp surge in non-performing loans and delinquencies",
        "Rating agency downgrades bank credit outlook amid rising credit risk and borrower bankruptcies",
        "Higher credit costs and loan write-offs reduce bank profitability",
    ],
    "governance": [
        "Board of directors appoints new Chief Executive Officer CEO and corporate chairman",
        "Executive management succession announced following annual shareholder proxy meeting",
        "Shareholders vote on executive compensation and board member appointments",
        "Major institutional shareholder sells or acquires significant equity stake in bank",
    ],
    "ruido": [
        "General off-topic news, sports tournaments, stadium sponsorships, entertainment and lifestyle",
        "Daily broad stock market recap, top gainers and losers lists, and market summary podcasts",
        "Technology gadgets, video game launches, 3D printing toys, and retail consumer products",
        "Biotechnology clinical trials, pharmaceutical drugs, and non-banking healthcare companies",
        "Stock market technical chart patterns, daily percentage movers lists, and website cookie notices",
        "Ex-dividend date calendar, dividend payout schedules, and shareholder record dates for other companies",
        "Pre-open stock roundups, premarket stars, and morning market movers summaries",
    ],
}
