# Source of truth for all NSE-listed equities and their relationships.
# This data was manually verified against the NSE equity listings.
# DO NOT modify without updating the source.

NSE_COMPANIES = [
    # Banking
    {"ticker": "ABSA",  "name": "Absa Bank Kenya Plc",           "sector": "Banking"},
    {"ticker": "BKG",   "name": "BK Group Plc",                  "sector": "Banking"},
    {"ticker": "DTK",   "name": "Diamond Trust Bank Kenya",      "sector": "Banking"},
    {"ticker": "EQTY",  "name": "Equity Group Holdings Plc",     "sector": "Banking"},
    {"ticker": "HFCK",  "name": "HF Group Plc",                  "sector": "Banking"},
    {"ticker": "IMH",   "name": "I&M Holdings Plc",              "sector": "Banking"},
    {"ticker": "KCB",   "name": "KCB Group Plc",                 "sector": "Banking"},
    {"ticker": "NCBA",  "name": "NCBA Group Plc",                "sector": "Banking"},
    {"ticker": "SCBK",  "name": "Standard Chartered Bank Kenya", "sector": "Banking"},
    {"ticker": "COOP",  "name": "Co-operative Bank of Kenya",    "sector": "Banking"},
    # Insurance
    {"ticker": "BRIT",  "name": "Britam Holdings Plc",           "sector": "Insurance"},
    {"ticker": "CIC",   "name": "CIC Insurance Group",           "sector": "Insurance"},
    {"ticker": "JUB",   "name": "Jubilee Holdings Ltd",          "sector": "Insurance"},
    {"ticker": "KNRE",  "name": "Kenya Reinsurance Corp",        "sector": "Insurance"},
    {"ticker": "LBTY",  "name": "Liberty Kenya Holdings",        "sector": "Insurance"},
    {"ticker": "SLAM",  "name": "Sanlam Kenya Plc",              "sector": "Insurance"},
    # Telco
    {"ticker": "SCOM",  "name": "Safaricom Plc",                 "sector": "Telco"},
    # Energy
    {"ticker": "KEGN",  "name": "KenGen Plc",                    "sector": "Energy"},
    {"ticker": "KPLC",  "name": "Kenya Power & Lighting Co.",    "sector": "Energy"},
    {"ticker": "TOTL",  "name": "TotalEnergies Marketing Kenya", "sector": "Energy"},
    {"ticker": "UMME",  "name": "Umeme Ltd",                     "sector": "Energy"},
    # Manufacturing
    {"ticker": "BAT",   "name": "British American Tobacco Kenya","sector": "Manufacturing"},
    {"ticker": "EABL",  "name": "East African Breweries Ltd",    "sector": "Manufacturing"},
    {"ticker": "UNGA",  "name": "Unga Group Plc",               "sector": "Manufacturing"},
    {"ticker": "EVRD",  "name": "Eveready East Africa",          "sector": "Manufacturing"},
    {"ticker": "CARB",  "name": "Carbacid Investments",          "sector": "Manufacturing"},
    {"ticker": "BOC",   "name": "BOC Kenya",                     "sector": "Manufacturing"},
    {"ticker": "FTGH",  "name": "Flame Tree Group Holdings",     "sector": "Manufacturing"},
    # Construction
    {"ticker": "BAMB",  "name": "Bamburi Cement Plc",            "sector": "Construction"},
    {"ticker": "PORT",  "name": "East African Portland Cement",  "sector": "Construction"},
    {"ticker": "CRWN",  "name": "Crown Paints Kenya Plc",        "sector": "Construction"},
    {"ticker": "CABL",  "name": "East African Cables",           "sector": "Construction"},
    # Agriculture
    {"ticker": "EGAD",  "name": "Eaagads Ltd",                   "sector": "Agriculture"},
    {"ticker": "KUKZ",  "name": "Kakuzi Plc",                    "sector": "Agriculture"},
    {"ticker": "KAPC",  "name": "Kapchorua Tea",                 "sector": "Agriculture"},
    {"ticker": "LIMT",  "name": "Limuru Tea",                    "sector": "Agriculture"},
    {"ticker": "SASN",  "name": "Sasini Plc",                    "sector": "Agriculture"},
    {"ticker": "WTK",   "name": "Williamson Tea Kenya",          "sector": "Agriculture"},
    # Media & Commercial
    {"ticker": "NMG",   "name": "Nation Media Group",            "sector": "Media"},
    {"ticker": "SGL",   "name": "Standard Group Plc",            "sector": "Media"},
    {"ticker": "SCAN",  "name": "Scangroup Plc",                 "sector": "Media"},
    {"ticker": "KQ",    "name": "Kenya Airways Plc",             "sector": "Transport"},
    {"ticker": "TPSE",  "name": "TPS Eastern Africa (Serena)",   "sector": "Hospitality"},
    # Investment
    {"ticker": "CTUM",  "name": "Centum Investment Company",     "sector": "Investment"},
    {"ticker": "TCL",   "name": "TransCentury Plc",              "sector": "Investment"},
    {"ticker": "OCH",   "name": "Olympia Capital Holdings",      "sector": "Investment"},
    {"ticker": "NSE",   "name": "Nairobi Securities Exchange Plc","sector": "Investment"},
    # Real Estate
    {"ticker": "HAFR",  "name": "Home Afrika Ltd",               "sector": "Real Estate"},
    # Automotive
    {"ticker": "CGEN",  "name": "Car & General Kenya",           "sector": "Automotive"},
]

# Competitor relationships — only assert what is listed here.
# Format: (ticker_1, ticker_2, relationship_type)
# Private companies (no ticker) use their name string as identifier.
COMPETITOR_RELATIONSHIPS = [
    # Telco
    ("SCOM", "Airtel Kenya",    "competitor"),
    ("SCOM", "Telkom Kenya",    "competitor"),
    # Banking
    ("EQTY", "KCB",             "competitor"),
    ("EQTY", "COOP",            "competitor"),
    ("EQTY", "NCBA",            "competitor"),
    ("EQTY", "ABSA",            "competitor"),
    ("KCB",  "NCBA",            "competitor"),
    ("KCB",  "ABSA",            "competitor"),
    ("ABSA", "SCBK",            "competitor"),
    ("IMH",  "KCB",             "competitor"),
    ("IMH",  "EQTY",            "competitor"),
    ("DTK",  "IMH",             "competitor"),
    ("DTK",  "KCB",             "competitor"),
    ("HFCK", "COOP",            "competitor"),
    ("HFCK", "KCB",             "competitor"),
    # Insurance
    ("BRIT", "CIC",             "competitor"),
    ("BRIT", "JUB",             "competitor"),
    ("BRIT", "SLAM",            "competitor"),
    ("CIC",  "JUB",             "competitor"),
    ("LBTY", "SLAM",            "competitor"),
    ("LBTY", "BRIT",            "competitor"),
    # Energy
    ("TOTL", "Vivo Energy",     "competitor"),
    ("TOTL", "Rubis",           "competitor"),
    ("CARB", "BOC",             "competitor"),
    # Construction
    ("BAMB", "PORT",            "competitor"),
    ("BAMB", "Savannah Cement", "competitor"),
    ("PORT", "Savannah Cement", "competitor"),
    ("CRWN", "Basco Paints",    "competitor"),
    ("CABL", "Aberdare Cables", "competitor"),
    # Agriculture
    ("SASN", "KUKZ",            "competitor"),
    ("WTK",  "LIMT",            "competitor"),
    ("LIMT", "KAPC",            "competitor"),
    ("EGAD", "WTK",             "competitor"),
    # Media
    ("NMG",  "SGL",             "competitor"),
    # Aviation
    ("KQ",   "Ethiopian Airlines", "competitor"),
    ("KQ",   "RwandAir",           "competitor"),
    ("TPSE", "Sarova Hotels",      "competitor"),
    # Investment
    ("CTUM", "TCL",             "competitor"),
    ("OCH",  "CTUM",            "competitor"),
]

# Sector lookup helpers
VALID_TICKERS = {c["ticker"] for c in NSE_COMPANIES}
SECTOR_MAP    = {c["ticker"]: c["sector"] for c in NSE_COMPANIES}
NAME_MAP      = {c["ticker"]: c["name"]   for c in NSE_COMPANIES}

def get_sector_peers(ticker: str) -> list[str]:
    """Return all NSE-listed tickers in the same sector as the given ticker."""
    sector = SECTOR_MAP.get(ticker)
    if not sector:
        return []
    return [c["ticker"] for c in NSE_COMPANIES if c["sector"] == sector and c["ticker"] != ticker]

def get_competitors(ticker: str) -> list[str]:
    """Return NSE-listed competitor tickers for a given ticker."""
    peers = []
    for t1, t2, _ in COMPETITOR_RELATIONSHIPS:
        if t1 == ticker and t2 in VALID_TICKERS:
            peers.append(t2)
        elif t2 == ticker and t1 in VALID_TICKERS:
            peers.append(t1)
    return list(set(peers))
