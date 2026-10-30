import functools
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus, unquote, urljoin

import requests
from bs4 import BeautifulSoup
from flask import (
    Flask, abort, g, jsonify, redirect, render_template, request, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from predict import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_secret_key():
    """Sessions are signed with this key. It is created once and kept in
    secret.key so people stay signed in when the app restarts."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(BASE_DIR, "secret.key")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key


app = Flask(__name__)
app.secret_key = load_secret_key()
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),  # "Keep me signed in"
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}
MAX_PER_SITE = 8
DB_FILE = os.path.join(BASE_DIR, "prices.db")
RECHECK_HOURS = 20  # tracked products are re-checked about once a day


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created TEXT NOT NULL);
            -- one row per product we've seen on Amazon or Flipkart
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,        -- 'amazon:B0ABC12345' or 'flipkart:ACCG...'
                site TEXT, title TEXT, url TEXT, image TEXT,
                rating REAL, reviews INTEGER, last_seen TEXT);
            -- the lowest price we saw for each product on each day
            CREATE TABLE IF NOT EXISTS product_prices (
                product_id TEXT, day TEXT, price REAL, mrp REAL,
                PRIMARY KEY (product_id, day));
            -- products a user asked us to check every day
            CREATE TABLE IF NOT EXISTS tracked (
                user_id INTEGER, product_id TEXT, query TEXT, created TEXT,
                PRIMARY KEY (user_id, product_id));
            -- when each search was last run, so tracking doesn't repeat it too often
            CREATE TABLE IF NOT EXISTS checks (query TEXT PRIMARY KEY, last_check TEXT);
        """)


@contextmanager
def db():
    conn = sqlite3.connect(DB_FILE, timeout=15)
    try:
        with conn:  # commits on success, rolls back on error
            yield conn
    finally:
        conn.close()


init_db()


# ---------------------------------------------------------- price history

def product_id(site, url):
    """A stable ID for a product, taken from its link, so the same product is
    recognised again on later days even if its title or position changes."""
    link = unquote(url or "")
    if site == "Amazon":
        m = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", link)
        return f"amazon:{m.group(1)}" if m else None
    m = re.search(r"[?&]pid=([A-Z0-9]{8,20})", link)
    if m:
        return f"flipkart:{m.group(1)}"
    m = re.search(r"/p/(itm[0-9a-z]+)", link)
    return f"flipkart:{m.group(1)}" if m else None


def save_products(items, query):
    """Record today's price for every product found, and when this search ran."""
    today = date.today().isoformat()
    now = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        for p in items:
            if not p.get("id"):
                continue
            conn.execute(
                "INSERT INTO products (id, site, title, url, image, rating, reviews, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "title=excluded.title, url=excluded.url, image=COALESCE(excluded.image, image), "
                "rating=COALESCE(excluded.rating, rating), reviews=COALESCE(excluded.reviews, reviews), "
                "last_seen=excluded.last_seen",
                (p["id"], p["site"], p["title"], p["url"], p.get("image"),
                 p.get("rating"), p.get("reviews"), now),
            )
            conn.execute(
                "INSERT INTO product_prices (product_id, day, price, mrp) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(product_id, day) DO UPDATE SET "
                "price=MIN(price, excluded.price), mrp=COALESCE(excluded.mrp, mrp)",
                (p["id"], today, p["price"], p.get("mrp")),
            )
        conn.execute(
            "INSERT INTO checks (query, last_check) VALUES (?, ?) "
            "ON CONFLICT(query) DO UPDATE SET last_check=excluded.last_check",
            (normal_query(query), now),
        )


def normal_query(query):
    return " ".join(query.lower().split())


def price_history(conn, pid):
    rows = conn.execute(
        "SELECT day, price, mrp FROM product_prices WHERE product_id = ? ORDER BY day", (pid,)
    ).fetchall()
    return rows


def prediction_for(conn, pid):
    rows = price_history(conn, pid)
    if not rows:
        return None, rows
    mrp = rows[-1][2]
    return predict([(d, p) for d, p, _ in rows], mrp=mrp), rows


def compact(pred):
    """The few fields a results card needs."""
    if not pred:
        return None
    fields = {k: pred[k] for k in ("verdict", "label", "confidence", "forecast", "trend_pct")}
    fields["days"] = pred["stats"]["days"]
    return fields


# ---------------------------------------------------------------- accounts

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with db() as conn:
        row = conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if not row:  # account no longer exists
        session.clear()
        return None
    return {"id": row[0], "name": row[1], "email": row[2]}


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        g.user = current_user()
        if g.user is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Your session has ended. Sign in again to search."}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def csrf_token():
    """A random token per session, put in every form, so other websites
    cannot submit forms to this app on someone's behalf."""
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


def check_csrf():
    sent = request.form.get("csrf", "") or request.headers.get("X-CSRF-Token", "")
    if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
        abort(400, "This form has expired. Go back, reload the page and try again.")


def safe_next(target):
    """Only redirect to pages on this site after signing in."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("index")


def start_session(user_id, remember):
    session.clear()  # new session on sign-in
    session["user_id"] = user_id
    session.permanent = bool(remember)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("index"))
    errors, form = {}, {"name": "", "email": ""}
    next_url = request.values.get("next", "")

    if request.method == "POST":
        check_csrf()
        name = " ".join(request.form.get("name", "").split())
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        form = {"name": name, "email": email}

        if not name:
            errors["name"] = "Enter your name."
        elif len(name) > 60:
            errors["name"] = "Use 60 characters or fewer."
        if not EMAIL_RE.match(email) or len(email) > 254:
            errors["email"] = "Enter an email address like name@example.com."
        if len(password) < MIN_PASSWORD:
            errors["password"] = f"Use at least {MIN_PASSWORD} characters."
        elif password != confirm:
            errors["confirm"] = "The passwords don't match."

        if not errors:
            try:
                with db() as conn:
                    cur = conn.execute(
                        "INSERT INTO users (name, email, password_hash, created) "
                        "VALUES (?, ?, ?, ?)",
                        (name, email, generate_password_hash(password),
                         datetime.now().isoformat(timespec="seconds")),
                    )
                    user_id = cur.lastrowid
            except sqlite3.IntegrityError:
                errors["email"] = "An account with this email already exists. Sign in instead."
            else:
                start_session(user_id, remember=True)
                return redirect(safe_next(next_url))

    return render_template(
        "auth.html", mode="signup", errors=errors, form=form, next=next_url,
    ), (400 if errors else 200)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("index"))
    errors, form = {}, {"email": ""}
    next_url = request.values.get("next", "")

    if request.method == "POST":
        check_csrf()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        form = {"email": email}
        with db() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row and check_password_hash(row[1], password):
            start_session(row[0], remember=request.form.get("remember") == "on")
            return redirect(safe_next(next_url))
        # Same message either way, so nobody can find out which emails have accounts.
        errors["form"] = "That email and password don't match an account."

    return render_template(
        "auth.html", mode="login", errors=errors, form=form, next=next_url,
    ), (400 if errors else 200)


@app.route("/logout", methods=["POST"])
def logout():
    check_csrf()
    session.clear()
    return redirect(url_for("login", out=1))


def to_number(text):
    """'₹1,299.00' -> 1299.0"""
    digits = re.sub(r"[^\d.]", "", text or "")
    try:
        return float(digits)
    except ValueError:
        return None


def to_rating(text):
    """'4.2 out of 5 stars' -> 4.2, '4.3' -> 4.3. Anything outside 0-5 is ignored."""
    m = re.search(r"(\d(?:\.\d)?)", text or "")
    if not m:
        return None
    value = float(m.group(1))
    return value if 0 < value <= 5 else None


def to_count(text):
    """'12,345' / '(1,23,456)' / '1.2K' / '2.5L' / '12,345 ratings' -> int."""
    m = re.search(r"([\d,]*\.?\d+)\s*([KkLlMm])?", text or "")
    if not m:
        return None
    number = float(m.group(1).replace(",", ""))
    scale = {"k": 1_000, "l": 100_000, "m": 1_000_000}.get((m.group(2) or "").lower(), 1)
    return int(number * scale)


def with_mrp(item, mrp):
    """Attach MRP and discount only when the MRP is really higher than the price."""
    price = item["price"]
    if price and mrp and mrp > price:
        item["mrp"] = mrp
        item["discount"] = round((mrp - price) / mrp * 100)
    else:
        item["mrp"] = None
        item["discount"] = None
    return item


def parse_amazon(html, base="https://www.amazon.in"):
    soup = BeautifulSoup(html, "lxml")
    items = []
    for card in soup.select('div[data-component-type="s-search-result"]'):
        title = card.select_one("h2 span")
        price = card.select_one("span.a-price:not(.a-text-price) span.a-offscreen")
        mrp = card.select_one("span.a-price.a-text-price span.a-offscreen")
        link = card.select_one("a.a-link-normal[href*='/dp/'], h2 a")
        img = card.select_one("img.s-image")
        if not (title and price):
            continue

        # Rating: '4.2 out of 5 stars', either as hidden text or an aria-label.
        rating = None
        for star in card.select("span.a-icon-alt"):  # skip other icons, e.g. Prime
            if "out of 5" in star.get_text():
                rating = to_rating(star.get_text())
                break
        if rating is None:
            labelled = card.select_one("[aria-label*='out of 5 stars']")
            if labelled:
                rating = to_rating(labelled["aria-label"])

        # Review count: '12,345 ratings' as an aria-label, or the underlined number.
        reviews = None
        count = card.select_one("[aria-label$='ratings'], [aria-label$='rating']")
        if count:
            reviews = to_count(count["aria-label"])
        if reviews is None:
            count = card.select_one(
                "a[href*='customerReviews'] span.s-underline-text, "
                "span.a-size-base.s-underline-text, a[href*='customerReviews'] span"
            )
            if count:
                reviews = to_count(count.get_text())

        items.append(with_mrp({
            "site": "Amazon",
            "title": title.get_text(strip=True),
            "price": to_number(price.get_text()),
            "url": urljoin(base, link["href"]) if link else base,
            "image": img["src"] if img else None,
            "rating": rating,
            "reviews": reviews if rating else None,
        }, to_number(mrp.get_text()) if mrp else None))
        if len(items) >= MAX_PER_SITE:
            break
    return items


RUPEE_TEXT = re.compile(r"₹\s*[\d,]+(?:\.\d+)?")
RATING_TEXT = re.compile(r"[1-5](?:\.\d)?")


def flipkart_prices(card):
    """All rupee amounts in a card, in page order. The first is the selling price,
    a later, higher one is the struck-out MRP. Flipkart often splits '₹' and the
    digits with an HTML comment, so this reads whole leaf elements, not raw strings."""
    values = []
    for tag in card.find_all(["div", "span"]):
        if tag.find(True):  # only leaf elements
            continue
        text = tag.get_text("", strip=True)
        if RUPEE_TEXT.fullmatch(text):
            values.append(to_number(text))
    if not values:  # fallback to the old string-based lookup
        node = card.find(string=re.compile(r"^\s*₹\s*[\d,]+"))
        if node:
            values.append(to_number(node))
    return values


def flipkart_rating(card):
    node = card.select_one("[id^='productRating']")
    if node:
        rating = to_rating(node.get_text(strip=True))
        if rating:
            return rating
    # Fallback: a small element whose own text is just '4.3' next to a star image.
    for tag in card.find_all(["div", "span"]):
        own = "".join(tag.find_all(string=True, recursive=False)).strip()
        if RATING_TEXT.fullmatch(own) and tag.find("img"):
            return float(own)
    return None


def flipkart_reviews(card):
    text = card.get_text(" ", strip=True)
    m = re.search(r"([\d,]+)\s*Ratings", text)
    if m:
        return to_count(m.group(1))
    node = card.find(string=re.compile(r"^\s*\(\s*[\d,]+\s*\)\s*$"))
    return to_count(node) if node else None


def parse_flipkart(html, base="https://www.flipkart.com"):
    soup = BeautifulSoup(html, "lxml")
    items = []
    # Flipkart class names change often, so this looks at structure instead:
    # each result has a product link and some text that starts with the rupee sign.
    for card in soup.select("div[data-id]"):
        link = card.select_one("a[href*='/p/']")
        prices = flipkart_prices(card)
        if not (link and prices):
            continue
        titled = card.select_one("a[title]")
        img_alt = card.select_one("img[alt]:not([alt=''])")
        title = (
            (titled["title"] if titled else None)
            or (img_alt["alt"] if img_alt else None)
            or link.get_text(" ", strip=True)
        )
        if not title:
            continue
        img = card.select_one("img[src^='http']")
        rating = flipkart_rating(card)
        price = prices[0]
        mrp = max(prices[1:], default=None)
        items.append(with_mrp({
            "site": "Flipkart",
            "title": title,
            "price": price,
            "url": urljoin(base, link["href"]),
            "image": img["src"] if img else None,
            "rating": rating,
            "reviews": flipkart_reviews(card) if rating else None,
        }, mrp))
        if len(items) >= MAX_PER_SITE:
            break
    return items


def fetch_html(url):
    r = requests.get(url, headers=HEADERS, timeout=12)
    r.raise_for_status()
    return r.text


def scrape_amazon(query):
    return parse_amazon(fetch_html(f"https://www.amazon.in/s?k={quote_plus(query)}"))


def scrape_flipkart(query):
    return parse_flipkart(fetch_html(f"https://www.flipkart.com/search?q={quote_plus(query)}"))


SCRAPERS = {"Amazon": scrape_amazon, "Flipkart": scrape_flipkart}


def run_scraper(name, query):
    try:
        return name, SCRAPERS[name](query), None
    except Exception as e:
        return name, [], f"{name} could not be reached ({type(e).__name__})."


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=g.user)


def run_search(query):
    with ThreadPoolExecutor(max_workers=len(SCRAPERS)) as pool:
        results = list(pool.map(lambda n: run_scraper(n, query), SCRAPERS))

    products, warnings = [], []
    for name, items, err in results:
        products.extend(i for i in items if i["price"])
        if err:
            warnings.append(err)
        elif not items:
            warnings.append(f"{name} returned no results. It may have blocked the request.")

    for p in products:
        p["id"] = product_id(p["site"], p["url"])
    if products:
        save_products(products, query)
    products.sort(key=lambda p: p["price"])
    return products, warnings


@app.route("/api/search")
@login_required
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Type a product name to search."}), 400

    products, warnings = run_search(query)

    with db() as conn:
        tracked = {r[0] for r in conn.execute(
            "SELECT product_id FROM tracked WHERE user_id = ?", (g.user["id"],))}
        for p in products:
            pred = prediction_for(conn, p["id"])[0] if p["id"] else None
            p["prediction"] = compact(pred)
            p["tracked"] = p["id"] in tracked

    return jsonify({"query": query, "products": products, "warnings": warnings})


def product_payload(conn, pid, user_id):
    row = conn.execute(
        "SELECT id, site, title, url, image, rating, reviews FROM products WHERE id = ?", (pid,)
    ).fetchone()
    if not row:
        return None
    pred, rows = prediction_for(conn, pid)
    tracked = conn.execute(
        "SELECT 1 FROM tracked WHERE user_id = ? AND product_id = ?", (user_id, pid)
    ).fetchone() is not None
    return {
        "product": dict(zip(("id", "site", "title", "url", "image", "rating", "reviews"), row)),
        "history": [{"day": d, "price": p, "mrp": m} for d, p, m in rows],
        "prediction": pred,
        "tracked": tracked,
    }


@app.route("/api/product")
@login_required
def product_details():
    with db() as conn:
        data = product_payload(conn, request.args.get("id", ""), g.user["id"])
    if not data:
        return jsonify({"error": "We don't have this product saved. Search for it again."}), 404
    return jsonify(data)


@app.route("/api/track", methods=["POST", "DELETE"])
@login_required
def track():
    check_csrf()
    body = request.get_json(silent=True) or {}
    pid = body.get("id", "")
    with db() as conn:
        if request.method == "DELETE":
            conn.execute("DELETE FROM tracked WHERE user_id = ? AND product_id = ?",
                         (g.user["id"], pid))
            return jsonify({"tracked": False})
        if not conn.execute("SELECT 1 FROM products WHERE id = ?", (pid,)).fetchone():
            return jsonify({"error": "We don't have this product saved. Search for it again."}), 404
        conn.execute(
            "INSERT OR IGNORE INTO tracked (user_id, product_id, query, created) VALUES (?, ?, ?, ?)",
            (g.user["id"], pid, normal_query(body.get("query", "")) or None,
             datetime.now().isoformat(timespec="seconds")),
        )
    return jsonify({"tracked": True})


@app.route("/api/tracked")
@login_required
def tracked_list():
    with db() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT product_id FROM tracked WHERE user_id = ? ORDER BY created DESC", (g.user["id"],))]
        items = []
        for pid in ids:
            data = product_payload(conn, pid, g.user["id"])
            if data and data["prediction"]:
                items.append({**data["product"], "price": data["prediction"]["stats"]["current"],
                              "prediction": compact(data["prediction"])})
    return jsonify({"items": items})


# ------------------------------------------------- daily check of tracked items

def recheck_tracked():
    """Run each tracked product's search again if it hasn't run in the last
    RECHECK_HOURS, so price history builds up even when nobody searches."""
    cutoff = (datetime.now() - timedelta(hours=RECHECK_HOURS)).isoformat(timespec="seconds")
    with db() as conn:
        due = [r[0] for r in conn.execute(
            "SELECT DISTINCT t.query FROM tracked t LEFT JOIN checks c ON c.query = t.query "
            "WHERE t.query IS NOT NULL AND (c.last_check IS NULL OR c.last_check < ?)", (cutoff,))]
    for query in due:
        products, _ = run_search(query)
        app.logger.info("Re-checked %r: %d prices saved", query, len(products))
        time.sleep(5)  # be gentle with the stores


def tracker_loop():
    while True:
        try:
            recheck_tracked()
        except Exception:
            app.logger.exception("Daily price check failed")
        time.sleep(60 * 60)  # look again every hour


if __name__ == "__main__":
    threading.Thread(target=tracker_loop, daemon=True).start()
    # Debug mode lets anyone who sees an error page run code on this computer,
    # so it stays off unless you ask for it while developing: set FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
