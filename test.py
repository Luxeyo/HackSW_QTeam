import os  #env vars
import time  #timestamps for caching
import aiohttp  #async HTTP calls
import discord  #discord API
from discord.ext import commands  #bot framework

from mkt_cap import get_market_caps_usd, format_usd  #market cap helpers split out


BOT_TOKEN = os.getenv("BOT_TOKEN")  #discord bot token
GUILD_ID_RAW = os.getenv("GUILD_ID")  #target guild
COINGECKO_KEY = os.getenv("COINGECKO_KEY")  #coingecko api key

if not BOT_TOKEN or not GUILD_ID_RAW:  #fail fast if config missing
    raise ValueError("Missing BOT_TOKEN or GUILD_ID env vars.")

GUILD_ID = int(GUILD_ID_RAW)  #cast guild id
COINGECKO_BASE = "https://api.coingecko.com/api/v3"  #base url

intents = discord.Intents.default()  #default intents are enough
bot = commands.Bot(command_prefix="!", intents=intents)  #bot instance


_coin_list_cache = {  #cache coin list to avoid spamming CG
    "ts": 0.0,
    "coins": [],
    "symbol_to_ids": {},
    "name_to_ids": {},
}
_resolve_cache = {}  #cache resolved user inputs


def cg_headers() -> dict:  #build headers for CG requests
    headers = {}
    if COINGECKO_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_KEY
    return headers


async def fetch_coin_list_if_needed(ttl_seconds: int = 6 * 60 * 60) -> None:  #refresh coin list if stale
    now = time.time()
    if _coin_list_cache["coins"] and (now - _coin_list_cache["ts"] < ttl_seconds):
        return

    url = f"{COINGECKO_BASE}/coins/list"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=cg_headers()) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"/coins/list failed {resp.status}: {text}")
            coins = await resp.json()

    symbol_to_ids = {}
    name_to_ids = {}

    for c in coins:
        cid = c.get("id", "")
        sym = (c.get("symbol") or "").lower().strip()
        name = (c.get("name") or "").lower().strip()

        if cid and sym:
            symbol_to_ids.setdefault(sym, []).append(cid)
        if cid and name:
            name_to_ids.setdefault(name, []).append(cid)

    _coin_list_cache["ts"] = now
    _coin_list_cache["coins"] = coins
    _coin_list_cache["symbol_to_ids"] = symbol_to_ids
    _coin_list_cache["name_to_ids"] = name_to_ids


async def cg_search(query: str) -> list[dict]:  #fallback fuzzy search
    url = f"{COINGECKO_BASE}/search"
    params = {"query": query}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=cg_headers()) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("coins", []) or []


async def pick_highest_market_cap_id(candidate_ids: list[str]) -> str | None:  #disambiguate symbols
    if not candidate_ids:
        return None

    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(candidate_ids[:250]),
        "order": "market_cap_desc",
        "per_page": min(len(candidate_ids), 250),
        "page": 1,
        "sparkline": "false",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=cg_headers()) as resp:
            if resp.status != 200:
                return candidate_ids[0]
            markets = await resp.json()

    if not markets:
        return candidate_ids[0]

    markets.sort(key=lambda x: (x.get("market_cap") or 0), reverse=True)
    return markets[0].get("id") or candidate_ids[0]


async def resolve_to_coingecko_id(user_input: str) -> str | None:  #resolve ticker/name to CG id
    raw = (user_input or "").strip()
    if not raw:
        return None

    key = raw.lower()
    if key in _resolve_cache:
        return _resolve_cache[key]

    await fetch_coin_list_if_needed()

    sym_ids = _coin_list_cache["symbol_to_ids"].get(key, [])
    if sym_ids:
        chosen = await pick_highest_market_cap_id(sym_ids)
        if chosen:
            _resolve_cache[key] = chosen
            return chosen

    name_ids = _coin_list_cache["name_to_ids"].get(key, [])
    if name_ids:
        chosen = await pick_highest_market_cap_id(name_ids)
        if chosen:
            _resolve_cache[key] = chosen
            return chosen

    search_results = await cg_search(raw)
    if search_results:
        chosen = search_results[0].get("id")
        if chosen:
            _resolve_cache[key] = chosen
            return chosen

    _resolve_cache[key] = key  #last resort
    return key


async def get_usd_prices(coingecko_ids: list[str]) -> dict:  #fetch live prices
    url = f"{COINGECKO_BASE}/simple/price"
    params = {
        "ids": ",".join(coingecko_ids),
        "vs_currencies": "usd",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=cg_headers()) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"/simple/price failed {resp.status}: {text}")
            return await resp.json()


async def calculate_risk_score_placeholder(from_id: str, to_id: str, amount: float) -> str:  #stub
    return "Risk score placeholder: not calculated yet."


class RiskView(discord.ui.View):  #button container
    def __init__(self, from_id: str, to_id: str, amount: float):
        super().__init__(timeout=300)
        self.from_id = from_id
        self.to_id = to_id
        self.amount = amount

    @discord.ui.button(label="Calculate risk score", style=discord.ButtonStyle.primary)
    async def calc_risk(self, interaction: discord.Interaction, button: discord.ui.Button):
        result = await calculate_risk_score_placeholder(self.from_id, self.to_id, self.amount)
        await interaction.response.send_message(result, ephemeral=True)


@bot.tree.command(  #slash command entry
    name="trade",
    description="Submit a trade request",
    guild=discord.Object(id=GUILD_ID),
)
async def trade(interaction: discord.Interaction, from_token: str, to_token: str, amount: float):
    await interaction.response.send_message("Trade received. Fetching live data…", ephemeral=True)  #ack fast

    cg_from = await resolve_to_coingecko_id(from_token)  #resolve from token
    cg_to = await resolve_to_coingecko_id(to_token)  #resolve to token

    if not cg_from or not cg_to:
        await interaction.followup.send("Could not resolve token(s).", ephemeral=True)
        return

    try:
        prices = await get_usd_prices([cg_from, cg_to])  #price fetch

        if cg_from not in prices or cg_to not in prices:
            await interaction.followup.send("CoinGecko returned no price for one token.", ephemeral=True)
            return

        from_price = float(prices[cg_from]["usd"])
        to_price = float(prices[cg_to]["usd"])
        converted_amount = (amount * from_price) / to_price  #conversion

        caps = await get_market_caps_usd([cg_from, cg_to], COINGECKO_KEY)  #market caps
        from_cap = caps.get(cg_from)
        to_cap = caps.get(cg_to)

    except Exception as e:
        await interaction.followup.send(f"Data fetch failed: {e}", ephemeral=True)
        return

    display_from = from_token.strip().upper()  #display formatting
    display_to = to_token.strip().upper()

    embed = discord.Embed(title="Trade preview", description="Live price quote")  #main embed
    embed.add_field(name="User", value=str(interaction.user), inline=False)
    embed.add_field(name="From", value=f"🪙 `{display_from}`", inline=False)
    embed.add_field(name="To", value=f"🪙 `{display_to}`", inline=False)
    embed.add_field(name="Amount", value=f"`{amount} {display_from}`", inline=False)

    embed.add_field(
        name="Prices (USD)",
        value=f"{display_from}: `${from_price:.6g}`\n{display_to}: `${to_price:.6g}`",
        inline=False,
    )

    embed.add_field(
        name="Market cap (USD)",
        value=f"{display_from}: `{format_usd(from_cap)}`\n{display_to}: `{format_usd(to_cap)}`",
        inline=False,
    )

    embed.add_field(
        name="Quote",
        value=f"`{amount} {display_from} ≈ {converted_amount:.6g} {display_to}`",
        inline=False,
    )

    embed.add_field(name="Risk score", value="`not calculated`", inline=False)

    view = RiskView(from_id=cg_from, to_id=cg_to, amount=amount)  #attach button
    await interaction.channel.send(embed=embed, view=view)

    await interaction.followup.send("Posted trade preview.", ephemeral=True)  #final confirmation


@bot.event
async def on_ready():
    await bot.tree.sync(guild=discord.Object(id=GUILD_ID))  #sync slash commands
    print(f"Logged in as {bot.user} and synced commands for guild {GUILD_ID}")


bot.run(BOT_TOKEN)  #start bot

