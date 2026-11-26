"""
ABD MOMENTUM - otonom tarama ve al-sat botu (SANAL hesap, öğrenme amaçlı)

GitHub Actions'ta hafta içi her 5 dakikada bir çalışır:
  1) Piyasayı ve risk kurallarını kontrol eder
  2) Hisseleri tarar (radar), sinyal varsa ve otomatik işlem açıksa alım yapar
  3) Kâr-al / zarar-kes satışlarını ve gün sonu özetini Telegram'a bildirir
  4) Mini uygulama için site/panel.json ve site/analiz.json dosyalarını yazar
  5) Günde bir kez, kapanıştan sonra sinyallerin 1 yıllık isabetini test eder
"""
import os
import json
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest, TakeProfitRequest,
                                     StopLossRequest, GetOrdersRequest, GetCalendarRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# ============================ AYARLAR ============================
HISSELER = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD",
            "NFLX", "PLTR", "INTC", "COIN", "SOFI", "MARA", "RIVN", "NIO",
            "UBER", "SHOP", "SNAP", "F"]
OTOMATIK_ISLEM = os.environ.get("OTOMATIK", "acik").lower() != "kapali"   # GitHub değişkeni
HACIM_ESIGI = 3.0          # hacim, önceki günlerin aynı saatindeki ortalamanın kaç katı
FIYAT_ESIGI = 0.5          # son 15 dk'da en az % kaç yükseliş
MAKS_POZISYON = 2          # küçük hesap: aynı anda en fazla 2 hisse
POZISYON_ORANI = 0.50      # her pozisyon için bakiyenin en fazla %50'si (sadece TAM adet alınır)
KAR_AL = 1.5
ZARAR_KES = 1.0
GUNLUK_ZARAR_LIMITI = 3.0
ILK_ISLEM_SAATI = "09:45"
YENI_ISLEM_BITIS_DK = 45
HEPSINI_KAPAT_DK = 20
KONTROL_MUM = 6            # sinyal doğruluğu: 6 mum (30 dk) sonra kontrol
TEST_GUN = 365             # doğruluk testi kaç günlük veriyle yapılır
# =================================================================

NY = "America/New_York"
DURUM_DOSYASI = "durum.json"
SON_TARAMA = None          # son taramanın tablosu
SON_BARS = None            # son taramanın mum verisi (analiz ekranı için)
KAYIT = []


# ---------------------------------------------------------------- yardımcılar
def kaydet(*parcalar):
    satir = " ".join(str(p) for p in parcalar)
    (lambda *a,**k:None)(satir)
    KAYIT.append(satir)


def panel_adresi():
    depo = os.environ.get("GITHUB_REPOSITORY")
    if not depo:
        return None
    sahip, ad = depo.split("/")
    return f"https://{sahip.lower()}.github.io/{ad}/"


def bildir(mesaj):
    token, sohbet = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not sohbet:
        return
    veri = {"chat_id": sohbet, "text": mesaj}
    panel = panel_adresi()
    if panel:
        veri["reply_markup"] = {"inline_keyboard": [[{"text": "📊 Paneli aç", "web_app": {"url": panel}}]]}
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=veri, timeout=10)
    except Exception as e:
        (lambda *a,**k:None)(f"Telegram hatası: {e}")


def durum_oku():
    try:
        with open(DURUM_DOSYASI) as f:
            return json.load(f)
    except Exception:
        return {}


def durum_yaz(durum):
    with open(DURUM_DOSYASI, "w") as f:
        json.dump(durum, f)


def istemciler():
    key, secret = os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def ema(seri, n):
    return seri.ewm(span=n, adjust=False).mean()


def normal_seans(d):
    """Alpaca mumlarını New York saatine çevirip sadece 09:30-16:00 arasını bırakır."""
    d = d.copy()
    d.index = pd.to_datetime(d.index).tz_convert(NY)
    d["saat"] = d.index.strftime("%H:%M")
    d = d[(d["saat"] >= "09:30") & (d["saat"] < "16:00")]
    d["tarih"] = d.index.date
    return d


# ---------------------------------------------------------------- tarama (radar)
def gostergeler(bars, simdi):
    """Her hissenin son TAMAMLANMIŞ mumunu değerlendirir; radar tablosunu döndürür."""
    satirlar, analiz = [], {}
    for h in bars.index.get_level_values(0).unique():
        d = bars.xs(h, level=0)
        d = d[pd.to_datetime(d.index).tz_convert(NY) + pd.Timedelta(minutes=5) <= simdi]
        d = normal_seans(d)
        if len(d) < 30:
            continue
        d["ort_hacim"] = d.groupby("saat")["volume"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        d["HacimX"] = d["volume"] / d["ort_hacim"]
        d["degisim"] = (d["close"] / d.groupby("tarih")["close"].shift(3) - 1) * 100
        d["e9"], d["e21"], d["e50"] = ema(d["close"], 9), ema(d["close"], 21), ema(d["close"], 50)
        bugun = d[d["tarih"] == d["tarih"].iloc[-1]]
        tp = (bugun["high"] + bugun["low"] + bugun["close"]) / 3
        vwap = (tp * bugun["volume"]).cumsum().iloc[-1] / bugun["volume"].cumsum().iloc[-1]
        dunku_kapanis = d[d["tarih"] < d["tarih"].iloc[-1]]["close"]
        son = d.iloc[-1]

        hx = float(son["HacimX"]) if pd.notna(son["HacimX"]) else 0.0
        dg = float(son["degisim"]) if pd.notna(son["degisim"]) else 0.0
        kosul = {
            "hacim": hx >= HACIM_ESIGI,
            "yukselis": dg >= FIYAT_ESIGI,
            "ema": son["e9"] > son["e21"],
            "vwap": son["close"] > vwap,
            "kirilim": len(bugun) > 1 and son["close"] > bugun["high"].iloc[:-1].max(),
        }
        puan = int(min(hx / HACIM_ESIGI, 1.5) / 1.5 * 35 + min(max(dg, 0) / FIYAT_ESIGI, 2) / 2 * 25
                   + 15 * kosul["ema"] + 15 * kosul["vwap"] + 10 * kosul["kirilim"])
        gerekce = [f"Hacim {hx:.1f}x".replace(".", ",")]
        if kosul["vwap"]: gerekce.append("VWAP üstü")
        if kosul["ema"]: gerekce.append("EMA 9 > 21")
        if kosul["kirilim"]: gerekce.append("Gün içi zirve kırılımı")
        etiket = "AL" if kosul["hacim"] and kosul["yukselis"] else ("İZLE" if puan >= 50 else "")

        satirlar.append({
            "Hisse": h, "Zaman": d.index[-1].strftime("%m-%d %H:%M"),
            "Fiyat": round(float(son["close"]), 2),
            "HacimX": round(hx, 1), "15dk%": round(dg, 2),
            "Gun%": round((son["close"] / dunku_kapanis.iloc[-1] - 1) * 100, 2) if len(dunku_kapanis) else 0.0,
            "Puan": puan, "Etiket": etiket, "Gerekce": " · ".join(gerekce),
        })
        son3 = d[d["tarih"].isin(sorted(d["tarih"].unique())[-3:])]
        analiz[h] = {
            "t": son3.index.strftime("%m-%d %H:%M").tolist(),
            "o": son3["open"].round(2).tolist(), "h": son3["high"].round(2).tolist(),
            "l": son3["low"].round(2).tolist(), "c": son3["close"].round(2).tolist(),
            "v": son3["volume"].astype(int).tolist(),
        }
    return pd.DataFrame(satirlar), analiz


def tara(veri, simdi):
    global SON_TARAMA, SON_BARS
    istek = StockBarsRequest(symbol_or_symbols=HISSELER, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                             start=simdi - timedelta(days=12), end=simdi, feed=DataFeed.IEX)
    SON_TARAMA, SON_BARS = gostergeler(veri.get_stock_bars(istek).df, simdi)
    return SON_TARAMA


def sinyal_bul(tablo):
    return tablo[tablo["Etiket"] == "AL"].sort_values("Puan", ascending=False)


# ---------------------------------------------------------------- doğruluk testi
def ema_sinyalleri(c, e9, e21, e50):
    """Analiz ekranındaki kural (JS ile birebir aynı): EMA9 EMA21'i yukarı keserse ve
    fiyat EMA50 üstündeyse AL; aşağı keserse SAT."""
    yukari = (e9.shift(1) <= e21.shift(1)) & (e9 > e21) & (c > e50)
    asagi = (e9.shift(1) >= e21.shift(1)) & (e9 < e21)
    return yukari, asagi


def dogruluk_testi(veri, simdi):
    """Son TEST_GUN günün tam piyasa (SIP) verisiyle iki kuralın 30 dk isabetini ölçer."""
    istek = StockBarsRequest(symbol_or_symbols=HISSELER, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                             start=simdi - timedelta(days=TEST_GUN), end=simdi - timedelta(minutes=20),
                             feed=DataFeed.SIP)
    bars = veri.get_stock_bars(istek).df
    ema_top, radar_top, hisse_sonuc = [], [], {}
    for h in bars.index.get_level_values(0).unique():
        d = normal_seans(bars.xs(h, level=0))
        if len(d) < 500:
            continue
        c = d["close"]
        ileri = d.groupby("tarih")["close"].shift(-KONTROL_MUM)
        getiri = (ileri / c - 1) * 100
        al, sat = ema_sinyalleri(c, ema(c, 9), ema(c, 21), ema(c, 50))
        # kz: sinyal yönünde kazanç (AL'da yükseliş, SAT'ta düşüş pozitif sayılır)
        sonuc = pd.concat([getiri[al], -getiri[sat]]).dropna().rename("kz").to_frame()
        sonuc["dogru"] = sonuc["kz"] > 0
        ema_top.append(sonuc)
        hisse_sonuc[h] = {"adet": int(len(sonuc)), "isabet": round(100 * sonuc["dogru"].mean(), 1) if len(sonuc) else None}
        # radar kuralı
        d["ort_hacim"] = d.groupby("saat")["volume"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        dg = (c / d.groupby("tarih")["close"].shift(3) - 1) * 100
        rs = (d["volume"] / d["ort_hacim"] >= HACIM_ESIGI) & (dg >= FIYAT_ESIGI) & (d["saat"] >= "09:45")
        radar_top.append(getiri[rs].dropna())

    e = pd.concat(ema_top) if ema_top else pd.DataFrame(columns=["kz", "dogru"])
    r = pd.concat(radar_top) if radar_top else pd.Series(dtype=float)
    return {
        "tarih": simdi.strftime("%Y-%m-%d"), "gun": TEST_GUN,
        "ema": {"adet": int(len(e)), "isabet": round(100 * e["dogru"].mean(), 1) if len(e) else None,
                "ort": round(float(e["kz"].mean()), 3) if len(e) else None},
        "radar": {"adet": int(len(r)), "isabet": round(100 * (r > 0).mean(), 1) if len(r) else None,
                  "ort": round(float(r.mean()), 3) if len(r) else None},
        "hisse": hisse_sonuc,
    }


# ---------------------------------------------------------------- emir takibi
def dolan_emirler(trading, baslangic):
    emirler = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=baslangic, limit=500))
    dolan = [o for o in emirler if o.filled_at is not None and float(o.filled_qty or 0) > 0]
    return sorted(dolan, key=lambda o: o.filled_at)


def zirve(veri, sembol, bas, son, feed=DataFeed.IEX):
    """bas-son arasında hissenin gördüğü en yüksek fiyat (1 dakikalık mumlar)."""
    bas, son = pd.Timestamp(bas), pd.Timestamp(son)
    if son <= bas:
        return None
    df = veri.get_stock_bars(StockBarsRequest(symbol_or_symbols=sembol, timeframe=TimeFrame.Minute,
                                              start=bas.to_pydatetime(), end=son.to_pydatetime(), feed=feed)).df
    if df is None or df.empty:
        return None
    return float(df["high"].max())


def islem_ciftleri(emirler):
    """Günün emirlerinden hisse başına giriş/çıkış özeti."""
    c = {}
    for o in emirler:
        d = c.setdefault(o.symbol, {"al_adet": 0.0, "al_tutar": 0.0, "sat_adet": 0.0, "sat_tutar": 0.0, "al_zaman": None})
        q, p = float(o.filled_qty), float(o.filled_avg_price)
        if o.side == OrderSide.BUY:
            d["al_adet"] += q; d["al_tutar"] += q * p
            d["al_zaman"] = d["al_zaman"] or o.filled_at
        else:
            d["sat_adet"] += q; d["sat_tutar"] += q * p
    for d in c.values():
        d["giris"] = d["al_tutar"] / d["al_adet"] if d["al_adet"] else None
        d["cikis"] = d["sat_tutar"] / d["sat_adet"] if d["sat_adet"] else None
    return c


def yuzde(a, b):
    return f"%{(a / b - 1) * 100:+.2f}".replace(".", ",")


def satislari_bildir(trading, veri, durum, simdi):
    son = durum.get("son_kontrol")
    baslangic = pd.Timestamp(son).to_pydatetime() if son else simdi - timedelta(minutes=15)
    alislar = {}
    for o in dolan_emirler(trading, pd.Timestamp(simdi).normalize().to_pydatetime()):
        if o.side == OrderSide.BUY:
            alislar[o.symbol] = (float(o.filled_avg_price), o.filled_at)
            continue
        if o.filled_at <= baslangic:
            continue
        fiyat = float(o.filled_avg_price)
        giris, al_zaman = alislar.get(o.symbol, (None, None))
        kz = f" ({yuzde(fiyat, giris)})" if giris else ""
        mesaj = f"{'💰' if giris and fiyat > giris else '🔻'} SATILDI: {o.symbol} {o.filled_qty} adet @ {fiyat:.2f} ${kz}"
        if giris:
            mesaj += f"\nGiriş: {giris:.2f} $"
            try:
                tepe = zirve(veri, o.symbol, al_zaman, simdi)
                if tepe:
                    mesaj += f"\n📈 Alımdan bu yana en yüksek: {tepe:.2f} $ ({yuzde(tepe, giris)})"
            except Exception as e:
                kaydet(f"   zirve okunamadı: {e}")
        kaydet(mesaj)
        bildir(mesaj)
    durum["son_kontrol"] = pd.Timestamp(simdi).isoformat()


def gun_sonu_ozeti(trading, veri, hesap, durum, simdi):
    bugun = simdi.strftime("%Y-%m-%d")
    if durum.get("son_ozet") == bugun or simdi.hour < 16:
        return
    if not trading.get_calendar(GetCalendarRequest(start=simdi.date(), end=simdi.date())):
        return
    deger, dunku = float(hesap.equity), float(hesap.last_equity)
    emirler = dolan_emirler(trading, pd.Timestamp(simdi).normalize().to_pydatetime())
    hisse_kz = {}
    for o in emirler:
        tutar = float(o.filled_qty) * float(o.filled_avg_price)
        hisse_kz[o.symbol] = hisse_kz.get(o.symbol, 0) + (tutar if o.side == OrderSide.SELL else -tutar)
    alim = sum(o.side == OrderSide.BUY for o in emirler)
    satirlar = [f"📊 GÜN SONU ÖZETİ {bugun}",
                f"Hesap: {deger:,.2f} $  ({deger - dunku:+,.2f} $, %{(deger / dunku - 1) * 100:+.2f})",
                f"Alım: {alim}  Satım: {len(emirler) - alim}"]
    satirlar += [f"  {h}: {kz:+,.2f} $" for h, kz in sorted(hisse_kz.items(), key=lambda x: -x[1])]
    if not emirler:
        satirlar.append("Bugün işlem yapılmadı.")
    # alımdan kapanışa kadar görülen en yüksek fiyat
    kapanis = pd.Timestamp(simdi).normalize() + pd.Timedelta(hours=16)
    bugunku = []
    for h, d in islem_ciftleri(emirler).items():
        if not d["giris"] or not d["cikis"]:
            continue
        try:
            tepe = zirve(veri, h, d["al_zaman"], min(kapanis, pd.Timestamp(simdi) - pd.Timedelta(minutes=16)),
                         feed=DataFeed.SIP)
        except Exception as e:
            kaydet(f"   {h} zirve okunamadı: {e}")
            tepe = None
        kayit = {"tarih": bugun, "hisse": h, "giris": round(d["giris"], 2), "cikis": round(d["cikis"], 2),
                 "sonuc": round((d["cikis"] / d["giris"] - 1) * 100, 2),
                 "max": round((tepe / d["giris"] - 1) * 100, 2) if tepe else None}
        bugunku.append(kayit)
        satirlar.append(f"  📈 {h}: giriş {d['giris']:.2f} → çıkış {d['cikis']:.2f} ({yuzde(d['cikis'], d['giris'])})"
                        + (f", kapanışa kadar en yüksek {tepe:.2f} ({yuzde(tepe, d['giris'])})" if tepe else ""))
    if bugunku:
        durum["mfe"] = (durum.get("mfe", []) + bugunku)[-300:]
        tum = durum["mfe"]
        ort = lambda xs: sum(xs) / len(xs) if xs else 0
        maxlar = [x["max"] for x in tum if x["max"] is not None]
        y = lambda v: f"%{v:+.2f}".replace(".", ",")
        satirlar.append(f"📐 Bugün: ort. sonuç {y(ort([x['sonuc'] for x in bugunku]))}, "
                        f"ort. en yüksek {y(ort([x['max'] for x in bugunku if x['max'] is not None]))}")
        satirlar.append(f"📐 Tüm zamanlar ({len(tum)} işlem): ort. sonuç {y(ort([x['sonuc'] for x in tum]))}, "
                        f"ort. en yüksek {y(ort(maxlar))}")
    mesaj = "\n".join(satirlar)
    kaydet(mesaj)
    bildir(mesaj)
    durum["son_ozet"] = bugun


# ---------------------------------------------------------------- panel
def mfe_ozet(durum):
    tum = durum.get("mfe", [])
    if not tum:
        return {"Ort. sonuç": "henüz yok", "Ort. en yüksek": "henüz yok"}
    maxlar = [x["max"] for x in tum if x["max"] is not None]
    f = lambda v: f"%{v:+.2f}".replace(".", ",")
    return {"Ort. sonuç": f"{f(sum(x['sonuc'] for x in tum) / len(tum))} ({len(tum)} işlem)",
            "Ort. en yüksek": f(sum(maxlar) / len(maxlar)) if maxlar else "-"}


def panel_yaz(trading, durum):
    simdi = pd.Timestamp.now(tz=NY)
    hesap, saat = trading.get_account(), trading.get_clock()
    deger, dunku = float(hesap.equity), float(hesap.last_equity)
    hesap_no = str(getattr(hesap, "account_number", "") or "")
    if hesap_no and durum.get("hesap_no") != hesap_no:      # yeni hesaba geçildi: istatistikleri sıfırla
        for k in ("baslangic", "gecmis", "mfe"):
            durum.pop(k, None)
        durum["hesap_no"] = hesap_no
    durum.setdefault("baslangic", round(deger, 2))
    gecmis = durum.get("gecmis", {})
    gecmis[simdi.strftime("%Y-%m-%d")] = round(deger, 2)
    durum["gecmis"] = dict(sorted(gecmis.items())[-90:])

    pozisyonlar = [{"hisse": p.symbol, "adet": float(p.qty), "giris": round(float(p.avg_entry_price), 2),
                    "simdi": round(float(p.current_price), 2), "kz": round(float(p.unrealized_pl), 2),
                    "kz_yuzde": round(100 * float(p.unrealized_plpc), 2)} for p in trading.get_all_positions()]
    acik_emirler = [{"hisse": o.symbol, "yon": "AL" if o.side == OrderSide.BUY else "SAT",
                     "tur": str(o.type.value if hasattr(o.type, "value") else o.type),
                     "adet": float(o.qty or 0),
                     "fiyat": float(o.limit_price or o.stop_price or 0)}
                    for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50))]
    islemler = [{"zaman": pd.Timestamp(o.filled_at).tz_convert(NY).strftime("%H:%M"), "hisse": o.symbol,
                 "yon": "AL" if o.side == OrderSide.BUY else "SAT", "adet": float(o.filled_qty),
                 "fiyat": round(float(o.filled_avg_price), 2)}
                for o in dolan_emirler(trading, simdi.normalize().to_pydatetime())]
    tarama = []
    if SON_TARAMA is not None and not SON_TARAMA.empty:
        tarama = SON_TARAMA.sort_values("Puan", ascending=False).replace({np.nan: None}).to_dict("records")

    panel = {
        "guncelleme": simdi.isoformat(), "piyasa_acik": bool(saat.is_open),
        "sonraki_acilis": saat.next_open.isoformat(), "sonraki_kapanis": saat.next_close.isoformat(),
        "hesap": round(deger, 2), "dunku": round(dunku, 2), "baslangic": durum["baslangic"],
        "nakit": round(float(hesap.cash), 2), "alim_gucu": round(float(hesap.buying_power), 2),
        "gunluk_kz": round(deger - dunku, 2),
        "gunluk_yuzde": round((deger / dunku - 1) * 100, 2) if dunku else 0,
        "otomatik": OTOMATIK_ISLEM,
        "pozisyonlar": pozisyonlar, "acik_emirler": acik_emirler, "islemler": islemler,
        "tarama": tarama, "gecmis": durum["gecmis"], "kayit": KAYIT[-15:],
        "dogruluk": durum.get("dogruluk"),
        "kurallar": {"hacim": HACIM_ESIGI, "yukselis": FIYAT_ESIGI, "kontrol_mum": KONTROL_MUM},
        "risk": {"Pozisyon büyüklüğü": f"%{int(POZISYON_ORANI * 100)}", "En fazla hisse": MAKS_POZISYON,
                 "Kâr al": f"%{KAR_AL:g}".replace(".", ","), "Zarar kes": f"%{ZARAR_KES:g}".replace(".", ","),
                 "Günlük zarar limiti": f"%{GUNLUK_ZARAR_LIMITI:g}".replace(".", ","), "Kapanışta": "Hepsini sat",
                 **mfe_ozet(durum)},
    }
    os.makedirs("site", exist_ok=True)
    with open("site/panel.json", "w") as f:
        json.dump(panel, f, ensure_ascii=False)
    if SON_BARS:
        with open("site/analiz.json", "w") as f:
            json.dump({"guncelleme": simdi.isoformat(), "hisseler": SON_BARS}, f, separators=(",", ":"))


# ---------------------------------------------------------------- ana akış
def _calistir(test_modu, durum):
    trading, veri = istemciler()
    saat, hesap = trading.get_clock(), trading.get_account()
    deger, dunku = float(hesap.equity), float(hesap.last_equity)
    gun_ici_islem = int(hesap.daytrade_count or 0)
    gunluk = (deger / dunku - 1) * 100 if dunku else 0.0
    simdi = pd.Timestamp.now(tz=NY).to_pydatetime()

    kaydet(f"🤖 BOT ÇALIŞTI  {simdi:%Y-%m-%d %H:%M} (New York)  ·  otomatik işlem: {'AÇIK' if OTOMATIK_ISLEM else 'KAPALI'}")
    kaydet(f"💼 Hesap: {deger:,.0f} $   Bugün: %{gunluk:+.2f}   Gün içi işlem (5 gün): {gun_ici_islem}")

    if not test_modu:
        try:
            satislari_bildir(trading, veri, durum, simdi)
        except Exception as e:
            kaydet(f"Satış kontrolü hatası: {e}")

    if not saat.is_open:
        kaydet(f"🔒 Piyasa kapalı. Sonraki açılış: {saat.next_open.astimezone(simdi.tzinfo):%Y-%m-%d %H:%M} (NY)")
        if test_modu:
            tablo = tara(veri, simdi)
            kaydet(tablo.sort_values("Puan", ascending=False).to_string(index=False))
            return
        try:
            gun_sonu_ozeti(trading, veri, hesap, durum, simdi)
        except Exception as e:
            kaydet(f"Özet hatası: {e}")
        try:
            tara(veri, simdi)                       # panel kapalıyken de son veriyi göstersin
        except Exception as e:
            kaydet(f"Tarama hatası: {e}")
        if durum.get("dogruluk", {}).get("tarih") != simdi.strftime("%Y-%m-%d") and simdi.hour >= 16:
            try:
                kaydet("🧪 Günlük doğruluk testi başlıyor...")
                durum["dogruluk"] = dogruluk_testi(veri, simdi)
                d = durum["dogruluk"]
                kaydet(f"🧪 EMA sinyali isabeti %{d['ema']['isabet']} ({d['ema']['adet']} sinyal), "
                       f"radar sinyali isabeti %{d['radar']['isabet']} ({d['radar']['adet']} sinyal)")
            except Exception as e:
                kaydet(f"Doğruluk testi hatası: {e}")
        return

    kalan_dk = (saat.next_close - saat.timestamp).total_seconds() / 60
    pozisyonlar = trading.get_all_positions()
    for p in pozisyonlar:
        kaydet(f"📂 {p.symbol}: {p.qty} adet, giriş {float(p.avg_entry_price):.2f}, "
               f"şimdi {float(p.current_price):.2f}, %{100 * float(p.unrealized_plpc):+.2f}")

    if kalan_dk <= HEPSINI_KAPAT_DK:
        if pozisyonlar:
            trading.close_all_positions(cancel_orders=True)
            kaydet("🔔 Kapanışa az kaldı: tüm pozisyonlar kapatıldı.")
            bildir("🔔 Kapanışa az kaldı: tüm pozisyonlar kapatıldı.")
        tara(veri, simdi)
        return
    if gunluk <= -GUNLUK_ZARAR_LIMITI:
        trading.close_all_positions(cancel_orders=True)
        msg = f"🛑 Günlük zarar limiti (%{GUNLUK_ZARAR_LIMITI}) aşıldı. Her şey kapatıldı, bugün işlem yok."
        kaydet(msg)
        if durum.get("limit_bildirildi") != simdi.strftime("%Y-%m-%d"):
            bildir(msg)
            durum["limit_bildirildi"] = simdi.strftime("%Y-%m-%d")
        tara(veri, simdi)
        return

    tablo = tara(veri, simdi)
    sinyaller = sinyal_bul(tablo)
    kaydet(f"🔍 Tarama: {len(tablo)} hisse, {len(sinyaller)} AL sinyali, "
           f"{int((tablo['Etiket'] == 'İZLE').sum())} izlemede.")

    if not OTOMATIK_ISLEM:
        for _, s in sinyaller.iterrows():
            anahtar = f"{simdi:%Y-%m-%d}-{s['Hisse']}"
            if anahtar not in durum.get("bildirilen_sinyaller", []):
                bildir(f"📡 SİNYAL (otomatik işlem kapalı): {s['Hisse']} {s['Fiyat']} $\n{s['Gerekce']}")
                durum.setdefault("bildirilen_sinyaller", []).append(anahtar)
        durum["bildirilen_sinyaller"] = durum.get("bildirilen_sinyaller", [])[-50:]
        return
    if kalan_dk <= YENI_ISLEM_BITIS_DK:
        kaydet(f"⏳ Kapanışa {YENI_ISLEM_BITIS_DK} dk'dan az kaldı, yeni işlem açılmıyor.")
        return
    if simdi.strftime("%H:%M") < ILK_ISLEM_SAATI:
        kaydet("⏳ Açılışın ilk 15 dakikası, bekleniyor.")
        return
    if deger < 25000 and gun_ici_islem >= 3:
        kaydet("⚠️ 25.000 $ altı hesapta 5 günde 3 gün içi işlem sınırı doldu. Yeni işlem yok.")
        return
    bos = MAKS_POZISYON - len(pozisyonlar)
    if bos <= 0 or sinyaller.empty:
        return

    nakit = float(hesap.cash)
    eldekiler = {p.symbol for p in pozisyonlar}
    bekleyen = {o.symbol for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))}
    for _, s in sinyaller.iterrows():
        if bos <= 0:
            break
        if s["Hisse"] in eldekiler or s["Hisse"] in bekleyen:
            continue
        fiyat = s["Fiyat"]
        butce = min(deger * POZISYON_ORANI, nakit * 0.98)
        adet = int(butce / fiyat)
        if adet < 1:
            kaydet(f"   {s['Hisse']}: {fiyat} $, bütçe {butce:.2f} $ — 1 tam adet alınamıyor, atlandı.")
            continue
        emir = MarketOrderRequest(
            symbol=s["Hisse"], qty=adet, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(fiyat * (1 + KAR_AL / 100), 2)),
            stop_loss=StopLossRequest(stop_price=round(fiyat * (1 - ZARAR_KES / 100), 2)))
        try:
            trading.submit_order(emir)
            msg = (f"✅ ALINDI: {s['Hisse']} {adet} adet ~{fiyat} $ (≈{adet * fiyat:.2f} $)\n{s['Gerekce']} · 15 dk %{s['15dk%']}\n"
                   f"Kâr-al {fiyat * (1 + KAR_AL / 100):.2f} / Zarar-kes {fiyat * (1 - ZARAR_KES / 100):.2f}")
            kaydet(msg)
            bildir(msg)
            bos -= 1
            nakit -= adet * fiyat
        except Exception as e:
            kaydet(f"❌ {s['Hisse']} emri reddedildi: {e}")
            bildir(f"❌ {s['Hisse']} emri reddedildi: {str(e)[:200]}")


def calistir(test_modu=False):
    durum = durum_oku()
    try:
        _calistir(test_modu, durum)
    finally:
        if not test_modu:
            try:
                panel_yaz(istemciler()[0], durum)
            except Exception as e:
                (lambda *a,**k:None)(f"Panel hatası: {e}")
            durum_yaz(durum)


if __name__ == "__main__":
    try:
        calistir()
    except Exception as e:
        bildir(f"⚠️ BOT HATASI: {type(e).__name__}: {str(e)[:300]}")
        raise
