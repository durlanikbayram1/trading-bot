"""
OTONOM TARAMA VE AL-SAT BOTU (öğrenme amaçlı, SANAL hesap)
Her çalıştırıldığında: piyasayı kontrol eder, riskleri denetler, hisseleri tarar,
sinyal varsa kâr-al ve zarar-kes emirleriyle birlikte alım yapar.
Strateji: hacim patlaması + yükseliş (Ders 2-4). Testlerimize göre KAZANDIRMIYOR;
amaç altyapıyı öğrenmek. İleride sadece `sinyal_bul` fonksiyonu değiştirilecek.
"""
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest, TakeProfitRequest,
                                     StopLossRequest, GetOrdersRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# ===================== AYARLAR =====================
HISSELER = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD",
            "NFLX", "PLTR", "INTC", "COIN", "SOFI", "MARA", "RIVN", "NIO",
            "UBER", "SHOP", "SNAP", "F"]
HACIM_ESIGI = 3.0          # hacim, önceki günlerin aynı saatindeki ortalamanın kaç katı
FIYAT_ESIGI = 0.5          # son 15 dk'da en az % kaç yükseliş
MAKS_POZISYON = 3          # aynı anda en fazla kaç hisse
POZISYON_ORANI = 0.10      # her pozisyon hesabın %10'u
KAR_AL = 1.5               # % kârda otomatik sat
ZARAR_KES = 1.0            # % zararda otomatik sat
GUNLUK_ZARAR_LIMITI = 2.0  # hesap gün içinde % bu kadar düşerse her şeyi kapat, o gün dur
ILK_ISLEM_SAATI = "09:45"  # açılışın ilk 15 dk'sında işlem yok
YENI_ISLEM_BITIS_DK = 45   # kapanışa 45 dk kala yeni işlem yok
HEPSINI_KAPAT_DK = 20      # kapanışa 20 dk kala tüm pozisyonları kapat (GitHub gecikmelerine pay)
# ===================================================

NY = "America/New_York"


def istemciler():
    key, secret = os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def gostergeler(bars, simdi):
    """Alpaca'nın çoklu-hisse mum tablosundan her hissenin son TAMAMLANMIŞ mumunu değerlendirir."""
    satirlar = []
    for h in bars.index.get_level_values(0).unique():
        d = bars.xs(h, level=0).copy()
        d.index = pd.to_datetime(d.index).tz_convert(NY)
        d = d[d.index + pd.Timedelta(minutes=5) <= simdi]           # bitmemiş mumu at
        d["saat"] = d.index.strftime("%H:%M")
        d = d[(d["saat"] >= "09:30") & (d["saat"] < "16:00")]
        if len(d) < 30:
            continue
        d["tarih"] = d.index.date
        d["ort_hacim"] = d.groupby("saat")["volume"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        d["HacimX"] = d["volume"] / d["ort_hacim"]
        d["degisim"] = (d["close"] / d.groupby("tarih")["close"].shift(3) - 1) * 100
        son = d.iloc[-1]
        satirlar.append({"Hisse": h, "Zaman": d.index[-1].strftime("%m-%d %H:%M"),
                         "Fiyat": round(float(son["close"]), 2),
                         "HacimX": round(float(son["HacimX"]), 1) if pd.notna(son["HacimX"]) else None,
                         "15dk%": round(float(son["degisim"]), 2) if pd.notna(son["degisim"]) else None})
    return pd.DataFrame(satirlar)


def tara(veri, simdi):
    istek = StockBarsRequest(symbol_or_symbols=HISSELER,
                             timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                             start=simdi - timedelta(days=12), end=simdi,
                             feed=DataFeed.IEX)   # ücretsiz planda anlık veri IEX'ten gelir
    bars = veri.get_stock_bars(istek).df
    return gostergeler(bars, simdi)


def sinyal_bul(tablo):
    t = tablo.dropna()
    return t[(t["HacimX"] >= HACIM_ESIGI) & (t["15dk%"] >= FIYAT_ESIGI)].sort_values("HacimX", ascending=False)


def calistir(test_modu=False):
    trading, veri = istemciler()
    saat = trading.get_clock()
    hesap = trading.get_account()
    deger, dunku = float(hesap.equity), float(hesap.last_equity)
    gun_ici_islem = int(hesap.daytrade_count or 0)
    gunluk = (deger / dunku - 1) * 100 if dunku else 0.0
    simdi = pd.Timestamp.now(tz=NY).to_pydatetime()

    print(f"🤖 BOT ÇALIŞTI  {simdi.strftime('%Y-%m-%d %H:%M')} (New York)")
    print(f"💼 Hesap: {deger:,.0f} $   Bugün: %{gunluk:+.2f}   Gün içi işlem sayısı (5 gün): {gun_ici_islem}")

    if not saat.is_open:
        print(f"🔒 Piyasa kapalı. Sonraki açılış: {saat.next_open.astimezone(simdi.tzinfo):%Y-%m-%d %H:%M} (NY)")
        if not test_modu:
            return
        print("\n🧪 TEST MODU: son seansın verisiyle tarama yapılıyor, EMİR GÖNDERİLMEYECEK.\n")
        tablo = tara(veri, simdi)
        print(tablo.sort_values("HacimX", ascending=False).to_string(index=False))
        s = sinyal_bul(tablo)
        print("\n🚀 Bot şu an açık olsaydı alacakları:" if not s.empty else "\nSinyal yok.")
        if not s.empty:
            print(s.head(MAKS_POZISYON).to_string(index=False))
        return

    kalan_dk = (saat.next_close - saat.timestamp).total_seconds() / 60
    pozisyonlar = trading.get_all_positions()
    if pozisyonlar:
        print("📂 Açık pozisyonlar:")
        for p in pozisyonlar:
            print(f"   {p.symbol}: {p.qty} adet, giriş {float(p.avg_entry_price):.2f}, "
                  f"şimdi {float(p.current_price):.2f}, kâr/zarar %{100 * float(p.unrealized_plpc):+.2f}")

    # --- Risk kuralları ---
    if kalan_dk <= HEPSINI_KAPAT_DK:
        if pozisyonlar:
            trading.close_all_positions(cancel_orders=True)
            print("🔔 Kapanışa az kaldı: tüm pozisyonlar kapatıldı.")
        return
    if gunluk <= -GUNLUK_ZARAR_LIMITI:
        trading.close_all_positions(cancel_orders=True)
        print(f"🛑 Günlük zarar limiti (%{GUNLUK_ZARAR_LIMITI}) aşıldı. Her şey kapatıldı, bugün işlem yok.")
        return
    if kalan_dk <= YENI_ISLEM_BITIS_DK:
        print(f"⏳ Kapanışa {YENI_ISLEM_BITIS_DK} dk'dan az kaldı, yeni işlem açılmıyor.")
        return
    if simdi.strftime("%H:%M") < ILK_ISLEM_SAATI:
        print("⏳ Açılışın ilk 15 dakikası, bekleniyor.")
        return
    if deger < 25000 and gun_ici_islem >= 3:
        print("⚠️ 25.000 $ altı hesapta 5 günde 3 gün içi işlem sınırına ulaşıldı. Yeni işlem yok.")
        return

    bos = MAKS_POZISYON - len(pozisyonlar)
    if bos <= 0:
        print("📦 Pozisyon sınırı dolu, yeni alım yok.")
        return

    # --- Tarama ---
    tablo = tara(veri, simdi)
    sinyaller = sinyal_bul(tablo)
    if sinyaller.empty:
        print("🔍 Tarama tamam: sinyal yok.")
        return

    eldekiler = {p.symbol for p in pozisyonlar}
    bekleyen = {o.symbol for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))}
    for _, s in sinyaller.iterrows():
        if bos <= 0:
            break
        if s["Hisse"] in eldekiler or s["Hisse"] in bekleyen:
            continue
        fiyat = s["Fiyat"]
        adet = int(deger * POZISYON_ORANI / fiyat)
        if adet < 1:
            print(f"   {s['Hisse']}: hesap küçük, 1 adet bile alınamıyor.")
            continue
        emir = MarketOrderRequest(
            symbol=s["Hisse"], qty=adet, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(fiyat * (1 + KAR_AL / 100), 2)),
            stop_loss=StopLossRequest(stop_price=round(fiyat * (1 - ZARAR_KES / 100), 2)))
        try:
            trading.submit_order(emir)
            print(f"✅ ALINDI: {s['Hisse']} {adet} adet ~{fiyat} $  (hacim {s['HacimX']}x, 15dk %{s['15dk%']})  "
                  f"kâr-al {fiyat * (1 + KAR_AL / 100):.2f} / zarar-kes {fiyat * (1 - ZARAR_KES / 100):.2f}")
            bos -= 1
        except Exception as e:
            print(f"❌ {s['Hisse']} emri reddedildi: {e}")


if __name__ == "__main__":
    calistir()
