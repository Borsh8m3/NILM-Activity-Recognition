import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
import os

# --- KONFIGURACJA ---
st.set_page_config(page_title="NILM Energy Intelligence", layout="wide")

APPLIANCE_INFO = {
    'Appliance1': {'name': 'Fridge_Freezer', 'icon': '❄️'},
    'Appliance2': {'name': 'Chest_Freezer', 'icon': '🍦'},
    'Appliance3': {'name': 'Upright_Freezer', 'icon': '🧊'},
    'Appliance4': {'name': 'Tumble_Dryer', 'icon': '💨'},
    'Appliance5': {'name': 'Washing_Machine', 'icon': '🧺'},
    'Appliance6': {'name': 'Dishwasher', 'icon': '🍽️'},
    'Appliance7': {'name': 'Computer', 'icon': '💻'},
    'Appliance8': {'name': 'TV', 'icon': '📺'},
    'Appliance9': {'name': 'Electric_Heater', 'icon': '🔥'}
}

APPLIANCE_MAP = {k: v['name'] for k, v in APPLIANCE_INFO.items()}
TARGET_APPS = list(APPLIANCE_MAP.values())

@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(base_dir, "CLEAN_House4.csv")
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.rename(columns=APPLIANCE_MAP)
    start_idx_list = df[df['Time'].dt.strftime('%H:%M:%S') == '00:00:00'].index
    first_midnight = start_idx_list[0] if not start_idx_list.empty else 0
    
    loaded_models = {}
    for app in TARGET_APPS:
        model_path = os.path.join(base_dir, f"{app}.joblib")
        if os.path.exists(model_path):
            try: loaded_models[app] = joblib.load(model_path)
            except Exception as e: print(f"Błąd modelu {app}: {e}")
    return df, loaded_models, first_midnight

df_raw, models_dict, midnight_idx = load_assets()

# przygotowanie danych do heatmapy
@st.cache_data
def get_shifted_start_idx(df, start_idx, days_to_skip=30):
    """Oblicza indeks w dataframe przesunięty o określoną liczbę dni do przodu."""
    start_time = df.iloc[start_idx]['Time']
    end_time = start_time + pd.Timedelta(days=days_to_skip)
    future_idx_list = df[df['Time'] >= end_time].index
    return future_idx_list[0] if len(future_idx_list) > 0 else start_idx

# Obliczamy indeks startowy dla symulacji (30 dni do przodu od oryginalnego midnight_idx)
shifted_midnight_idx = get_shifted_start_idx(df_raw, midnight_idx, days_to_skip=30)

# --- STAN SESJI ---
if 'history' not in st.session_state: st.session_state.history = pd.DataFrame()
# Symulacja zawsze startuje z 30-dniowym wyprzedzeniem
if 'current_idx' not in st.session_state: 
    st.session_state.current_idx = shifted_midnight_idx
if 'running' not in st.session_state: st.session_state.running = False
if 'activity_logs' not in st.session_state: st.session_state.activity_logs = []
if 'last_notified_day' not in st.session_state: st.session_state.last_notified_day = None
if 'app_states' not in st.session_state: 
    st.session_state.app_states = {app: {'active': False, 'start_t': None} for app in TARGET_APPS}
if 'consumption' not in st.session_state:
    st.session_state.consumption = {app: 0.0 for app in TARGET_APPS}
if 'total_agg_kWh' not in st.session_state: st.session_state.total_agg_kWh = 0.0

# ZMIENNE DO PLANU DNIA I PODSUMOWAŃ
if 'daily_routine' not in st.session_state: st.session_state.daily_routine = []
if 'yesterday_routine' not in st.session_state: st.session_state.yesterday_routine = []
if 'yesterday_cost' not in st.session_state: st.session_state.yesterday_cost = 0.0

# --- SIDEBAR (USTAWIENIA) ---
st.sidebar.header("⚙️ Ustawienia")
selected_apps = st.sidebar.multiselect("Monitoring Wykresu", TARGET_APPS, default=['Washing_Machine', 'TV', 'Computer'])

st.sidebar.divider()
st.sidebar.subheader("🌱 Cele Energetyczne")
daily_limit = st.sidebar.number_input("Limit dzienny (kWh)", min_value=1.0, max_value=50.0, value=15.0)
energy_price = st.sidebar.number_input("Cena (PLN/kWh)", value=0.90)

st.sidebar.divider()
speed_factor = st.sidebar.slider("Prędkość symulacji", 1.0, 100.0, 10.0, 1.0)
step_size = int(max(1, speed_factor / 5)) 
sleep_time = (8.0 * step_size) / speed_factor

col_run, col_stop = st.sidebar.columns(2)
if col_run.button("▶ START"): st.session_state.running = True
if col_stop.button("⏹ STOP"): st.session_state.running = False

if st.sidebar.button("🧹 Resetuj system"):
    st.session_state.activity_logs = []
    st.session_state.history = pd.DataFrame()
    st.session_state.consumption = {app: 0.0 for app in TARGET_APPS}
    st.session_state.daily_routine = []
    st.session_state.yesterday_routine = []
    st.session_state.total_agg_kWh = 0.0
    st.session_state.current_idx = shifted_midnight_idx
    st.session_state.last_notified_day = None
    st.session_state.running = False
    st.rerun()

# --- LAYOUT GŁÓWNY ---
tab_monitor, tab_stats, tab_logs, tab_heatmap = st.tabs(["📊 Live Dashboard", "📈 Analiza Zużycia & Rutyna", "📜 Historia", "🔥 Analiza Aktywności"])

with tab_monitor:
    c1, c2 = st.columns([4, 1])
    with c1: chart_area = st.empty()
    with c2:
        st.write("### Eco-Score")
        eco_indicator = st.empty()
        progress_area = st.empty()

    st.write("### 🏷️ Status Urządzeń")
    status_area = st.empty()

with tab_stats:
    st.subheader("Profil Dnia Użytkownika")
    col_routine, col_pie = st.columns([1, 1])
    routine_area = col_routine.empty()
    pie_area = col_pie.empty()
    st.divider()
    rank_area = st.empty()

with tab_logs:
    log_display = st.container()

with tab_heatmap:
    st.subheader("🔥 Aktywności Urządzeń")
    st.markdown("<p style='color: #888;'>Wykres przedstawia dane o zachowaniach domowników. Jaśniejszy kolor oznacza wyższe prawdopodobieństwo uruchomienia o danej porze.</p>", unsafe_allow_html=True)
    
    # KLUCZOWY MOMENT: Wycinamy dane od oryginalnego początku historii (midnight_idx) 
    # aż do aktualnego momentu w symulacji (current_idx).
    # Przy starcie aplikacji jest to dokładnie 30 dni danych. Z każdym krokiem ta baza rośnie.
    current_slice = df_raw.iloc[midnight_idx : st.session_state.current_idx]
    
    if not current_slice.empty:
        hours = current_slice['Time'].dt.hour
        heatmap_records = []
        
        # Agregacja za pomocą Pandas
        for app in TARGET_APPS:
            if app in current_slice.columns:
                active_mask = current_slice[app] > 10.0 # Zakładamy próg aktywności > 10W
                hourly_activity = active_mask.groupby(hours).mean().reset_index()
                
                # Wyciągamy ikonę z Twojego słownika APPLIANCE_INFO (lub dajemy domyślną wtyczkę, jeśli nie ma)
                icon = next((v['icon'] for k, v in APPLIANCE_INFO.items() if v['name'] == app), '🔌')
                display_name = f"{app.replace('_', ' ')} {icon}"
                
                for _, row in hourly_activity.iterrows():
                    heatmap_records.append({
                        'Urządzenie': display_name,
                        'Godzina': f"{int(row['Time']):02d}:00",
                        'Aktywność': row[app]
                    })
        
        df_heatmap_live = pd.DataFrame(heatmap_records)
        
        if not df_heatmap_live.empty:
            st.vega_lite_chart(df_heatmap_live, {
                'height': {'step': 40}, # Wysokość każdego wiersza
                'mark': {
                    'type': 'rect',
                    'cornerRadius': 6,          
                    'stroke': '#1e1e1e',        
                    'strokeWidth': 2            
                },
                'encoding': {
                    'x': {
                        'field': 'Godzina', 
                        'type': 'ordinal', 
                        'title': None,          
                        'axis': {
                            'labelAngle': -45,  
                            'labelColor': '#aaaaaa',
                            'labelFontSize': 11,
                            'domain': False,    
                            'ticks': False      
                        }
                    },
                    'y': {
                        'field': 'Urządzenie', 
                        'type': 'nominal', 
                        'title': None,          
                        'axis': {
                            'labelColor': '#dddddd',
                            'labelFontSize': 13,
                            'labelFontWeight': 'bold',
                            'domain': False,
                            'ticks': False, 
                            'labelLimit': 500,
                            'labelOverlap': False # Wymusza pokazanie wszystkich etykiet
                        }
                    },
                    'color': {
                        'field': 'Aktywność', 
                        'type': 'quantitative',
                        'scale': {'scheme': 'inferno'}, 
                        'legend': {
                            'title': 'Częstotliwość użycia',
                            'titleColor': '#aaaaaa',
                            'labelColor': '#aaaaaa',
                            'orient': 'bottom', 
                            'gradientLength': 400, 
                            'gradientThickness': 10
                        }
                    },
                    'tooltip': [
                        {'field': 'Urządzenie', 'type': 'nominal', 'title': 'Urządzenie'},
                        {'field': 'Godzina', 'type': 'ordinal', 'title': '🕒 Godzina'},
                        {'field': 'Aktywność', 'type': 'quantitative', 'format': '.1%', 'title': '⚡ Aktywność'}
                    ]
                },
                'config': {
                    'view': {'stroke': 'transparent'}, 
                    'axis': {'grid': False},           
                    'background': 'transparent'        
                }
            }, use_container_width=True)
        else:
            st.warning("Brak wystarczających danych do agregacji.")
    else:
        st.info("Trwa ładowanie danych historycznych...")

# --- SYMULACJA ---
if st.session_state.running:
    idx = st.session_state.current_idx
    if idx < len(df_raw):
        rows = df_raw.iloc[idx : idx + step_size].copy()
        st.session_state.history = pd.concat([st.session_state.history, rows]).tail(600)
        st.session_state.current_idx += step_size
        df_win = st.session_state.history
        
        last_row = rows.iloc[-1]
        t = last_row['Time']
        current_date = t.date()
        
        # Inicjalizacja dnia (uruchamia się tylko raz na początku)
        if st.session_state.last_notified_day is None:
            st.session_state.last_notified_day = current_date
        
        new_kWh = (rows['Aggregate'].sum() * 8) / 3600000
        st.session_state.total_agg_kWh += new_kWh

        # 1. DETEKCJA ZMIANY DNIA I RESET
        if current_date != st.session_state.last_notified_day:
            # Zapisujemy podsumowanie minionego dnia
            st.session_state.yesterday_routine = list(st.session_state.daily_routine)
            st.session_state.yesterday_cost = st.session_state.total_agg_kWh * energy_price
            
            # Tworzymy log
            st.session_state.activity_logs.insert(0, f"📅 ZAKOŃCZONO DZIEŃ. Koszt: {st.session_state.yesterday_cost:.2f} PLN")
            st.session_state.activity_logs.insert(0, f"📅 SYSTEM: Rozpoczęto nowy dzień {current_date}")
            
            # Resetujemy liczniki i listy na nowy dzień
            st.session_state.daily_routine = []
            st.session_state.total_agg_kWh = 0.0
            st.session_state.consumption = {app: 0.0 for app in TARGET_APPS}
            
            # Aktualizujemy śledzenie dnia
            st.session_state.last_notified_day = current_date

        # 2. WYKRES
        with chart_area.container():
            plot_df = df_win.copy().set_index('Time')
            plot_df = plot_df.loc[t - pd.Timedelta(minutes=20) : t]
            st.line_chart(plot_df[['Aggregate'] + [c for c in selected_apps if c in plot_df.columns]], height=350)

        active_heavy = 0
        heavy_list = ['Washing_Machine', 'Tumble_Dryer', 'Electric_Heater', 'Dishwasher']

        # 3. PREDYKCJA I KAFELKI
        with status_area.container():
            if len(df_win) >= 37:
                cols_ui = st.columns(len(TARGET_APPS))
                feat = pd.DataFrame([{
                    'Aggregate': float(last_row['Aggregate']),
                    'Time': float(t.hour + t.minute / 60),
                    'day_of_week': int(t.dayofweek),
                    'day_progress': float((t.hour*3600 + t.minute*60 + t.second)/86400),
                    'agg_lag_1': float(df_win.iloc[-2]['Aggregate']),
                    'agg_lag_2': float(df_win.iloc[-3]['Aggregate']),
                    'agg_lag_3': float(df_win.iloc[-4]['Aggregate']),
                    'power_delta': float(last_row['Aggregate'] - df_win.iloc[-2]['Aggregate']),
                    'roll_mean_5min': float(df_win['Aggregate'].tail(37).mean()),
                    'roll_std_5min': float(df_win['Aggregate'].tail(37).std()),
                    'hour_sin': np.sin(2 * np.pi * t.hour / 24), 'hour_cos': np.cos(2 * np.pi * t.hour / 24),
                    'day_sin': np.sin(2 * np.pi * t.dayofweek / 7), 'day_cos': np.cos(2 * np.pi * t.dayofweek / 7),
                    'minute_sin': np.sin(2 * np.pi * t.minute / 60), 'minute_cos': np.cos(2 * np.pi * t.minute / 60)
                }])

                for i, (app_id, info) in enumerate(APPLIANCE_INFO.items()):
                    app_name = info['name']
                    is_active = False
                    if app_name in models_dict:
                        try: is_active = models_dict[app_name].predict(feat.values)[0] > 0.5
                        except: pass
                    
                    if is_active:
                        if app_name in heavy_list: active_heavy += 1
                        p_est = {'Fridge_Freezer': 150, 'Washing_Machine': 2200, 'TV': 120, 'Computer': 250, 'Electric_Heater': 2000}
                        st.session_state.consumption[app_name] += (p_est.get(app_name, 200) * 8 * step_size) / 3600000

                    state = st.session_state.app_states[app_name]
                    
                    # LOGIKA WYKRYWANIA WŁĄCZENIA
                    if is_active and not state['active']:
                        st.session_state.app_states[app_name] = {'active': True, 'start_t': t}
                        
                        # Dodajemy do listy zdarzeń (ignorując lodówki)
                        if app_name not in ['Fridge_Freezer', 'Chest_Freezer', 'Upright_Freezer']:
                            st.session_state.daily_routine.append({
                                'time': t.strftime('%H:%M'),
                                'app': app_name
                            })
                            
                    elif not is_active and state['active']:
                        dur = (t - state['start_t']).total_seconds()
                        if dur > 10:
                            m, s = divmod(int(dur), 60)
                            st.session_state.activity_logs.insert(0, f"{info['icon']} **{app_name.replace('_',' ')}**: {state['start_t'].strftime('%H:%M')} - {t.strftime('%H:%M')} ({m}m {s}s)")
                        st.session_state.app_states[app_name] = {'active': False, 'start_t': None}

                    bg = "#2ca02c" if is_active else "#1e212b"
                    border = "#2ca02c" if is_active else "#3d414d"
                    txt = "white" if is_active else "#888"
                    shadow = "0px 0px 15px rgba(44, 160, 44, 0.5)" if is_active else "none"

                    cols_ui[i].markdown(f"""
                        <div style="background-color:{bg}; color:{txt}; padding:10px; border-radius:8px; 
                        text-align:center; border: 2px solid {border}; box-shadow: {shadow}; height: 100px;">
                        <p style="margin:0; font-size:22px;">{info['icon']}</p>
                        <p style="margin:0; font-size:9px; font-weight:bold;">{app_name.replace('_',' ')}</p>
                        <b style="font-size:12px;">{"ACTIVE" if is_active else "OFF"}</b></div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"Trwa inicjalizacja modeli AI... {len(df_win)}/37")

        # 4. ECO-INDICATOR REFRESH
        with eco_indicator:
            if active_heavy >= 3: st.markdown("<h2 style='color: #d62728; text-align:center;'>🔴 Low</h2>", unsafe_allow_html=True)
            elif active_heavy >= 1: st.markdown("<h2 style='color: #ff7f0e; text-align:center;'>🟠 Med</h2>", unsafe_allow_html=True)
            else: st.markdown("<h2 style='color: #2ca02c; text-align:center;'>🟢 High</h2>", unsafe_allow_html=True)

        with progress_area:
            prog = min(st.session_state.total_agg_kWh / daily_limit, 1.0)
            st.progress(prog)
            st.write(f"Zużyto dziś: {st.session_state.total_agg_kWh:.2f} / {daily_limit} kWh")

        # 5. STATYSTYKI I RUTYNA W TABIE
        with routine_area:
            # Pokaż podsumowanie z wczoraj (jeśli symulacja przeszła przez północ)
            if st.session_state.yesterday_routine or st.session_state.yesterday_cost > 0:
                with st.expander(f"📁 Zobacz podsumowanie wczorajszego dnia (Koszt: {st.session_state.yesterday_cost:.2f} PLN)"):
                    if not st.session_state.yesterday_routine:
                        st.write("Brak aktywności człowieka.")
                    else:
                        for item in st.session_state.yesterday_routine:
                            icon = next((v['icon'] for k, v in APPLIANCE_INFO.items() if v['name'] == item['app']), '🔌')
                            st.markdown(f"**{item['time']}** — {icon} Uruchomiono: {item['app'].replace('_', ' ')}")
            
            # Pokaż aktualny plan dnia (lista rośnie z czasem)
            st.write(f"**🕰️ Aktualny Plan Dnia ({st.session_state.last_notified_day})**")
            if st.session_state.daily_routine:
                for item in st.session_state.daily_routine:
                    icon = next((v['icon'] for k, v in APPLIANCE_INFO.items() if v['name'] == item['app']), '🔌')
                    st.markdown(f"**{item['time']}** — {icon} Uruchomiono: {item['app'].replace('_', ' ')}")
            else:
                st.info("Czekam na pierwsze aktywności człowieka w tym dniu... (ignoruję lodówki)")

        stats_data = [{'Urządzenie': k.replace('_',' '), 'kWh': v} 
                      for k, v in st.session_state.consumption.items() if v > 0]
        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            with pie_area:
                st.write("**Struktura dzisiejszego zużycia**")
                st.vega_lite_chart(df_stats, {
                    'mark': {'type': 'arc', 'innerRadius': 40},
                    'encoding': {'theta': {'field': 'kWh', 'type': 'quantitative'}, 'color': {'field': 'Urządzenie', 'type': 'nominal'}}
                })
            with rank_area:
                st.dataframe(df_stats[['Urządzenie', 'kWh']].sort_values(by='kWh', ascending=False), hide_index=True, use_container_width=True)

        with tab_logs:
            for entry in st.session_state.activity_logs[:25]: st.write(entry)

        time.sleep(sleep_time)
        st.rerun()