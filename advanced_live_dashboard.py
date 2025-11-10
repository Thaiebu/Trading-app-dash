"""
Advanced Options Trading Dashboard - 3 Sections (FIXED)
Section 3 now properly shows Futures OI on a secondary Y-axis with matching timestamps
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Advanced Options Dashboard - 3 Sections"

# Color scheme
COLORS = {
    'call': '#26a69a',
    'put': '#ef5350',
    'future': '#FFA726',
    'background': '#1e1e1e',
    'text': '#ffffff',
    'card_bg': "#8d8a8a"
}

# ============== FIXED DATA GENERATION FUNCTIONS ==============

def generate_time_series(num_intervals=10, interval='3Min'):
    """Generate consistent timestamps for both options and futures"""
    return pd.date_range(end=datetime.now(), periods=num_intervals, freq=interval)


def generate_options_data(symbol, current_price, expiry_date, times, is_live=True):
    """Generate realistic options data with historical tracking - FIXED"""
    np.random.seed(42)
    
    # Strike intervals based on symbol
    strike_interval = 50 if symbol in ['NIFTY50', 'FINNIFTY'] else 100
    base = int(current_price / strike_interval) * strike_interval
    strikes = [base + (i * strike_interval) for i in range(-20, 21)]
    
    # Initialize data storage
    historical_data = []
    
    for time_idx, timestamp in enumerate(times):
        for strike in strikes:
            moneyness = abs((current_price - strike) / current_price)
            distance_factor = np.exp(-moneyness * 10)
            
            # Base OI with time variation
            base_call_oi = int(np.random.randint(20000, 80000) * distance_factor)
            base_put_oi = int(np.random.randint(25000, 90000) * distance_factor)
            
            # Add ATM boost
            if abs(moneyness) < 0.01:
                base_call_oi *= 3
                base_put_oi *= 3.2
            
            # Add time-based variation (simulate OI buildup/reduction)
            time_factor = 1 + (time_idx * 0.05)  # OI increases over time
            noise = np.random.uniform(-0.1, 0.1)
            
            call_oi = int(base_call_oi * time_factor * (1 + noise))
            put_oi = int(base_put_oi * time_factor * (1 + noise))
            
            historical_data.append({
                'timestamp': timestamp,
                'strike': strike,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'call_volume': int(np.random.randint(1000, 10000) * distance_factor),
                'put_volume': int(np.random.randint(1000, 10000) * distance_factor),
                'call_price': max(5, current_price - strike + np.random.uniform(-50, 50)),
                'put_price': max(5, strike - current_price + np.random.uniform(-50, 50))
            })
    
    df = pd.DataFrame(historical_data)
    
    # Calculate change in OI
    latest = df[df['timestamp'] == df['timestamp'].max()].copy()
    previous = df[df['timestamp'] == df['timestamp'].unique()[-2]].copy()
    
    merged = latest.merge(previous, on='strike', suffixes=('_current', '_previous'))
    merged['call_oi_change'] = merged['call_oi_current'] - merged['call_oi_previous']
    merged['put_oi_change'] = merged['put_oi_current'] - merged['put_oi_previous']
    
    return df, merged


def generate_futures_data(symbol, times):
    """Generate futures OI data - FIXED to use consistent timestamps"""
    
    # Use realistic base OI values based on symbol
    base_oi_map = {
        'NIFTY50': 8500000,
        'BANKNIFTY': 4200000,
        'FINNIFTY': 1800000
    }
    base_oi = base_oi_map.get(symbol, 5000000)
    
    data = []
    for idx, timestamp in enumerate(times):
        # Create visible trend with realistic variation
        trend = idx * 80000  # Gradual increase
        variation = np.random.randint(-150000, 150000)
        oi = base_oi + trend + variation
        
        data.append({
            'timestamp': timestamp,
            'futures_oi': oi
        })
    
    return pd.DataFrame(data)


def get_current_price(symbol):
    """Get simulated current price"""
    prices = {
        'NIFTY50': 24500,
        'BANKNIFTY': 51500,
        'FINNIFTY': 22800
    }
    return prices.get(symbol, 24500)


def get_expiry_dates():
    """Generate next 4 expiry dates (weekly)"""
    today = datetime.now()
    # Find next Thursday
    days_ahead = 3 - today.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    
    next_thursday = today + timedelta(days=days_ahead)
    
    expiries = []
    for i in range(4):
        expiry = next_thursday + timedelta(weeks=i)
        expiries.append(expiry.strftime('%d-%m-%Y'))
    
    return expiries


# App Layout
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '20px'}, children=[
    
    # Header
    html.H1('🔴 ADVANCED OPTIONS TRADING DASHBOARD - 3 SECTIONS', 
            style={'textAlign': 'center', 'marginBottom': '10px'}),
    
    html.Div(id='last-update-time', 
             style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '20px'}),
    
    # Auto-refresh interval
    dcc.Interval(id='interval-refresh', interval=3*60*1000, n_intervals=0),  # 3 minutes
    
    # ========== CONTROL PANEL ==========
    html.Div([
        html.H3('📊 Control Panel', style={'marginBottom': '15px'}),
        
        html.Div([
            # Symbol Selection
            html.Div([
                html.Label('Symbol:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='symbol-select',
                    options=[
                        {'label': 'NIFTY 50', 'value': 'NIFTY50'},
                        {'label': 'BANK NIFTY', 'value': 'BANKNIFTY'},
                        {'label': 'FIN NIFTY', 'value': 'FINNIFTY'}
                    ],
                    value='NIFTY50',
                    style={'backgroundColor': COLORS['card_bg'], 'color':  '#1e1e1e'}
                )
            ], style={'flex': '1', 'marginRight': '10px'}),
            
            # Expiry Selection
            html.Div([
                html.Label('Expiry:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='expiry-select',
                    options=[{'label': exp, 'value': exp} for exp in get_expiry_dates()],
                    value=get_expiry_dates()[0],
                    style={'backgroundColor': COLORS['card_bg'], 'color':  '#1e1e1e'}
                )
            ], style={'flex': '1', 'marginRight': '10px'}),
            
            # Interval Selection
            html.Div([
                html.Label('Interval:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='interval-select',
                    options=[
                        {'label': '1 Min', 'value': '1Min'},
                        {'label': '3 Min', 'value': '3Min'},
                        {'label': '5 Min', 'value': '5Min'},
                        {'label': '15 Min', 'value': '15Min'},
                        {'label': '1 Hour', 'value': '1H'}
                    ],
                    value='3Min',
                    style={'backgroundColor': COLORS['card_bg'], 'color':  '#1e1e1e'}
                )
            ], style={'flex': '1', 'marginRight': '10px'}),
            
        ], style={'display': 'flex', 'marginBottom': '15px'}),
        
        html.Div([
            # Range Checkbox and Inputs
            html.Div([
                dcc.Checklist(
                    id='range-checkbox',
                    options=[{'label': ' Enable Range Filter', 'value': 'enabled'}],
                    value=[],
                    style={'marginBottom': '10px'}
                ),
                html.Div([
                    html.Div([
                        html.Label('Range Start:', style={'marginRight': '5px'}),
                        dcc.Input(id='range-start', type='number', value=24000, 
                                 style={'width': '100px', 'backgroundColor': COLORS['card_bg'], 
                                       'color':  '#1e1e1e', 'border': '1px solid #555'})
                    ], style={'marginRight': '15px'}),
                    html.Div([
                        html.Label('Range End:', style={'marginRight': '5px'}),
                        dcc.Input(id='range-end', type='number', value=25000,
                                 style={'width': '100px', 'backgroundColor': COLORS['card_bg'],
                                       'color':  '#1e1e1e', 'border': '1px solid #555'})
                    ])
                ], style={'display': 'flex'})
            ], style={'flex': '1', 'marginRight': '20px'}),
            
            # Live / Historical Selection
            html.Div([
                dcc.RadioItems(
                    id='live-historical',
                    options=[
                        {'label': ' Live Data', 'value': 'live'},
                        {'label': ' Historical Data', 'value': 'historical'}
                    ],
                    value='live',
                    style={'marginBottom': '10px'}
                ),
                html.Div([
                    html.Label('Historical Date:', style={'marginRight': '5px'}),
                    dcc.DatePickerSingle(
                        id='historical-date',
                        date=datetime.now().date(),
                        display_format='DD-MM-YYYY',
                        style={'backgroundColor': COLORS['card_bg']}
                    )
                ], id='historical-date-div')
            ], style={'flex': '1'}),
            
        ], style={'display': 'flex'}),
        
    ], style={'backgroundColor': COLORS['card_bg'], 'padding': '20px', 'borderRadius': '10px', 
             'marginBottom': '30px'}),
    
    html.Hr(style={'borderColor': '#555'}),
    
    # ========== SECTION 1: Call vs Put Open Interest ==========
    html.Div([
        html.H2('📊 SECTION 1: Call vs Put Open Interest', 
                style={'marginBottom': '20px', 'color': COLORS['call']}),
        
        html.Div(id='section1-info', style={'marginBottom': '15px', 'fontSize': '14px'}),
        
        dcc.Graph(id='section1-oi-chart', style={'marginBottom': '20px'}),
        
    ], style={'marginBottom': '40px'}),
    
    html.Hr(style={'borderColor': '#555'}),
    
    # ========== SECTION 2: Change in OI (Bar Chart) ==========
    html.Div([
        html.H2('📈 SECTION 2: Change in Open Interest (Bar Chart)', 
                style={'marginBottom': '20px', 'color': COLORS['put']}),
        
        dcc.Graph(id='section2-oi-change-bar', style={'marginBottom': '20px'}),
        
    ], style={'marginBottom': '40px'}),
    
    html.Hr(style={'borderColor': '#555'}),
    
    # ========== SECTION 3: Change in OI (Line Chart) ==========
    html.Div([
        html.H2('📉 SECTION 3: Change in OI Over Time (Line Chart - CE, PE, Futures)', 
                style={'marginBottom': '20px', 'color': COLORS['future']}),
        
        dcc.Graph(id='section3-oi-change-line', style={'marginBottom': '20px'}),
        
    ], style={'marginBottom': '40px'}),
    
])


# Callback to update all sections
@app.callback(
    [Output('last-update-time', 'children'),
     Output('section1-info', 'children'),
     Output('section1-oi-chart', 'figure'),
     Output('section2-oi-change-bar', 'figure'),
     Output('section3-oi-change-line', 'figure'),
     Output('historical-date-div', 'style')],
    [Input('interval-refresh', 'n_intervals'),
     Input('symbol-select', 'value'),
     Input('expiry-select', 'value'),
     Input('interval-select', 'value'),
     Input('range-checkbox', 'value'),
     Input('range-start', 'value'),
     Input('range-end', 'value'),
     Input('live-historical', 'value'),
     Input('historical-date', 'date')]
)
def update_all_sections(n, symbol, expiry, interval, range_enabled, range_start, 
                       range_end, live_mode, hist_date):
    """Update all three sections with selected parameters"""
    
    # Get current price and generate data
    current_price = get_current_price(symbol)
    is_live = (live_mode == 'live')
    
    # FIXED: Generate consistent timestamps first
    times = generate_time_series(num_intervals=10, interval=interval)
    
    # Generate options data with consistent timestamps
    df_full, df_change = generate_options_data(symbol, current_price, expiry, times, is_live)
    
    # Generate futures data with same timestamps
    df_futures = generate_futures_data(symbol, times)
    
    # Apply range filter if enabled
    if 'enabled' in range_enabled and range_start and range_end:
        df_filtered = df_full[(df_full['strike'] >= range_start) & (df_full['strike'] <= range_end)]
        df_change_filtered = df_change[(df_change['strike'] >= range_start) & (df_change['strike'] <= range_end)]
    else:
        df_filtered = df_full
        df_change_filtered = df_change
    
    # Get latest data
    latest_time = df_filtered['timestamp'].max()
    latest_data = df_filtered[df_filtered['timestamp'] == latest_time]
    
    # Update time display
    mode_text = "LIVE" if is_live else f"Historical ({hist_date})"
    last_update = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} | Mode: {mode_text} | Refreshing every {interval}"
    
    # Section info
    total_call_oi = latest_data['call_oi'].sum()
    total_put_oi = latest_data['put_oi'].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    
    section1_info = html.Div([
        html.Div([
            html.Span(f"Symbol: {symbol} | ", style={'fontWeight': 'bold'}),
            html.Span(f"Expiry: {expiry} | ", style={'fontWeight': 'bold'}),
            html.Span(f"As of {latest_time.strftime('%H:%M')} on {expiry}", style={'fontWeight': 'bold'})
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Span(f"Total Call OI: {total_call_oi:,.0f} | ", style={'color': COLORS['call']}),
            html.Span(f"Total Put OI: {total_put_oi:,.0f} | ", style={'color': COLORS['put']}),
            html.Span(f"PCR: {pcr:.4f}", style={'color': COLORS['put'] if pcr > 1 else COLORS['call']})
        ])
    ])
    
    # ========== SECTION 1: Call vs Put OI Bar Chart ==========
    oi_by_strike = latest_data.groupby('strike').agg({
        'call_oi': 'sum',
        'put_oi': 'sum'
    }).reset_index()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=oi_by_strike['strike'],
        y=oi_by_strike['call_oi'],
        name='Call OI',
        marker_color=COLORS['call'],
        hovertemplate='Strike: %{x}<br>Call OI: %{y:,.0f}<extra></extra>'
    ))
    fig1.add_trace(go.Bar(
        x=oi_by_strike['strike'],
        y=oi_by_strike['put_oi'],
        name='Put OI',
        marker_color=COLORS['put'],
        hovertemplate='Strike: %{x}<br>Put OI: %{y:,.0f}<extra></extra>'
    ))
    fig1.add_vline(x=current_price, line_dash="dash", line_color="yellow", 
                   line_width=2, annotation_text=f"Spot: ₹{current_price:,.0f}")
    fig1.update_layout(
        title=f'{symbol} - Call vs Put Open Interest',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        template='plotly_dark',
        barmode='group',
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    # ========== SECTION 2: Change in OI Bar Chart ==========
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_change_filtered['strike'],
        y=df_change_filtered['call_oi_change'],
        name='Call OI Change',
        marker_color=COLORS['call'],
        hovertemplate='Strike: %{x}<br>Call Change: %{y:,.0f}<extra></extra>'
    ))
    fig2.add_trace(go.Bar(
        x=df_change_filtered['strike'],
        y=df_change_filtered['put_oi_change'],
        name='Put OI Change',
        marker_color=COLORS['put'],
        hovertemplate='Strike: %{x}<br>Put Change: %{y:,.0f}<extra></extra>'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig2.add_vline(x=current_price, line_dash="dash", line_color="yellow", line_width=2)
    fig2.update_layout(
        title=f'{symbol} - Change in Open Interest (Last Interval)',
        xaxis_title='Strike Price',
        yaxis_title='OI Change',
        template='plotly_dark',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    # ========== SECTION 3: Change in OI Line Chart (FIXED WITH DUAL Y-AXIS) ==========
    # Aggregate total OI by timestamp
    time_series = df_filtered.groupby('timestamp').agg({
        'call_oi': 'sum',
        'put_oi': 'sum'
    }).reset_index()
    
    # Merge with futures OI (now timestamps match!)
    time_series = time_series.merge(df_futures, on='timestamp', how='left')
    
    # Create figure with secondary y-axis
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Call OI (primary y-axis)
    fig3.add_trace(
        go.Scatter(
            x=time_series['timestamp'],
            y=time_series['call_oi'],
            name='CE (Call) OI',
            mode='lines+markers',
            line=dict(color=COLORS['call'], width=3),
            marker=dict(size=8),
            hovertemplate='%{x}<br>Call OI: %{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add Put OI (primary y-axis)
    fig3.add_trace(
        go.Scatter(
            x=time_series['timestamp'],
            y=time_series['put_oi'],
            name='PE (Put) OI',
            mode='lines+markers',
            line=dict(color=COLORS['put'], width=3),
            marker=dict(size=8),
            hovertemplate='%{x}<br>Put OI: %{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add Futures OI (secondary y-axis)
    fig3.add_trace(
        go.Scatter(
            x=time_series['timestamp'],
            y=time_series['futures_oi'],
            name='Futures OI',
            mode='lines+markers',
            line=dict(color=COLORS['future'], width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='%{x}<br>Futures OI: %{y:,.0f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig3.update_layout(
        title=f'{symbol} - Open Interest Change Over Time (CE, PE, Futures)',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )
    
    # Update axes labels
    fig3.update_xaxes(title_text="Time")
    fig3.update_yaxes(title_text="Options OI (CE/PE)", secondary_y=False)
    fig3.update_yaxes(title_text="Futures OI", secondary_y=True)
    
    # Show/hide historical date picker
    hist_date_style = {'flex': '1'} if live_mode == 'historical' else {'flex': '1', 'display': 'none'}
    
    return last_update, section1_info, fig1, fig2, fig3, hist_date_style


if __name__ == '__main__':
    print("=" * 80)
    print("ADVANCED OPTIONS TRADING DASHBOARD - 3 SECTIONS (FULLY FIXED)")
    print("=" * 80)
    print("")
    print("✅ FIXES APPLIED:")
    print("  • Timestamp synchronization between options and futures data")
    print("  • No more NaN values in Section 3")
    print("  • Futures OI now visible on secondary Y-axis")
    print("")
    print("Features:")
    print("  ✓ Section 1: Call vs Put Open Interest")
    print("  ✓ Section 2: Change in OI (Bar Chart)")
    print("  ✓ Section 3: Change in OI Over Time - CE/PE (left) + Futures (right)")
    print("")
    print("Controls:")
    print("  • Symbol Selection (NIFTY50/BANKNIFTY/FINNIFTY)")
    print("  • Expiry Date Selection")
    print("  • Interval (1Min/3Min/5Min/15Min/1H)")
    print("  • Range Filter (Enable/Disable)")
    print("  • Live/Historical Mode")
    print("  • Auto-refresh every 3 minutes")
    print("")
    print("Starting server...")
    print("Access at: http://127.0.0.1:8050")
    print("=" * 80)
    app.run(debug=True, host='127.0.0.1', port=8050)
