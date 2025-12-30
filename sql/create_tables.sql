-- ============================================================
-- Financial Portfolio Analytics - Database Schema
-- PostgreSQL / MySQL compatible
-- ============================================================

-- Drop tables if exist (for clean setup)
DROP TABLE IF EXISTS portfolio_performance CASCADE;
DROP TABLE IF EXISTS portfolio_holdings CASCADE;
DROP TABLE IF EXISTS portfolios CASCADE;
DROP TABLE IF EXISTS asset_returns CASCADE;
DROP TABLE IF EXISTS asset_prices CASCADE;
DROP TABLE IF EXISTS assets CASCADE;
DROP TABLE IF EXISTS risk_metrics CASCADE;

-- ============================================================
-- ASSET MASTER TABLE
-- ============================================================
CREATE TABLE assets (
    asset_id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) UNIQUE NOT NULL,
    asset_name VARCHAR(200) NOT NULL,
    asset_class VARCHAR(50) NOT NULL,  -- Equity, Bond, Gold, etc.
    sector VARCHAR(100),
    exchange VARCHAR(20) DEFAULT 'NSE',
    currency VARCHAR(3) DEFAULT 'INR',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ticker ON assets(ticker);
CREATE INDEX idx_asset_class ON assets(asset_class);

-- Sample data
INSERT INTO assets (ticker, asset_name, asset_class, sector) VALUES
    ('NIFTYBEES.NS', 'Nippon India ETF Nifty BeES', 'Equity', 'Large Cap'),
    ('JUNIORBEES.NS', 'Nippon India ETF Junior BeES', 'Equity', 'Mid Cap'),
    ('BANKBEES.NS', 'Nippon India ETF Bank BeES', 'Equity', 'Banking'),
    ('GOLDBEES.NS', 'Nippon India ETF Gold BeES', 'Commodity', 'Gold'),
    ('ITBEES.NS', 'Nippon India ETF IT BeES', 'Equity', 'Technology');

-- ============================================================
-- PRICE DATA TABLE
-- ============================================================
CREATE TABLE asset_prices (
    price_id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES assets(asset_id),
    price_date DATE NOT NULL,
    open_price DECIMAL(18, 4),
    high_price DECIMAL(18, 4),
    low_price DECIMAL(18, 4),
    close_price DECIMAL(18, 4),
    adjusted_close DECIMAL(18, 4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_id, price_date)
);

CREATE INDEX idx_asset_price_date ON asset_prices(asset_id, price_date);
CREATE INDEX idx_price_date ON asset_prices(price_date);

-- ============================================================
-- RETURNS DATA TABLE
-- ============================================================
CREATE TABLE asset_returns (
    return_id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES assets(asset_id),
    return_date DATE NOT NULL,
    daily_return DECIMAL(18, 8) NOT NULL,
    log_return DECIMAL(18, 8),
    cumulative_return DECIMAL(18, 8),
    UNIQUE(asset_id, return_date)
);

CREATE INDEX idx_asset_return_date ON asset_returns(asset_id, return_date);

-- ============================================================
-- PORTFOLIO MASTER TABLE
-- ============================================================
CREATE TABLE portfolios (
    portfolio_id SERIAL PRIMARY KEY,
    portfolio_name VARCHAR(100) UNIQUE NOT NULL,
    portfolio_type VARCHAR(50) NOT NULL,  -- Conservative, Balanced, Aggressive
    optimization_method VARCHAR(50),       -- Equal Weight, Min Vol, Max Sharpe
    initial_capital DECIMAL(18, 2) NOT NULL,
    creation_date DATE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample portfolios
INSERT INTO portfolios (portfolio_name, portfolio_type, optimization_method, initial_capital, creation_date) VALUES
    ('Balanced - Equal Weight', 'Balanced', 'Equal Weight', 1000000, '2015-01-01'),
    ('Balanced - Min Vol', 'Balanced', 'Minimum Volatility', 1000000, '2015-01-01'),
    ('Balanced - Max Sharpe', 'Balanced', 'Maximum Sharpe', 1000000, '2015-01-01');

-- ============================================================
-- PORTFOLIO HOLDINGS TABLE
-- ============================================================
CREATE TABLE portfolio_holdings (
    holding_id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(portfolio_id),
    asset_id INTEGER REFERENCES assets(asset_id),
    weight DECIMAL(10, 8) NOT NULL,  -- Portfolio weight (0-1)
    shares DECIMAL(18, 4),            -- Number of shares held
    allocation_date DATE NOT NULL,
    is_current BOOLEAN DEFAULT TRUE,
    UNIQUE(portfolio_id, asset_id, allocation_date)
);

CREATE INDEX idx_portfolio_holdings ON portfolio_holdings(portfolio_id, is_current);

-- ============================================================
-- PORTFOLIO PERFORMANCE TABLE
-- ============================================================
CREATE TABLE portfolio_performance (
    performance_id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(portfolio_id),
    performance_date DATE NOT NULL,
    portfolio_value DECIMAL(18, 2) NOT NULL,
    daily_return DECIMAL(18, 8),
    cumulative_return DECIMAL(18, 8),
    benchmark_value DECIMAL(18, 2),
    benchmark_return DECIMAL(18, 8),
    alpha DECIMAL(18, 8),
    beta DECIMAL(18, 8),
    UNIQUE(portfolio_id, performance_date)
);

CREATE INDEX idx_portfolio_performance_date ON portfolio_performance(portfolio_id, performance_date);

-- ============================================================
-- RISK METRICS TABLE
-- ============================================================
CREATE TABLE risk_metrics (
    metric_id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(portfolio_id),
    calculation_date DATE NOT NULL,
    period_start_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    
    -- Return metrics
    total_return DECIMAL(18, 4),
    cagr DECIMAL(18, 4),
    
    -- Risk metrics
    volatility DECIMAL(18, 4),
    max_drawdown DECIMAL(18, 4),
    
    -- Risk-adjusted metrics
    sharpe_ratio DECIMAL(18, 6),
    sortino_ratio DECIMAL(18, 6),
    calmar_ratio DECIMAL(18, 6),
    
    -- Tail risk
    var_95 DECIMAL(18, 6),
    cvar_95 DECIMAL(18, 6),
    var_99 DECIMAL(18, 6),
    cvar_99 DECIMAL(18, 6),
    
    -- Distribution metrics
    skewness DECIMAL(18, 6),
    kurtosis DECIMAL(18, 6),
    
    -- Benchmark comparison
    alpha DECIMAL(18, 6),
    beta DECIMAL(18, 6),
    tracking_error DECIMAL(18, 6),
    information_ratio DECIMAL(18, 6),
    upside_capture DECIMAL(18, 6),
    downside_capture DECIMAL(18, 6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_metrics_portfolio ON risk_metrics(portfolio_id, calculation_date);

-- ============================================================
-- USEFUL VIEWS
-- ============================================================

-- Current portfolio holdings view
CREATE VIEW v_current_holdings AS
SELECT 
    p.portfolio_name,
    p.portfolio_type,
    a.ticker,
    a.asset_name,
    a.asset_class,
    h.weight,
    h.shares,
    h.allocation_date
FROM portfolio_holdings h
JOIN portfolios p ON h.portfolio_id = p.portfolio_id
JOIN assets a ON h.asset_id = a.asset_id
WHERE h.is_current = TRUE
ORDER BY p.portfolio_name, h.weight DESC;

-- Latest risk metrics view
CREATE VIEW v_latest_risk_metrics AS
SELECT 
    p.portfolio_name,
    p.portfolio_type,
    r.calculation_date,
    r.cagr,
    r.volatility,
    r.sharpe_ratio,
    r.max_drawdown,
    r.var_95,
    r.alpha,
    r.beta
FROM risk_metrics r
JOIN portfolios p ON r.portfolio_id = p.portfolio_id
WHERE r.calculation_date = (
    SELECT MAX(calculation_date) 
    FROM risk_metrics 
    WHERE portfolio_id = r.portfolio_id
);

-- Portfolio performance summary view
CREATE VIEW v_portfolio_summary AS
SELECT 
    p.portfolio_name,
    p.portfolio_type,
    p.initial_capital,
    pp.portfolio_value AS current_value,
    (pp.portfolio_value - p.initial_capital) AS profit_loss,
    ((pp.portfolio_value / p.initial_capital) - 1) * 100 AS return_pct,
    pp.performance_date AS last_updated
FROM portfolios p
JOIN portfolio_performance pp ON p.portfolio_id = pp.portfolio_id
WHERE pp.performance_date = (
    SELECT MAX(performance_date) 
    FROM portfolio_performance 
    WHERE portfolio_id = pp.portfolio_id
);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function to calculate daily returns
CREATE OR REPLACE FUNCTION calculate_daily_returns()
RETURNS VOID AS $$
BEGIN
    TRUNCATE TABLE asset_returns;
    
    INSERT INTO asset_returns (asset_id, return_date, daily_return)
    SELECT 
        curr.asset_id,
        curr.price_date,
        (curr.adjusted_close / prev.adjusted_close - 1) AS daily_return
    FROM asset_prices curr
    JOIN asset_prices prev ON 
        curr.asset_id = prev.asset_id 
        AND prev.price_date = (
            SELECT MAX(price_date) 
            FROM asset_prices 
            WHERE asset_id = curr.asset_id 
            AND price_date < curr.price_date
        );
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- SAMPLE QUERIES (for testing)
-- ============================================================

-- Check data completeness
-- SELECT 
--     a.ticker,
--     COUNT(DISTINCT ap.price_date) as days_of_data,
--     MIN(ap.price_date) as first_date,
--     MAX(ap.price_date) as last_date
-- FROM assets a
-- LEFT JOIN asset_prices ap ON a.asset_id = ap.asset_id
-- GROUP BY a.ticker;

-- Portfolio allocation by asset class
-- SELECT 
--     p.portfolio_name,
--     a.asset_class,
--     SUM(h.weight) * 100 as allocation_pct
-- FROM portfolio_holdings h
-- JOIN portfolios p ON h.portfolio_id = p.portfolio_id
-- JOIN assets a ON h.asset_id = a.asset_id
-- WHERE h.is_current = TRUE
-- GROUP BY p.portfolio_name, a.asset_class
-- ORDER BY p.portfolio_name, allocation_pct DESC;

-- Compare portfolio returns
-- SELECT 
--     portfolio_name,
--     cagr,
--     volatility,
--     sharpe_ratio,
--     max_drawdown
-- FROM v_latest_risk_metrics
-- ORDER BY sharpe_ratio DESC;