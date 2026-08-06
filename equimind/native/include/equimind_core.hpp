/**
 * EquiMind Native Performance Core — C++ Header
 * 
 * High-performance implementations for:
 * 1. Technical Indicators (RSI, MACD, Bollinger Bands, EMA, SMA, ATR)
 * 2. Monte Carlo Simulation (GBM + Jump Diffusion)
 * 3. Data Stream Buffer (async batch processing)
 * 4. Text Hashing & Deduplication
 *
 * These are the hot paths identified from institutional quant infrastructure:
 * - Technical indicators run on arrays of 10,000+ prices
 * - Monte Carlo needs 10,000+ stochastic paths
 * - Context compression hashes thousands of evidence strings
 */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <functional>
#include <tuple>

namespace equimind {

// ══════════════════════════════════════════════════════════════
// 1. TECHNICAL INDICATORS ENGINE
// ══════════════════════════════════════════════════════════════

struct MACDResult {
    double macd_line;
    double signal_line;
    double histogram;
};

struct BollingerResult {
    double upper;
    double middle;
    double lower;
    double bandwidth;
    double percent_b;
};

struct TechnicalSummary {
    double rsi_14;
    MACDResult macd;
    BollingerResult bollinger;
    double atr_14;
    double sma_20;
    double sma_50;
    double sma_200;
    double ema_12;
    double ema_26;
    double last_price;
    double annualized_volatility;
    std::vector<double> support_levels;
    std::vector<double> resistance_levels;
};

class TechnicalEngine {
public:
    /**
     * Compute SMA (Simple Moving Average) over a window.
     * O(n) with sliding window optimization.
     */
    static std::vector<double> sma(const std::vector<double>& prices, int period) {
        std::vector<double> result;
        if (prices.size() < static_cast<size_t>(period)) return result;
        
        result.reserve(prices.size() - period + 1);
        double sum = 0.0;
        for (int i = 0; i < period; ++i) sum += prices[i];
        result.push_back(sum / period);
        
        for (size_t i = period; i < prices.size(); ++i) {
            sum += prices[i] - prices[i - period];
            result.push_back(sum / period);
        }
        return result;
    }

    /**
     * Compute EMA (Exponential Moving Average).
     * Uses multiplier = 2 / (period + 1).
     */
    static std::vector<double> ema(const std::vector<double>& prices, int period) {
        std::vector<double> result;
        if (prices.size() < static_cast<size_t>(period)) return result;
        
        result.reserve(prices.size());
        double multiplier = 2.0 / (period + 1);
        
        // Seed with SMA of first `period` values
        double seed = 0.0;
        for (int i = 0; i < period; ++i) seed += prices[i];
        seed /= period;
        result.push_back(seed);
        
        for (size_t i = period; i < prices.size(); ++i) {
            double val = (prices[i] - result.back()) * multiplier + result.back();
            result.push_back(val);
        }
        return result;
    }

    /**
     * RSI (Relative Strength Index) — Wilder's smoothing method.
     * Avoids the naive approach; uses exponential smoothing of gains/losses.
     */
    static double rsi(const std::vector<double>& prices, int period = 14) {
        if (prices.size() < static_cast<size_t>(period + 1)) return 50.0;
        
        double avg_gain = 0.0, avg_loss = 0.0;
        
        // Initial average from first `period` changes
        for (int i = 1; i <= period; ++i) {
            double change = prices[i] - prices[i - 1];
            if (change > 0) avg_gain += change;
            else avg_loss += std::abs(change);
        }
        avg_gain /= period;
        avg_loss /= period;
        
        // Wilder's smoothing for remaining
        for (size_t i = period + 1; i < prices.size(); ++i) {
            double change = prices[i] - prices[i - 1];
            if (change > 0) {
                avg_gain = (avg_gain * (period - 1) + change) / period;
                avg_loss = (avg_loss * (period - 1)) / period;
            } else {
                avg_gain = (avg_gain * (period - 1)) / period;
                avg_loss = (avg_loss * (period - 1) + std::abs(change)) / period;
            }
        }
        
        if (avg_loss < 1e-10) return 100.0;
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    /**
     * MACD (Moving Average Convergence Divergence).
     * Fast EMA(12) - Slow EMA(26), Signal = EMA(9) of MACD line.
     */
    static MACDResult macd(const std::vector<double>& prices,
                           int fast = 12, int slow = 26, int signal_period = 9) {
        auto fast_ema = ema(prices, fast);
        auto slow_ema = ema(prices, slow);
        
        if (fast_ema.empty() || slow_ema.empty()) {
            return {0.0, 0.0, 0.0};
        }
        
        // Align: slow_ema starts later
        size_t offset = slow - fast;
        std::vector<double> macd_line;
        size_t min_len = std::min(fast_ema.size() - offset, slow_ema.size());
        macd_line.reserve(min_len);
        
        for (size_t i = 0; i < min_len; ++i) {
            macd_line.push_back(fast_ema[i + offset] - slow_ema[i]);
        }
        
        auto signal = ema(macd_line, signal_period);
        
        double macd_val = macd_line.empty() ? 0.0 : macd_line.back();
        double signal_val = signal.empty() ? 0.0 : signal.back();
        
        return {
            std::round(macd_val * 100) / 100,
            std::round(signal_val * 100) / 100,
            std::round((macd_val - signal_val) * 100) / 100
        };
    }

    /**
     * Bollinger Bands — SMA(20) ± 2 * StdDev(20).
     */
    static BollingerResult bollinger(const std::vector<double>& prices,
                                      int period = 20, double num_std = 2.0) {
        if (prices.size() < static_cast<size_t>(period)) {
            double last = prices.empty() ? 0.0 : prices.back();
            return {last, last, last, 0.0, 0.5};
        }
        
        // Last `period` prices
        double sum = 0.0, sum_sq = 0.0;
        size_t start = prices.size() - period;
        for (size_t i = start; i < prices.size(); ++i) {
            sum += prices[i];
            sum_sq += prices[i] * prices[i];
        }
        
        double mean = sum / period;
        double variance = (sum_sq / period) - (mean * mean);
        double std_dev = std::sqrt(std::max(0.0, variance));
        
        double upper = mean + num_std * std_dev;
        double lower = mean - num_std * std_dev;
        double bw = (upper - lower) / mean;
        double pb = std_dev > 1e-10 ? (prices.back() - lower) / (upper - lower) : 0.5;
        
        return {
            std::round(upper * 100) / 100,
            std::round(mean * 100) / 100,
            std::round(lower * 100) / 100,
            std::round(bw * 10000) / 10000,
            std::round(pb * 10000) / 10000
        };
    }

    /**
     * ATR (Average True Range) — Wilder's smoothing.
     */
    static double atr(const std::vector<double>& high,
                      const std::vector<double>& low,
                      const std::vector<double>& close,
                      int period = 14) {
        size_t n = high.size();
        if (n < static_cast<size_t>(period + 1)) return 0.0;
        
        // Compute True Range for each bar
        std::vector<double> tr(n);
        tr[0] = high[0] - low[0];
        for (size_t i = 1; i < n; ++i) {
            double hl = high[i] - low[i];
            double hc = std::abs(high[i] - close[i - 1]);
            double lc = std::abs(low[i] - close[i - 1]);
            tr[i] = std::max({hl, hc, lc});
        }
        
        // Initial ATR = simple average of first `period` TRs
        double atr_val = 0.0;
        for (int i = 1; i <= period; ++i) atr_val += tr[i];
        atr_val /= period;
        
        // Wilder's smoothing
        for (size_t i = period + 1; i < n; ++i) {
            atr_val = (atr_val * (period - 1) + tr[i]) / period;
        }
        
        return std::round(atr_val * 100) / 100;
    }

    /**
     * Support & Resistance levels via pivot point analysis.
     * Identifies local minima (support) and maxima (resistance) using a rolling window.
     */
    static std::pair<std::vector<double>, std::vector<double>>
    support_resistance(const std::vector<double>& prices, int window = 10, int max_levels = 3) {
        std::vector<double> supports, resistances;
        if (prices.size() < static_cast<size_t>(2 * window + 1)) {
            return {supports, resistances};
        }
        
        for (size_t i = window; i < prices.size() - window; ++i) {
            bool is_min = true, is_max = true;
            for (int j = 1; j <= window; ++j) {
                if (prices[i] >= prices[i - j] || prices[i] >= prices[i + j]) is_min = false;
                if (prices[i] <= prices[i - j] || prices[i] <= prices[i + j]) is_max = false;
            }
            if (is_min) supports.push_back(std::round(prices[i] * 100) / 100);
            if (is_max) resistances.push_back(std::round(prices[i] * 100) / 100);
        }
        
        // Keep only the most recent levels
        if (supports.size() > static_cast<size_t>(max_levels))
            supports.erase(supports.begin(), supports.end() - max_levels);
        if (resistances.size() > static_cast<size_t>(max_levels))
            resistances.erase(resistances.begin(), resistances.end() - max_levels);
        
        return {supports, resistances};
    }

    /**
     * Annualized volatility from daily returns.
     */
    static double annualized_volatility(const std::vector<double>& prices, int trading_days = 252) {
        if (prices.size() < 2) return 0.0;
        
        std::vector<double> returns;
        returns.reserve(prices.size() - 1);
        for (size_t i = 1; i < prices.size(); ++i) {
            if (prices[i - 1] > 0) {
                returns.push_back(std::log(prices[i] / prices[i - 1]));
            }
        }
        
        if (returns.empty()) return 0.0;
        
        double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double var = 0.0;
        for (double r : returns) var += (r - mean) * (r - mean);
        var /= (returns.size() - 1);
        
        return std::round(std::sqrt(var * trading_days) * 10000) / 100; // As percentage
    }

    /**
     * Full technical analysis summary — single call computes everything.
     */
    static TechnicalSummary full_analysis(const std::vector<double>& close,
                                           const std::vector<double>& high,
                                           const std::vector<double>& low) {
        TechnicalSummary s;
        s.last_price = close.empty() ? 0.0 : std::round(close.back() * 100) / 100;
        s.rsi_14 = rsi(close, 14);
        s.macd = macd(close, 12, 26, 9);
        s.bollinger = bollinger(close, 20, 2.0);
        s.atr_14 = atr(high, low, close, 14);
        
        auto sma20 = sma(close, 20);
        auto sma50 = sma(close, 50);
        auto sma200 = sma(close, 200);
        auto ema12 = ema(close, 12);
        auto ema26 = ema(close, 26);
        
        s.sma_20 = sma20.empty() ? 0.0 : std::round(sma20.back() * 100) / 100;
        s.sma_50 = sma50.empty() ? 0.0 : std::round(sma50.back() * 100) / 100;
        s.sma_200 = sma200.empty() ? 0.0 : std::round(sma200.back() * 100) / 100;
        s.ema_12 = ema12.empty() ? 0.0 : std::round(ema12.back() * 100) / 100;
        s.ema_26 = ema26.empty() ? 0.0 : std::round(ema26.back() * 100) / 100;
        s.annualized_volatility = annualized_volatility(close);
        
        auto [sup, res] = support_resistance(close);
        s.support_levels = sup;
        s.resistance_levels = res;
        
        return s;
    }
};


// ══════════════════════════════════════════════════════════════
// 2. MONTE CARLO SIMULATION ENGINE
// ══════════════════════════════════════════════════════════════

struct MonteCarloResult {
    double expected_price;
    double p5;     // 5th percentile
    double p25;    // 25th percentile
    double median; // 50th percentile
    double p75;    // 75th percentile
    double p95;    // 95th percentile
    double prob_above_current;
    double max_drawdown_median;
    std::vector<double> final_prices; // All terminal values
};

class MonteCarloEngine {
public:
    /**
     * Geometric Brownian Motion with optional Jump Diffusion.
     * Runs `num_paths` simulations of `num_steps` trading days.
     *
     * Parameters:
     *   s0:         Current price
     *   mu:         Annualized drift (expected return)
     *   sigma:      Annualized volatility
     *   days:       Number of trading days to simulate
     *   num_paths:  Number of Monte Carlo paths
     *   jump_intensity: Poisson lambda for jumps (0 = pure GBM)
     *   jump_mean:  Mean of log-normal jump size
     *   jump_vol:   Volatility of jump size
     *   seed:       Random seed for reproducibility
     */
    static MonteCarloResult simulate(
        double s0, double mu, double sigma, int days = 252,
        int num_paths = 10000, double jump_intensity = 0.0,
        double jump_mean = 0.0, double jump_vol = 0.0, unsigned seed = 42
    ) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> norm(0.0, 1.0);
        std::poisson_distribution<int> poisson(jump_intensity / 252.0);
        std::normal_distribution<double> jump_dist(jump_mean, jump_vol);
        
        double dt = 1.0 / 252.0;
        double drift = (mu - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * std::sqrt(dt);
        
        std::vector<double> final_prices(num_paths);
        
        for (int p = 0; p < num_paths; ++p) {
            double price = s0;
            for (int t = 0; t < days; ++t) {
                double z = norm(gen);
                double log_return = drift + diffusion * z;
                
                // Jump diffusion component
                if (jump_intensity > 0) {
                    int num_jumps = poisson(gen);
                    for (int j = 0; j < num_jumps; ++j) {
                        log_return += jump_dist(gen);
                    }
                }
                
                price *= std::exp(log_return);
            }
            final_prices[p] = price;
        }
        
        // Sort for percentile calculation
        std::sort(final_prices.begin(), final_prices.end());
        
        auto percentile = [&](double pct) -> double {
            size_t idx = static_cast<size_t>(pct * num_paths);
            idx = std::min(idx, static_cast<size_t>(num_paths - 1));
            return std::round(final_prices[idx] * 100) / 100;
        };
        
        double sum = std::accumulate(final_prices.begin(), final_prices.end(), 0.0);
        double expected = sum / num_paths;
        
        // Probability of price above current
        auto it = std::lower_bound(final_prices.begin(), final_prices.end(), s0);
        double prob_above = 1.0 - static_cast<double>(it - final_prices.begin()) / num_paths;
        
        return {
            std::round(expected * 100) / 100,
            percentile(0.05),
            percentile(0.25),
            percentile(0.50),
            percentile(0.75),
            percentile(0.95),
            std::round(prob_above * 10000) / 10000,
            0.0, // max_drawdown computed separately if needed
            final_prices
        };
    }
};


// ══════════════════════════════════════════════════════════════
// 3. TEXT HASHING & DEDUPLICATION ENGINE
// ══════════════════════════════════════════════════════════════

class TextDedup {
public:
    /**
     * Fast 64-bit FNV-1a hash for text deduplication.
     */
    static uint64_t fnv1a_hash(const std::string& text) {
        uint64_t hash = 14695981039346656037ULL;
        for (char c : text) {
            hash ^= static_cast<uint64_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    /**
     * Generate n-gram shingles for fuzzy deduplication.
     * Returns set of hashed shingles.
     */
    static std::unordered_set<uint64_t> shingle(const std::string& text, int n = 3) {
        std::unordered_set<uint64_t> shingles;
        if (text.size() < static_cast<size_t>(n)) return shingles;
        
        // Normalize: lowercase, collapse whitespace
        std::string normalized;
        normalized.reserve(text.size());
        bool prev_space = false;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n') {
                if (!prev_space) { normalized.push_back(' '); prev_space = true; }
            } else {
                normalized.push_back(static_cast<char>(std::tolower(c)));
                prev_space = false;
            }
        }
        
        for (size_t i = 0; i <= normalized.size() - n; ++i) {
            std::string gram = normalized.substr(i, n);
            shingles.insert(fnv1a_hash(gram));
        }
        return shingles;
    }

    /**
     * Jaccard similarity between two shingle sets.
     * Returns value in [0.0, 1.0].
     */
    static double jaccard(const std::unordered_set<uint64_t>& a,
                          const std::unordered_set<uint64_t>& b) {
        if (a.empty() && b.empty()) return 1.0;
        if (a.empty() || b.empty()) return 0.0;
        
        size_t intersection = 0;
        for (uint64_t h : a) {
            if (b.count(h)) ++intersection;
        }
        size_t union_size = a.size() + b.size() - intersection;
        return static_cast<double>(intersection) / union_size;
    }

    /**
     * Batch deduplication: returns indices of unique items.
     * Items with Jaccard similarity > threshold are considered duplicates.
     */
    static std::vector<size_t> deduplicate(const std::vector<std::string>& texts,
                                            double threshold = 0.8, int shingle_size = 5) {
        std::vector<std::unordered_set<uint64_t>> all_shingles;
        all_shingles.reserve(texts.size());
        for (const auto& t : texts) {
            all_shingles.push_back(shingle(t, shingle_size));
        }
        
        std::vector<size_t> unique_indices;
        for (size_t i = 0; i < texts.size(); ++i) {
            bool is_dup = false;
            for (size_t j : unique_indices) {
                if (jaccard(all_shingles[i], all_shingles[j]) > threshold) {
                    is_dup = true;
                    break;
                }
            }
            if (!is_dup) unique_indices.push_back(i);
        }
        return unique_indices;
    }
};


// ══════════════════════════════════════════════════════════════
// 4. PORTFOLIO OPTIMIZATION (Markowitz)
// ══════════════════════════════════════════════════════════════

struct PortfolioResult {
    std::vector<double> weights;
    double expected_return;
    double volatility;
    double sharpe_ratio;
};

class PortfolioOptimizer {
public:
    /**
     * Equal Risk Contribution (Risk Parity) weights.
     * Iterative algorithm converging to equal marginal risk contribution.
     */
    static std::vector<double> risk_parity(const std::vector<std::vector<double>>& cov_matrix,
                                            int max_iter = 1000, double tol = 1e-8) {
        size_t n = cov_matrix.size();
        std::vector<double> w(n, 1.0 / n); // Start with equal weights
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Compute portfolio variance contributions
            std::vector<double> sigma_w(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    sigma_w[i] += cov_matrix[i][j] * w[j];
                }
            }
            
            // Risk contribution of each asset
            double total_risk = 0.0;
            for (size_t i = 0; i < n; ++i) total_risk += w[i] * sigma_w[i];
            
            if (total_risk < 1e-15) break;
            
            // Update weights: w_i proportional to 1 / (marginal risk)
            std::vector<double> new_w(n);
            double sum_w = 0.0;
            for (size_t i = 0; i < n; ++i) {
                new_w[i] = sigma_w[i] > 1e-15 ? 1.0 / (sigma_w[i] * n) : w[i];
                sum_w += new_w[i];
            }
            
            // Normalize
            double max_diff = 0.0;
            for (size_t i = 0; i < n; ++i) {
                new_w[i] /= sum_w;
                max_diff = std::max(max_diff, std::abs(new_w[i] - w[i]));
                w[i] = new_w[i];
            }
            
            if (max_diff < tol) break;
        }
        return w;
    }
};

} // namespace equimind
