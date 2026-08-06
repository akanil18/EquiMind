/**
 * EquiMind Native — pybind11 Python Bindings
 *
 * Exposes C++ high-performance engines to Python:
 * - equimind_native.technical  → RSI, MACD, Bollinger, ATR, EMA, SMA
 * - equimind_native.montecarlo → GBM + Jump Diffusion simulation
 * - equimind_native.dedup      → FNV-1a hash, shingle, Jaccard dedup
 * - equimind_native.portfolio  → Risk Parity optimizer
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "equimind_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(equimind_native, m) {
    m.doc() = "EquiMind C++ Performance Core — optimized math for quant finance";

    // ── Technical Indicators ──────────────────────────────────
    py::module_ tech = m.def_submodule("technical", "C++ Technical Indicator Engine");

    py::class_<equimind::MACDResult>(tech, "MACDResult")
        .def_readonly("macd_line", &equimind::MACDResult::macd_line)
        .def_readonly("signal_line", &equimind::MACDResult::signal_line)
        .def_readonly("histogram", &equimind::MACDResult::histogram);

    py::class_<equimind::BollingerResult>(tech, "BollingerResult")
        .def_readonly("upper", &equimind::BollingerResult::upper)
        .def_readonly("middle", &equimind::BollingerResult::middle)
        .def_readonly("lower", &equimind::BollingerResult::lower)
        .def_readonly("bandwidth", &equimind::BollingerResult::bandwidth)
        .def_readonly("percent_b", &equimind::BollingerResult::percent_b);

    py::class_<equimind::TechnicalSummary>(tech, "TechnicalSummary")
        .def_readonly("rsi_14", &equimind::TechnicalSummary::rsi_14)
        .def_readonly("macd", &equimind::TechnicalSummary::macd)
        .def_readonly("bollinger", &equimind::TechnicalSummary::bollinger)
        .def_readonly("atr_14", &equimind::TechnicalSummary::atr_14)
        .def_readonly("sma_20", &equimind::TechnicalSummary::sma_20)
        .def_readonly("sma_50", &equimind::TechnicalSummary::sma_50)
        .def_readonly("sma_200", &equimind::TechnicalSummary::sma_200)
        .def_readonly("ema_12", &equimind::TechnicalSummary::ema_12)
        .def_readonly("ema_26", &equimind::TechnicalSummary::ema_26)
        .def_readonly("last_price", &equimind::TechnicalSummary::last_price)
        .def_readonly("annualized_volatility", &equimind::TechnicalSummary::annualized_volatility)
        .def_readonly("support_levels", &equimind::TechnicalSummary::support_levels)
        .def_readonly("resistance_levels", &equimind::TechnicalSummary::resistance_levels);

    tech.def("sma", &equimind::TechnicalEngine::sma,
             py::arg("prices"), py::arg("period"),
             "Simple Moving Average with O(n) sliding window");

    tech.def("ema", &equimind::TechnicalEngine::ema,
             py::arg("prices"), py::arg("period"),
             "Exponential Moving Average");

    tech.def("rsi", &equimind::TechnicalEngine::rsi,
             py::arg("prices"), py::arg("period") = 14,
             "RSI with Wilder's smoothing");

    tech.def("macd", &equimind::TechnicalEngine::macd,
             py::arg("prices"), py::arg("fast") = 12,
             py::arg("slow") = 26, py::arg("signal_period") = 9,
             "MACD (Moving Average Convergence Divergence)");

    tech.def("bollinger", &equimind::TechnicalEngine::bollinger,
             py::arg("prices"), py::arg("period") = 20,
             py::arg("num_std") = 2.0,
             "Bollinger Bands");

    tech.def("atr", &equimind::TechnicalEngine::atr,
             py::arg("high"), py::arg("low"), py::arg("close"),
             py::arg("period") = 14,
             "Average True Range with Wilder's smoothing");

    tech.def("annualized_volatility", &equimind::TechnicalEngine::annualized_volatility,
             py::arg("prices"), py::arg("trading_days") = 252,
             "Annualized volatility from daily close prices");

    tech.def("full_analysis", &equimind::TechnicalEngine::full_analysis,
             py::arg("close"), py::arg("high"), py::arg("low"),
             "Complete technical analysis in single call");

    // ── Monte Carlo Simulation ────────────────────────────────
    py::module_ mc = m.def_submodule("montecarlo", "C++ Monte Carlo Simulation Engine");

    py::class_<equimind::MonteCarloResult>(mc, "MonteCarloResult")
        .def_readonly("expected_price", &equimind::MonteCarloResult::expected_price)
        .def_readonly("p5", &equimind::MonteCarloResult::p5)
        .def_readonly("p25", &equimind::MonteCarloResult::p25)
        .def_readonly("median", &equimind::MonteCarloResult::median)
        .def_readonly("p75", &equimind::MonteCarloResult::p75)
        .def_readonly("p95", &equimind::MonteCarloResult::p95)
        .def_readonly("prob_above_current", &equimind::MonteCarloResult::prob_above_current)
        .def_readonly("final_prices", &equimind::MonteCarloResult::final_prices);

    mc.def("simulate", &equimind::MonteCarloEngine::simulate,
           py::arg("s0"), py::arg("mu"), py::arg("sigma"),
           py::arg("days") = 252, py::arg("num_paths") = 10000,
           py::arg("jump_intensity") = 0.0, py::arg("jump_mean") = 0.0,
           py::arg("jump_vol") = 0.0, py::arg("seed") = 42,
           "Monte Carlo simulation with GBM + optional Jump Diffusion");

    // ── Text Deduplication ────────────────────────────────────
    py::module_ dedup = m.def_submodule("dedup", "C++ Text Hashing & Deduplication");

    dedup.def("fnv1a_hash", &equimind::TextDedup::fnv1a_hash,
              py::arg("text"),
              "Fast 64-bit FNV-1a hash");

    dedup.def("deduplicate", &equimind::TextDedup::deduplicate,
              py::arg("texts"), py::arg("threshold") = 0.8,
              py::arg("shingle_size") = 5,
              "Batch fuzzy deduplication via Jaccard similarity on shingles");

    // ── Portfolio Optimization ────────────────────────────────
    py::module_ port = m.def_submodule("portfolio", "C++ Portfolio Optimization");

    port.def("risk_parity", &equimind::PortfolioOptimizer::risk_parity,
             py::arg("cov_matrix"), py::arg("max_iter") = 1000,
             py::arg("tol") = 1e-8,
             "Risk Parity (Equal Risk Contribution) portfolio weights");
}
