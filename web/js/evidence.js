/**
 * EquiMind Evidence Explorer — evidence.js
 * Manages evidence node data, search indexing, multi-criteria filtering, and card rendering
 */

const SAMPLE_EVIDENCE = [
  {
    id: "ev-01",
    ticker: "NVDA",
    sourceType: "sec",
    sourceName: "SEC EDGAR Form 10-Q",
    title: "NVIDIA Quarterly Report (Q2 FY2025) — Data Center Revenue +154% YoY",
    content: "Data Center compute revenue surged to $22.6 billion, driven by intense demand for the Hopper GPU architecture (H100/H200). Gross margin expanded to 75.1%. Customer concentration: top 4 cloud service providers represent 45% of total revenue.",
    credibility: "VERIFIED_OFFICIAL",
    credibilityLabel: "Official SEC",
    confidence: 0.98,
    sentiment: "BULLISH",
    timestamp: "2025-08-28 14:30:00",
    url: "https://www.sec.gov/edgar/searchedgar/companysearch",
    citationsCount: 14
  },
  {
    id: "ev-02",
    ticker: "NVDA",
    sourceType: "github",
    sourceName: "GitHub Repository (NVIDIA/CUDA)",
    title: "CUDA Ecosystem Commit Velocity & TensorRT-LLM Release 0.12.0",
    content: "Substantial acceleration in developer repository commits (+892 commits this month). TensorRT-LLM 0.12 introduces FP4 quantization support for Blackwell GPUs, improving inference throughput per watt by 3.2x over H100.",
    credibility: "HIGH",
    credibilityLabel: "Code Repo",
    confidence: 0.91,
    sentiment: "BULLISH",
    timestamp: "2025-08-29 09:15:20",
    url: "https://github.com/NVIDIA/TensorRT-LLM",
    citationsCount: 9
  },
  {
    id: "ev-03",
    ticker: "NVDA",
    sourceType: "news",
    sourceName: "Bloomberg Terminal",
    title: "TSMC N3P Yield Rates Reach 82% for Blackwell B200 Production",
    content: "Taiwan Semiconductor Manufacturing Co. (TSMC) has resolved early substrate warpage issues on CoWoS-L packaging. B200 chip shipments scheduled for full scale ramping in Q4. Hyperscalers Microsoft and Meta have secured 60% of initial allocation.",
    credibility: "HIGH",
    credibilityLabel: "Bloomberg",
    confidence: 0.88,
    sentiment: "BULLISH",
    timestamp: "2025-08-30 11:45:00",
    url: "https://www.bloomberg.com/technology",
    citationsCount: 12
  },
  {
    id: "ev-04",
    ticker: "NVDA",
    sourceType: "reddit",
    sourceName: "r/stocks & r/LocalLLaMA",
    title: "Enterprise AI Infrastructure Survey — CUDA Moat Intact",
    content: "Survey of 120 enterprise machine learning engineers shows 84% preferred NVIDIA GPUs for training due to software stack maturity. However, 32% are actively evaluating AMD ROCm 6.1 for inference workloads due to hardware cost differences.",
    credibility: "MEDIUM",
    credibilityLabel: "Community",
    confidence: 0.72,
    sentiment: "NEUTRAL",
    timestamp: "2025-08-31 16:20:00",
    url: "https://www.reddit.com/r/LocalLLaMA",
    citationsCount: 5
  },
  {
    id: "ev-05",
    ticker: "NVDA",
    sourceType: "macro",
    sourceName: "Federal Reserve Economic Data (FRED)",
    title: "U.S. Enterprise Technology Capex Spending Trends",
    content: "Information processing equipment investment increased at an annual rate of 12.4% in Q2. High interest rates (5.25%) have not dampened AI capital expenditure among fortune 500 enterprises, though non-AI IT budgets remain flat.",
    credibility: "VERIFIED_OFFICIAL",
    credibilityLabel: "FRED Macro",
    confidence: 0.95,
    sentiment: "BULLISH",
    timestamp: "2025-08-25 08:00:00",
    url: "https://fred.stlouisfed.org",
    citationsCount: 8
  },
  {
    id: "ev-06",
    ticker: "AAPL",
    sourceType: "sec",
    sourceName: "SEC EDGAR Form 10-Q",
    title: "Apple Inc. Quarterly Filing — India Sales Growth +28%",
    content: "iPhone revenue flat in Greater China (-3.2% YoY), offset by accelerated growth in India, Indonesia, and Latin America. Services business reached all-time high gross margin of 74.0%. R&D expenses increased to $7.8 billion for Apple Intelligence deployment.",
    credibility: "VERIFIED_OFFICIAL",
    credibilityLabel: "Official SEC",
    confidence: 0.99,
    sentiment: "NEUTRAL",
    timestamp: "2025-08-02 17:00:00",
    url: "https://www.sec.gov/edgar",
    citationsCount: 11
  },
  {
    id: "ev-07",
    ticker: "TSLA",
    sourceType: "news",
    sourceName: "Reuters Financial News",
    title: "BYD Expands Europe EV Market Share to 8.4% amid Price War",
    content: "BYD launched 3 new low-cost EV models in Germany and UK. Tesla Automotive gross margin (ex-regulatory credits) compressed to 14.6% in Q2. Full Self-Driving (FSD) V12.5 subscription uptake remains below 12% in North America.",
    credibility: "HIGH",
    credibilityLabel: "Reuters",
    confidence: 0.85,
    sentiment: "BEARISH",
    timestamp: "2025-08-14 13:10:00",
    url: "https://www.reuters.com/business/autos-transportation",
    citationsCount: 7
  },
  {
    id: "ev-08",
    ticker: "MSFT",
    sourceType: "sec",
    sourceName: "SEC EDGAR Form 10-K",
    title: "Microsoft Cloud Revenue Surpasses $36.8B quarterly rate",
    content: "Azure and other cloud services growth of 29% YoY, including 8 percentage points contributed by AI services. Capital expenditure totaled $19.0 billion in Q4, primarily for data center land, infrastructure, and custom Maia AI chips.",
    credibility: "VERIFIED_OFFICIAL",
    credibilityLabel: "Official SEC",
    confidence: 0.97,
    sentiment: "BULLISH",
    timestamp: "2025-07-31 16:30:00",
    url: "https://www.sec.gov/edgar",
    citationsCount: 15
  }
];

class EvidenceExplorer {
  constructor() {
    this.evidence = [...SAMPLE_EVIDENCE];
    this.filtered = [...SAMPLE_EVIDENCE];
    
    this.searchQuery = "";
    this.selectedSource = "all";
    this.selectedCredibility = "all";
    this.selectedSentiment = "all";
    this.minConfidence = 0.50;

    this.init();
  }

  init() {
    this.bindEvents();
    this.render();
  }

  bindEvents() {
    const searchInput = document.getElementById("evidence-search");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        this.searchQuery = e.target.value.toLowerCase();
        this.filterAndRender();
      });
    }

    // Source Filter Chips
    document.querySelectorAll("[data-filter-source]").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("[data-filter-source]").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        this.selectedSource = chip.getAttribute("data-filter-source");
        this.filterAndRender();
      });
    });

    // Credibility Filter Chips
    document.querySelectorAll("[data-filter-credibility]").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("[data-filter-credibility]").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        this.selectedCredibility = chip.getAttribute("data-filter-credibility");
        this.filterAndRender();
      });
    });

    // Sentiment Filter Chips
    document.querySelectorAll("[data-filter-sentiment]").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("[data-filter-sentiment]").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        this.selectedSentiment = chip.getAttribute("data-filter-sentiment");
        this.filterAndRender();
      });
    });

    // Confidence Slider
    const slider = document.getElementById("confidence-slider");
    const sliderVal = document.getElementById("confidence-slider-val");
    if (slider) {
      slider.addEventListener("input", (e) => {
        this.minConfidence = parseFloat(e.target.value);
        if (sliderVal) sliderVal.textContent = Math.round(this.minConfidence * 100) + "%";
        this.filterAndRender();
      });
    }
  }

  filterAndRender() {
    this.filtered = this.evidence.filter(item => {
      // Search match
      const text = `${item.ticker} ${item.title} ${item.content} ${item.sourceName}`.toLowerCase();
      if (this.searchQuery && !text.includes(this.searchQuery)) return false;

      // Source filter
      if (this.selectedSource !== "all" && item.sourceType !== this.selectedSource) return false;

      // Credibility filter
      if (this.selectedCredibility !== "all" && item.credibility !== this.selectedCredibility) return false;

      // Sentiment filter
      if (this.selectedSentiment !== "all" && item.sentiment !== this.selectedSentiment) return false;

      // Confidence threshold
      if (item.confidence < this.minConfidence) return false;

      return true;
    });

    this.render();
  }

  render() {
    const grid = document.getElementById("evidence-grid");
    const countEl = document.getElementById("evidence-count");
    if (countEl) countEl.textContent = this.filtered.length;

    if (!grid) return;

    if (this.filtered.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1">
          <div class="empty-icon">🔍</div>
          <div class="t-h3">No matching evidence found</div>
          <div class="t-small t-muted">Try adjusting your filters or search query.</div>
        </div>
      `;
      return;
    }

    grid.innerHTML = this.filtered.map(item => this.createCardHtml(item)).join("");

    // Add expand/collapse click handlers
    grid.querySelectorAll(".evidence-card").forEach(card => {
      card.addEventListener("click", (e) => {
        if (e.target.tagName !== "A") {
          card.classList.toggle("expanded");
        }
      });
    });
  }

  createCardHtml(item) {
    const sourceIcons = {
      sec: "📄",
      github: "🧑‍💻",
      news: "📰",
      reddit: "💬",
      macro: "🌍"
    };

    const credClasses = {
      VERIFIED_OFFICIAL: "credibility-verified",
      HIGH: "credibility-high",
      MEDIUM: "credibility-medium",
      LOW: "credibility-low"
    };

    const sentClasses = {
      BULLISH: "sentiment-bullish",
      BEARISH: "sentiment-bearish",
      NEUTRAL: "sentiment-neutral"
    };

    const icon = sourceIcons[item.sourceType] || "📌";
    const credClass = credClasses[item.credibility] || "credibility-medium";
    const sentClass = sentClasses[item.sentiment] || "sentiment-neutral";
    const confPercent = Math.round(item.confidence * 100);

    return `
      <div class="evidence-card" id="${item.id}">
        <div class="evidence-card-header">
          <div class="evidence-source-tag">
            <span>${icon}</span>
            <span>${item.sourceName}</span>
          </div>
          <span class="ticker-chip" style="font-size:0.75rem">${item.ticker}</span>
        </div>

        <div class="evidence-title">${item.title}</div>

        <div class="evidence-meta-bar">
          <span class="credibility-badge ${credClass}">${item.credibilityLabel}</span>
          <span class="sentiment-chip ${sentClass}">${item.sentiment}</span>
          <span class="t-muted">${item.timestamp.split(" ")[0]}</span>
        </div>

        <div class="evidence-content">${item.content}</div>

        <div class="evidence-footer">
          <div class="confidence-meter" title="Confidence Score: ${confPercent}%">
            <span class="t-xs t-muted">Conf</span>
            <div class="confidence-meter-bar">
              <div class="confidence-meter-fill" style="width:${confPercent}%"></div>
            </div>
            <span class="t-mono t-xs t-cyan">${confPercent}%</span>
          </div>

          <a href="${item.url}" target="_blank" rel="noopener noreferrer" class="evidence-url-link" onclick="event.stopPropagation()">
            Source Proof ↗
          </a>
        </div>
      </div>
    `;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.evidenceExplorer = new EvidenceExplorer();
});
