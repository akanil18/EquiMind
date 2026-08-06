/**
 * EquiMind Navigation Manager — nav.js
 * Handles left nav collapse/expand, active state, mobile menu, keyboard shortcuts
 */

class NavManager {
  constructor() {
    this.sidebar = document.getElementById('nav-sidebar');
    this.collapseBtn = document.getElementById('nav-collapse-btn');
    this.overlay = document.getElementById('nav-overlay');
    this.mobileMenuBtn = document.getElementById('mobile-menu-btn');
    this.collapsed = localStorage.getItem('nav-collapsed') === 'true';
    this.currentPage = window.location.pathname.split('/').pop() || 'index.html';

    this.init();
  }

  init() {
    if (this.collapsed && window.innerWidth > 768) {
      this.sidebar?.classList.add('collapsed');
    }

    this.setActiveItem();
    this.bindEvents();
    this.initKeyboardShortcuts();
  }

  setActiveItem() {
    const navItems = document.querySelectorAll('.nav-item[data-page]');
    navItems.forEach(item => {
      item.classList.remove('active');
      const page = item.getAttribute('data-page');
      if (
        page === this.currentPage ||
        (this.currentPage === '' && page === 'index.html') ||
        (this.currentPage === '/' && page === 'index.html')
      ) {
        item.classList.add('active');
      }
    });
  }

  toggleCollapse() {
    this.collapsed = !this.collapsed;
    this.sidebar?.classList.toggle('collapsed', this.collapsed);
    localStorage.setItem('nav-collapsed', this.collapsed);
  }

  openMobile() {
    this.sidebar?.classList.add('mobile-open');
    this.overlay?.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  closeMobile() {
    this.sidebar?.classList.remove('mobile-open');
    this.overlay?.classList.remove('active');
    document.body.style.overflow = '';
  }

  bindEvents() {
    this.collapseBtn?.addEventListener('click', () => this.toggleCollapse());
    this.mobileMenuBtn?.addEventListener('click', () => this.openMobile());
    this.overlay?.addEventListener('click', () => this.closeMobile());

    // Nav item clicks — navigate to page
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
      item.addEventListener('click', (e) => {
        const page = item.getAttribute('data-page');
        if (page) {
          // close mobile menu if open
          this.closeMobile();
        }
      });
    });
  }

  initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + \ to toggle nav
      if ((e.ctrlKey || e.metaKey) && e.key === '\\') {
        e.preventDefault();
        if (window.innerWidth > 768) {
          this.toggleCollapse();
        } else {
          if (this.sidebar?.classList.contains('mobile-open')) {
            this.closeMobile();
          } else {
            this.openMobile();
          }
        }
      }
      // Escape to close mobile
      if (e.key === 'Escape') {
        this.closeMobile();
      }
    });
  }
}

// Toast notification system
class ToastManager {
  constructor() {
    this.container = document.getElementById('toast-container');
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = 'toast-container';
      this.container.className = 'toast-container';
      document.body.appendChild(this.container);
    }
  }

  show(message, type = 'info', duration = 4000) {
    const icons = { info: '💡', success: '✅', warning: '⚠️', error: '❌', research: '🔬' };
    const borderColors = {
      info: 'var(--border-strong)',
      success: 'rgba(16, 185, 129, 0.3)',
      warning: 'rgba(245, 158, 11, 0.3)',
      error: 'rgba(239, 68, 68, 0.3)',
      research: 'rgba(0, 212, 255, 0.3)',
    };

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.style.borderColor = borderColors[type] || borderColors.info;
    toast.innerHTML = `
      <span style="font-size:1rem;flex-shrink:0">${icons[type] || '💡'}</span>
      <span style="font-size:0.875rem;line-height:1.5;color:var(--text-primary);flex:1">${message}</span>
      <button onclick="this.closest('.toast').remove()" style="color:var(--text-muted);font-size:1rem;flex-shrink:0;padding:2px;line-height:1">×</button>
    `;

    this.container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 200);
    }, duration);

    return toast;
  }

  success(msg, dur) { return this.show(msg, 'success', dur); }
  error(msg, dur)   { return this.show(msg, 'error', dur); }
  warning(msg, dur) { return this.show(msg, 'warning', dur); }
  research(msg, dur){ return this.show(msg, 'research', dur); }
}

// Particle Canvas — animated background for hero
class ParticleCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.animFrame = null;
    this.resize();
    this.createParticles();
    this.animate();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    if (!this.canvas) return;
    this.canvas.width  = this.canvas.offsetWidth;
    this.canvas.height = this.canvas.offsetHeight;
  }

  createParticles() {
    const count = Math.floor((this.canvas.width * this.canvas.height) / 12000);
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        r: Math.random() * 1.5 + 0.5,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        alpha: Math.random() * 0.5 + 0.2,
        color: Math.random() > 0.6 ? '#00D4FF' : '#7C3AED',
      });
    }
  }

  animate() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    this.particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();
    });

    // Draw connecting lines between close particles
    ctx.globalAlpha = 1;
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const a = this.particles[i];
        const b = this.particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = 'rgba(0, 212, 255, ' + (0.12 * (1 - dist / 120)) + ')';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    this.animFrame = requestAnimationFrame(() => this.animate());
  }

  destroy() {
    cancelAnimationFrame(this.animFrame);
  }
}

// Instantiate ToastManager globally immediately
window.toast = new ToastManager();

// ── Init on DOM ready ──
document.addEventListener('DOMContentLoaded', () => {
  window.nav = new NavManager();
  window.toast = new ToastManager();

  // Hero particle canvas
  if (document.getElementById('hero-canvas')) {
    window.heroCanvas = new ParticleCanvas('hero-canvas');
  }

  // Animate elements on scroll via IntersectionObserver
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-fade-in');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('[data-animate]').forEach(el => observer.observe(el));
});

// Export for module usage
if (typeof module !== 'undefined') {
  module.exports = { NavManager, ToastManager, ParticleCanvas };
}
