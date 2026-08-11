/* ==========================================================
   FlexBot · 主交互
   - 顶部 HUD 时钟
   - 数字滚动动画
   - 滚动渐入
   - 锚点平滑滚动增强
   ========================================================== */

(function () {
  "use strict";

  // ---------- HUD 时钟 ----------
  function updateClock() {
    const el = document.getElementById("hud-clock");
    if (!el) return;
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    el.textContent =
      pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds());
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ---------- 数字滚动动画 ----------
  function animateNumber(el) {
    const target = parseFloat(el.dataset.target);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const duration = 1400;
    const start = performance.now();
    const startVal = 0;
    const prefix = el.dataset.prefix || "";
    const suffix = el.dataset.suffix || "";

    function tick(now) {
      const elapsed = now - start;
      const t = Math.min(elapsed / duration, 1);
      // ease-out-quart
      const eased = 1 - Math.pow(1 - t, 4);
      const current = startVal + (target - startVal) * eased;
      el.textContent = prefix + current.toFixed(decimals) + suffix;
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        el.textContent = prefix + target.toFixed(decimals) + suffix;
      }
    }
    requestAnimationFrame(tick);
  }

  // ---------- 滚动渐入 ----------
  function setupReveal() {
    const targets = document.querySelectorAll(
      ".section__header, .pain-card, .module, .flow-step, .value-card, .stack-group, .arch-wrap, .callout, .flow-summary, .value-summary, .hero__inner > *, .algo-card, .breakthrough, .arch-detail__cell, .pipeline-step, .tech-stat"
    );

    targets.forEach((el) => {
      el.classList.add("reveal");
    });

    if (!("IntersectionObserver" in window)) {
      targets.forEach((el) => el.classList.add("is-visible"));
      return;
    }

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -60px 0px" }
    );

    targets.forEach((el) => io.observe(el));
  }

  // ---------- 数字滚动触发 ----------
  function setupNumberAnimation() {
    const numbers = document.querySelectorAll(".num");
    if (!("IntersectionObserver" in window)) {
      numbers.forEach(animateNumber);
      return;
    }

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateNumber(entry.target);
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.5 }
    );

    numbers.forEach((n) => io.observe(n));
  }

  // ---------- 锚点平滑滚动 ----------
  function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((link) => {
      link.addEventListener("click", (e) => {
        const href = link.getAttribute("href");
        if (!href || href === "#") return;
        const target = document.querySelector(href);
        if (!target) return;
        e.preventDefault();
        const top = target.getBoundingClientRect().top + window.pageYOffset - 36; // 减去 HUD 高度
        window.scrollTo({ top, behavior: "smooth" });
      });
    });
  }

  // ---------- 启动 ----------
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      setupReveal();
      setupNumberAnimation();
      setupSmoothScroll();
    });
  } else {
    setupReveal();
    setupNumberAnimation();
    setupSmoothScroll();
  }
})();
